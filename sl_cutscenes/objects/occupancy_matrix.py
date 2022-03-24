"""
Implementation of a simple (currently 2D) occupancy matrix to fill the scenes with
decorative objects such as chairs, tables and cupboards.
This is also used to avoid collisions.

TODO:
  - Make occupancy matrix 3D voxel-based for further scene understanding
"""

import os
from math import ceil

import torch
from torch.nn import functional as F

from sl_cutscenes.constants import FLOOR_NAMES
from sl_cutscenes.utils import utils as utils


class OccupancyMatrix:
    """
    Module that computes and updates an occupancy matrix of the room
    """

    def __init__(self, bounds, objects=None):
        """ Initializer of the occupancy matrix """
        self.bounds = bounds
        self.x_vect = torch.arange(bounds["min_x"], bounds["max_x"] + bounds["res"], bounds["res"])
        self.y_vect = torch.arange(bounds["min_y"], bounds["max_y"] + bounds["res"], bounds["res"])
        self.grid_y, self.grid_x = torch.meshgrid(self.x_vect, self.y_vect)
        self.occ_matrix = self.get_empty_occ_matrix()

        n_cells = int(bounds["dist"] / bounds["res"]) + 1
        self.margin_kernel = torch.ones(1, 1, n_cells, n_cells) / (n_cells ** 2)
        self.pad = (n_cells//2, n_cells//2, n_cells//2, n_cells//2)

        if objects is not None:
            self.init_occupancy_matrix(objects=objects)
        return

    def init_occupancy_matrix(self, objects):
        """ Obtaining an occupancy matrix with empty and occupied positions"""
        for obj in objects:
            # print(os.path.basename(obj.mesh.filename))
            if os.path.basename(obj.mesh.filename) in FLOOR_NAMES:
                continue
            self.update_occupancy_matrix(obj)
        self.add_object_margings()
        return

    def get_empty_occ_matrix(self):
        """ """
        matrix = torch.zeros(
                int((self.bounds["max_x"] + self.bounds["res"] - self.bounds["min_x"]) / self.bounds["res"]),
                int((self.bounds["max_y"] + self.bounds["res"] - self.bounds["min_y"]) / self.bounds["res"])
            )
        return matrix

    def get_restriction_matrix(self, width=1., end_x=None, end_y=None):
        """
        Obtaining a restriction matrix to place an object. The restriction matrix is a ones-matrix, with
        zeros in the areas where an object can be place.
        Useful to place objects only next to walls.
        """
        matrix = self.get_empty_occ_matrix() + 1
        # scaled_width = int(ceil((width * 2 + self.bounds["dist"] + self.bounds["res"]) / self.bounds["res"]) + 1)
        scaled_width = int(ceil((width * 2) / self.bounds["res"]))

        if(end_x is not None):
            matrix[:, :scaled_width] = 0 if end_x is False else 1
            matrix[:, -scaled_width:] = 0 if end_x else 1
        if(end_y is not None):
            matrix[:scaled_width, :] = 0 if end_y is False else 1
            matrix[-scaled_width:, :] = 0 if end_y else 1

        return matrix

    def update_occupancy_matrix(self, obj):
        """ Updating occupancy matrix given object """
        pose = obj.pose()
        pos_x, pos_y = pose[:2, -1]
        rot_mat = pose[:2, :2]
        angle = utils.get_angle_from_mat(rot_mat, deg=True)
        bbox_x_min, bbox_x_max = obj.mesh.bbox.min[0], obj.mesh.bbox.max[0]
        bbox_y_min, bbox_y_max = obj.mesh.bbox.min[1], obj.mesh.bbox.max[1]
        if(torch.isclose(angle.abs(), torch.tensor([90.]))):  # for rotated objects
            bbox_x_min, bbox_y_min = bbox_y_min, bbox_x_min
            bbox_x_max, bbox_y_max = bbox_y_max, bbox_x_max

        # using the 1e-3 to add some volume to walls
        min_size = self.bounds["res"] / 2
        min_x, min_y = min(bbox_x_min, -min_size) + pos_x, min(bbox_y_min, -min_size) + pos_y
        max_x, max_y = max(bbox_x_max, min_size) + pos_x, max(bbox_y_max, min_size) + pos_y
        y_coords = (self.grid_y >= min_y) & (self.grid_y <= max_y)
        x_coords = (self.grid_x >= min_x) & (self.grid_x <= max_x)
        occ_coords = y_coords & x_coords

        self.occ_matrix[occ_coords] = 1
        return

    def add_object_margings(self):
        """ Adding margin to objects in occupancy matrix. Indicated with value 0.5"""
        orig_pos = self.occ_matrix > 0.5
        self.occ_matrix[self.occ_matrix <= 0.5] = 0
        self.occ_matrix = F.pad(self.occ_matrix, self.pad).unsqueeze(0).unsqueeze(0)
        self.occ_matrix = F.conv2d(self.occ_matrix, self.margin_kernel, stride=1)[0, 0]
        self.occ_matrix[self.occ_matrix > 0] = 0.5
        self.occ_matrix[orig_pos] = 1
        return

    def find_free_spot(self, obj, restriction=None, rotated=False):
        """
        Finding a position in the non-restricted area of the occupancy matrix where the object
        does not collide with anything

        Args:
        -----
        obj: Stillleben Object
            Already loaded object that we want to add to the room
        restriction: Binary Tensor or None
            Indicates additional parts of the occupancy matrix where object cannot be placed.

        Returns:
        --------
        position: torch Tensor
            Location [x, y] where the object can be safely placed
        """

        # obtaining restricted occupancy matrix
        cur_occ_matrix = self.occ_matrix.clone()
        H, W = cur_occ_matrix.shape
        if restriction is not None:
            cur_occ_matrix[restriction > 0] = 1

        # filtering matrix to account for min-distance parameter
        kernel = torch.ceil((obj.mesh.bbox.max[:2] + self.bounds["dist"] + self.bounds["res"]) / self.bounds["res"])
        kernel = kernel.tolist()
        kernel[0] = kernel[0] * 2
        kernel = kernel if not rotated else kernel[::-1]
        for i, k in enumerate(kernel):
            kernel[i] = int(k + 1) if k % 2 == 0 else int(k)
        aux_matrix = F.conv2d(
                cur_occ_matrix.view(1, 1, H, W),
                torch.ones(1, 1, int(kernel[1]), int(kernel[0])),
                padding=(int(kernel[1])//2, int(kernel[0])//2),
            )[0, 0]

        # finding free position, if any
        position = None
        free_positions = torch.where(aux_matrix == 0)
        if(len(free_positions[0]) > 0):
            id = torch.randint(0, len(free_positions[0]), (1,))
            pos_y, pos_x = free_positions[0][id], free_positions[1][id]
            position = torch.cat([self.x_vect[pos_x], self.y_vect[pos_y]])
        else:
            print("No free positions...")

        return position
