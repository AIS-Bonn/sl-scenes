import stillleben as sl
import os
import random
import torch
import torch.nn.functional as F
from math import ceil, floor
from typing import List

import ycb_dynamic.utils.utils as utils
import ycb_dynamic.CONSTANTS as CONSTANTS
from ycb_dynamic.CONFIG import CONFIG
import ycb_dynamic.OBJECT_INFO as OBJECT_INFO


class MeshLoader:
    """
    Class to load the meshes for the objects in a scene
    """

    def __init__(self):
        """Module initializer"""
        self.base_dir = CONSTANTS.MESH_BASE_DIR
        self.text_dir = CONSTANTS.TEXT_BASE_DIR
        self.reset()

    def reset(self):
        self.class_idx = 0
        self.loaded_meshes = []

    def get_meshes(self):
        """ """
        extract_singular = lambda x: x[0] if len(x) == 1 else x
        return [extract_singular(item) for item in self.loaded_meshes]

    def load_meshes(self, obj_info : List[OBJECT_INFO.ObjectInfo], **kwargs):
        """
        Loads the meshes whose information is given in parameter 'obj_info.
        Each call of this method APPENDS a list to the loaded_meshes attribute.
        :param obj_info: The object information of the meshes to be loaded.
        :param kwargs: additional mesh modifiers such as scale, specified with a leading 'mod_'
        """
        paths = []
        for obj in obj_info:
            path = self.text_dir if obj.name.endswith("_floor") or obj.name.endswith("_wall") else self.base_dir
            paths.append((path / obj.mesh_fp).resolve())
        scales = [obj.scale for obj in obj_info]
        mod_scales = kwargs.get("mod_scale", [1.0] * len(scales))
        scales = [s * ms for (s, ms) in zip(scales, mod_scales)]
        flags = [mesh_flags(obj) for obj in obj_info]
        meshes = sl.Mesh.load_threaded(filenames=paths, flags=flags)

        # Setup class IDs
        for _, (mesh, scale) in enumerate(zip(meshes, scales)):
            pt = torch.eye(4)
            pt[:3, :3] *= scale
            mesh.pretransform = pt
            mesh.class_index = self.class_idx + 1
            self.class_idx += 1

        info_mesh_tuples = list(zip(obj_info, meshes))
        self.loaded_meshes.append(info_mesh_tuples)


def mesh_flags(info: OBJECT_INFO.ObjectInfo):
    if info.flags >= OBJECT_INFO.FLAG_CONCAVE:
        return sl.Mesh.Flag.NONE
    else:
        return sl.Mesh.Flag.PHYSICS_FORCE_CONVEX_HULL


class ObjectLoader:
    """
    Class to load the objects in a scene
    """

    def __init__(self):
        """Module initializer"""
        self.reset()

    def reset(self):
        self.instance_idx = 0
        self.loaded_objects = dict()

    @property
    def static_objects(self):
        return [obj for obj in self.loaded_objects.values() if obj.static]

    @property
    def dynamic_objects(self):
        return [obj for obj in self.loaded_objects.values() if not obj.static]

    def create_object(self, object_info: OBJECT_INFO.ObjectInfo, mesh: sl.Mesh, is_static: bool, **obj_mod):
        """
        Proper object setup
        :param mesh:
        :param object_info:
        :param is_static:
        :param obj_mod: Optional object modifiers, specified with a leading 'mod_'.
            IMPORTANT: scaling is done during mesh loading!!!
        :return:
        """
        ins_idx = self.instance_idx + 1
        self.instance_idx += 1
        obj = sl.Object(mesh)
        mod_weight = obj_mod.get("mod_weight", obj_mod.get("mod_scale", 1.0) ** 3)
        obj.mass = object_info.weight * mod_weight
        obj.metallic = object_info.metallic
        obj.roughness = object_info.roughness
        obj.restitution = object_info.restitution
        pose = obj_mod.get("mod_pose", torch.eye(4))
        mod_R = obj_mod.get("mod_R", torch.eye(3))
        pose[:3, :3] = torch.mm(mod_R, pose[:3, :3])
        mod_t = obj_mod.get("mod_t", torch.tensor([obj_mod.get("mod_x", 0.0),
                                                   obj_mod.get("mod_y", 0.0),
                                                   obj_mod.get("mod_z", 0.0)]))
        pose[:3, 3] += mod_t
        obj.set_pose(pose)
        obj.linear_velocity = obj_mod.get("mod_v_linear", torch.tensor([0.0, 0.0, 0.0]))
        obj.angular_velocity = obj_mod.get("mod_v_angular", torch.tensor([0.0, 0.0, 0.0]))
        obj.static = is_static
        obj.instance_index = ins_idx
        self.loaded_objects[ins_idx] = obj

        return obj

    def remove_object(self, instance_id, decrement_ins_idx=True):
        obj = self.loaded_objects.pop(instance_id, None)
        if decrement_ins_idx and obj is not None:
            self.instance_idx -= 1
        return obj


class DecoratorLoader:
    """
    Class to add random decorative objects to the scene, which do not participate of the scene dynamics.
    It is based on creating an occupancy matrix of the scene, finding empty locations and placing stuff there
    """

    def __init__(self, scene):
        """ Object initializer """
        self.config = CONFIG["decorator"]
        decorations = self.config["decorations"]
        bounds = self.config["bounds"]
        self.bounds = bounds
        self.pi = torch.acos(torch.zeros(1))

        self.scene = scene
        self.mesh_loader = MeshLoader()
        self.mesh_loader.load_meshes(decorations),
        self.meshes = self.mesh_loader.get_meshes()[0]

        self.x_vect = torch.arange(bounds["min_x"], bounds["max_x"] + bounds["res"], bounds["res"])
        self.y_vect = torch.arange(bounds["min_y"], bounds["max_y"] + bounds["res"], bounds["res"])

        return

    def add_object(self, object_loader, object_id):
        """ Loading an object and adding to the loader """
        obj_info, obj_mesh = self.meshes[object_id]
        pose = torch.eye(4)
        obj_mod = {"mod_pose": pose}
        obj = object_loader.create_object(obj_info, obj_mesh, True, **obj_mod)
        self.scene.add_object(obj)

        # shifting object to a free position and adjusting z-coord to be aligned with the table
        position = self.occ_matrix.find_free_spot(obj=obj)
        pose[:2, -1] = position if position is not None else torch.ones(2)
        pose[2, -1] += obj.mesh.bbox.max[-1]

        # Rotating object in yaw direction
        yaw_angle = random.choice([torch.tensor([i* CONSTANTS.PI / 2]) for i in range(4)])
        angles = torch.cat([yaw_angle, torch.zeros(2)])
        rot_matrix = utils.get_rot_matrix(angles=angles)
        pose[:3, :3] = pose[:3, :3] @ rot_matrix

        obj.set_pose(pose)
        self.occ_matrix.update_occupancy_matrix(obj)
        self.occ_matrix.add_object_margings()
        return

    def decorate_scene(self, object_loader):
        """ Randomly adding some decoderation to a scene """
        # initializing occupancy matrix
        self.occ_matrix = OccupancyMatrix(bounds=self.bounds, objects=self.scene.objects)

        # iteratively placing objects while avoiding collision
        N = torch.randint(low=self.config["min_objs"], high=self.config["max_objs"], size=(1,))
        for i in range(N):
            id = torch.randint(low=0, high=len(self.meshes), size=(1,))
            self.add_object(object_loader, object_id=id)

        return


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
            if(os.path.basename(obj.mesh.filename) in CONSTANTS.FLOOR_NAMES):
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
        scaled_width = int(ceil((width * 2 + self.bounds["dist"] + self.bounds["res"]) / self.bounds["res"]) + 1)

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
        min_x, min_y = min(bbox_x_min, -1e-3) + pos_x, min(bbox_y_min, -1e-3) + pos_y
        max_x, max_y = max(bbox_x_max, 1e-3) + pos_x, max(bbox_y_max, 1e-3) + pos_y
        y_coords = (self.grid_y >= min_y) & (self.grid_y < max_y)
        x_coords = (self.grid_x >= min_x) & (self.grid_x < max_x)
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
            print(f"No free positions...")

        return position

#
