import random

import torch

from ycb_dynamic import CONSTANTS as CONSTANTS
from ycb_dynamic.CONFIG import CONFIG
from ycb_dynamic.objects.mesh_loader import MeshLoader
from ycb_dynamic.objects.occupancy_matrix import OccupancyMatrix
from ycb_dynamic.utils import utils as utils


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