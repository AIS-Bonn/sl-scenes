"""
Modules for assembling realistic rooms for the YCB-Dynamic scenes
"""

import os
import torch
import random

import ycb_dynamic.utils.utils as utils
from ycb_dynamic.object_models import MeshLoader, ObjectLoader
import ycb_dynamic.CONSTANTS as CONSTANTS
from ycb_dynamic.CONFIG import CONFIG
import ycb_dynamic.OBJECT_INFO as OBJECT_INFO


class RoomAssembler:
    """
    Module for loading preexisting rooms or for assembling a room using
    floor, walls and objects
    """

    def __init__(self, scene):
        """ Module initializer """
        self.pi = torch.acos(torch.zeros(1))
        self.scene = scene
        self.config = CONFIG["room"]

        self.mesh_loader = MeshLoader()
        self.object_loader = ObjectLoader()

        return

    def make_room(self):
        """ Main logic for obtaining a room for the scene """
        use_assembled = random.random() < 0 #self.config["prob_assembled"]
        if(use_assembled):
            room = self.get_existing_room()
        else:
            room = self.assemble_room()
        return room

    def get_existing_room(self):
        """ Fetching one of the preexisting rooms """
        self.mesh_loader.load_meshes(CONSTANTS.ROOM)
        room_info_mesh = self.mesh_loader.get_meshes()[0]
        room = self.add_object_to_scene(room_info_mesh)
        return room

    def assemble_room(self):
        """ Assembling a custom room """
        floor, walls = self.assemble_structure()
        n_objs = random.randint(a=1, b=5)
        # n_objs = 1
        for i in range(n_objs):
            self.add_wall_furniture(floor, walls)
        return None

    def assemble_structure(self):
        """ Assembling the main structure of the room, including floor and walls """
        self.mesh_loader.load_meshes(CONSTANTS.FLOOR),
        self.mesh_loader.load_meshes(CONSTANTS.WALL * 4),

        # adding floor
        floor_info_mesh, wall_info_mesh = self.mesh_loader.get_meshes()
        floor = self.add_object_to_scene(floor_info_mesh)
        x1, y1, x2, y2 = *floor.mesh.bbox.min[:2], *floor.mesh.bbox.max[:2]
        coords = [[0., y1], [x1, 0.], [0., y2], [x2, 0.]]

        # adding walls
        walls = []
        for i, wall in enumerate(wall_info_mesh):
            pose = torch.eye(4)
            # BUG in get_rot_matrix()
            rot_matrix = utils.get_rot_matrix(angles=torch.cat([i * self.pi, torch.zeros(2)]))
            pose[:3, :3] = pose[:3, :3] @ rot_matrix
            pose[:2, -1] = torch.Tensor(coords[i])
            cur_wall = self.add_object_to_scene(wall, pose=pose)
            walls.append(cur_wall)

        return floor, walls

    def add_wall_furniture(self, floor, walls):
        """ Adding some pieces of furniture next to the walls"""

        # sampling random wall and random object
        wall_id = random.randint(0, 3)
        wall = walls[wall_id]
        self.mesh_loader.load_meshes(CONSTANTS.FURNITURE)
        furniture_info_mesh = self.mesh_loader.get_meshes()[-1]
        obj = self.add_object_to_scene(furniture_info_mesh)

        # obtaining rotation and location to place object
        floor_bbox, wall_bbox, obj_bbox = floor.mesh.bbox, wall.mesh.bbox, obj.mesh.bbox
        wall_pose = wall.pose()
        x_pos = wall_pose[0, -1] if wall_pose[0, -1] != 0 else \
                random.uniform(floor_bbox.min[0], floor_bbox.max[0])
        y_pos = wall_pose[1, -1] if wall_pose[1, -1] != 0 else \
                random.uniform(floor_bbox.min[1], floor_bbox.max[1])
        rot_matrix = utils.get_rot_matrix(angles=torch.cat([wall_id * self.pi, torch.zeros(2)]))

        # Adjusting object pose by translating to wall and applying corresponding rotation
        pose = obj.pose()
        pose[0, -1] = pose[0, -1] + x_pos + obj_bbox.max[0] if x_pos < 0 else \
                      pose[0, -1] + x_pos + obj_bbox.min[0]
        pose[1, -1] = pose[1, -1] + y_pos + obj_bbox.max[1] if y_pos < 0 else \
                      pose[1, -1] + y_pos + obj_bbox.min[1]
        # pose[:3, :3] = pose[:3, :3] @ rot_matrix  # TODO: adapt for rotations
        obj.set_pose(pose)

        self.mesh_loader.reset()
        return

    def add_object_to_scene(self, obj_info_mesh, pose=None):
        """ Adding object to the scene and adjusting the z-component"""
        obj_info, obj_mesh = obj_info_mesh
        obj = self.object_loader.create_object(obj_info, obj_mesh, is_static=True)
        self.scene.add_object(obj)

        pose = torch.eye(4) if pose is None else pose
        pose = self.adjust_z_coord(obj=obj, pose=pose)
        obj.set_pose(pose)
        return obj

    def adjust_z_coord(self, obj, pose):
        """ Adjusting the object z coordinate"""
        pose[2, -1] += (obj.mesh.bbox.max[-1] - obj.mesh.bbox.min[-1]) / 2
        if(os.path.basename(obj.mesh.filename) == "kings_room.obj"):  # TODO: move object-specific offsets to CONSTANTS
            pose[2, -1] = pose[2, -1] - 0.45
        return pose
#
