"""
Modules for assembling realistic rooms for the YCB-Dynamic scenes
"""

import os
import torch
import random

import ycb_dynamic.utils.utils as utils
from ycb_dynamic.objects.mesh_loader import MeshLoader
from ycb_dynamic.objects.object_loader import ObjectLoader
from ycb_dynamic.objects.occupancy_matrix import OccupancyMatrix
import ycb_dynamic.CONSTANTS as CONSTANTS
from ycb_dynamic.CONFIG import CONFIG


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

        self.use_assembled = None
        return

    def make_room(self):
        """ Main logic for obtaining a room for the scene """
        # use_assembled = random.random() < self.config["prob_assembled"]
        self.use_assembled = random.random() < 0
        if(self.use_assembled):
            self.get_existing_room()
        else:
            self.assemble_room()
        return

    def get_existing_room(self):
        """ Fetching one of the preexisting rooms """
        self.mesh_loader.load_meshes(CONSTANTS.ROOM)
        room_info_mesh = self.mesh_loader.get_meshes()[0]
        _ = self.add_object_to_scene(room_info_mesh)
        return

    def assemble_room(self):
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
            # BUG in get_rot_matrix() ?
            rot_matrix = utils.get_rot_matrix(angles=torch.cat([i * self.pi, torch.zeros(2)]))
            pose[:3, :3] = pose[:3, :3] @ rot_matrix
            pose[:2, -1] = torch.Tensor(coords[i])
            cur_wall = self.add_object_to_scene(wall, pose=pose)
            walls.append(cur_wall)

        self.floor, self.walls = floor, walls
        return

    def add_wall_furniture(self):
        """ Adding furniture to the walls, e.g., cabinets and kitchen stuff"""

        if not self.use_assembled:
            # intializing occupancy matrix for collision avoidance
            self.occ_matrix = OccupancyMatrix(
                    bounds=CONFIG["decorator"]["bounds"],
                    objects=self.scene.objects
                )
            n_objs = random.randint(a=3, b=6)  # TODO: get param from CONFIG

            from matplotlib import pyplot as plt
            plt.figure()
            plt.imshow(self.occ_matrix.occ_matrix)
            plt.show()

            for i in range(n_objs):
                self.add_furniture_element(floor=self.floor, walls=self.walls)

            plt.figure()
            plt.imshow(self.occ_matrix.occ_matrix)
            plt.show()

        return

    def add_furniture_element(self, floor, walls):
        """ Adding some pieces of furniture next to the walls"""

        # sampling random wall and random object
        wall_id = random.randint(0, 3)
        wall = walls[wall_id]

        self.mesh_loader.load_meshes([random.choice(CONSTANTS.FURNITURES)])
        furniture_info_mesh = self.mesh_loader.get_meshes()[-1]
        obj = self.add_object_to_scene(furniture_info_mesh)

        # obtaining rotation and location to place object
        wall_pose, max_obj_width = wall.pose(), obj.mesh.bbox.max[1]
        obj_rotated = wall_id in [1, 3]
        x_pos = wall_pose[0, -1] if wall_pose[0, -1] != 0 else None
        y_pos = wall_pose[1, -1] if wall_pose[1, -1] != 0 else None
        rot_matrix = utils.get_rot_matrix(angles=torch.cat([wall_id * -1 * self.pi, torch.zeros(2)]))

        # getting matrix with potential positions for oject
        end_x, end_y = None, None
        if x_pos is not None:
            end_x = False if x_pos < 0 else True
        if y_pos is not None:
            end_y = False if y_pos < 0 else True
        restriction_matrix = self.occ_matrix.get_restriction_matrix(width=max_obj_width, end_x=end_x, end_y=end_y)

        # finding possible position in occupancy matrix
        position = self.occ_matrix.find_free_spot(obj=obj, restriction=restriction_matrix, rotated=obj_rotated)
        if position is None:
            self.remove_object_from_scene(obj)
        else:
            # Adjusting object pose by translating and applying corresponding rotation
            x_pos, y_pos = position
            pose = obj.pose()
            pose[:3, :3] = pose[:3, :3] @ rot_matrix
            pose[0, -1] = pose[0, -1] + x_pos
            pose[1, -1] = pose[1, -1] + y_pos
            obj.set_pose(pose)

        self.occ_matrix.update_occupancy_matrix(obj)
        self.occ_matrix.add_object_margings()
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

    def remove_object_from_scene(self, obj):
        """ Removing an object from the scene """
        self.scene.remove_object(obj)
        self.object_loader.remove_object(obj.instance_index)
        return

    def adjust_z_coord(self, obj, pose):
        """ Adjusting the object z coordinate"""
        pose[2, -1] += (obj.mesh.bbox.max[-1] - obj.mesh.bbox.min[-1]) / 2
        if(os.path.basename(obj.mesh.filename) == "kings_room.obj"):  # TODO: move object-specific offsets to CONSTANTS
            pose[2, -1] = pose[2, -1] - 0.45
        return pose
#
