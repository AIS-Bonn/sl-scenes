"""
Bowl Scenario: Several balls/fruits are placed inside a bowl, approaching and colliding with each other.
"""
import numpy as np
import torch
import random
from copy import deepcopy

from sl_cutscenes.constants import SCENARIO_DEFAULTS
import sl_cutscenes.constants as CONSTANTS
from sl_cutscenes.scenarios.scenario import Scenario


class BowlScenario(Scenario):
    def __init__(self, cfg, scene):
        self.name = "Bowl"
        self.config = SCENARIO_DEFAULTS["scenes"]["bowl"]
        self.prep_time = 0.0  # during this time (in s), the scene will not be rendered
        super(BowlScenario, self).__init__(cfg, scene)   # also calls reset_sim()

    def can_render(self):
        """
        :return: True if scene has been prepared and can be rendered, False otherwise.
        """
        return self.sim_t > self.prep_time

    def load_meshes_(self):
        """
        SCENARIO-SPECIFIC
        """
        self.mesh_loader.load_meshes(CONSTANTS.TABLE)
        self.mesh_loader.load_meshes(CONSTANTS.BOWL, **{"mod_scale": [4.0]})  # scale bowl by 5
        self.mesh_loader.load_meshes(CONSTANTS.FRUIT_OBJECTS)

    def setup_objects_(self):
        """
        SCENARIO-SPECIFIC
        """
        table_info_mesh, wooden_bowl_info_mesh, fruits_info_mesh = self.mesh_loader.get_meshes()

        # place table
        table_mod = {"mod_pose": CONSTANTS.TABLE_POSE}
        self.table = self.add_object_to_scene(table_info_mesh, True, **table_mod)
        self.table = self.update_object_height(cur_obj=self.table)

        # place bowl
        bowl_pose = deepcopy(CONSTANTS.WOODEN_BOWL_POSE)
        bowl_pose[2, -1] += self.z_offset
        bowl_mod = {"mod_pose": bowl_pose}
        bowl = self.add_object_to_scene(wooden_bowl_info_mesh, True, **bowl_mod)
        self.bowl = self.update_object_height(cur_obj=bowl, objs=[self.table])

        # spawn several balls at random positions in the bowl
        k = random.randint(self.config["other"]["min_objs"], self.config["other"]["max_objs"])
        obj_placement_angles = np.linspace(0, 2*np.pi, num=self.config["other"]["max_objs"] + 1).tolist()[:-1]
        obj_placement_angles = random.sample(obj_placement_angles, k=k)  # no duplicates
        fruits_info_mesh = random.choices(fruits_info_mesh, k=k)  # duplicates OK
        for angle, fruit_info_mesh in zip(obj_placement_angles, fruits_info_mesh):
            fruit_pose = deepcopy(CONSTANTS.BOWL_FRUIT_INIT_POS)
            fruit_pose[:2, -1] = 0.33 * torch.tensor([np.sin(angle), np.cos(angle)])  # assign x and y coordiantes
            fruit_mod = {"mod_pose": fruit_pose}
            fruit = self.add_object_to_scene(fruit_info_mesh, False, **fruit_mod)
            fruit = self.update_object_height(cur_obj=fruit, objs=[self.table, self.bowl], scales=[1.0, 0.35])
            if self.is_there_collision():
                self.remove_obj_from_scene(fruit)

    def setup_cameras_(self):
        """
        SCENARIO-SPECIFIC
        """
        self.cameras = [
            self.update_camera_height(camera=cam, objs=[self.table, self.bowl], scales=[1.0, 0.0])
            for cam in self.cameras
        ]