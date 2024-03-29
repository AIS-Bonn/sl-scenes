"""
Bowling Scenario: A ball smashes through a tower of wooden blocks
"""
import torch
import random
from copy import deepcopy

import sl_cutscenes.utils.utils as utils
from sl_cutscenes.constants import SCENARIO_DEFAULTS
import sl_cutscenes.constants as CONSTANTS
from sl_cutscenes.scenarios.scenario import Scenario

class BowlingScenario(Scenario):
    def __init__(self, cfg, scene):
        self.name = "Bowling"
        self.config = SCENARIO_DEFAULTS["scenes"]["bowling"]
        self.prep_time = 1.000  # during this time (in s), the scene will not be rendered
        self.bowling_ball_loaded = False
        super(BowlingScenario, self).__init__(cfg, scene)   # also calls reset_sim()

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
        self.mesh_loader.load_meshes(CONSTANTS.BOWLING_BALL)
        self.mesh_loader.load_meshes(CONSTANTS.WOOD_BLOCK)

    def setup_objects_(self):
        """
        SCENARIO-SPECIFIC
        """
        table_info_mesh, bowling_ball_info_mesh, wood_block_info_mesh = self.mesh_loader.get_meshes()

        # place table
        table_mod = {"mod_pose": CONSTANTS.TABLE_POSE}
        self.table = self.add_object_to_scene(table_info_mesh, True, **table_mod)
        self.table = self.update_object_height(cur_obj=self.table)

        # assemble a pyramid of wooden blocks
        wood_block_poses = deepcopy(CONSTANTS.WOOD_BLOCK_POSES)
        for i, wood_block_pose in enumerate(wood_block_poses):
            wood_block_mod = {"mod_pose": wood_block_pose}
            wood_block = self.add_object_to_scene(wood_block_info_mesh, False, **wood_block_mod)
            wood_block = self.update_object_height(cur_obj=wood_block, objs=[self.table])
            if self.is_there_collision():
                self.remove_obj_from_scene(wood_block)

        # save bowling ball for later
        self.bowling_ball_info_mesh = bowling_ball_info_mesh

        return

    def add_bowling_ball(self):
        """
        add the bowling_ball with custom position and velocity
        """
        if not self.objects_loaded:
            self.setup_objects()

        mod_t = torch.tensor([
            random.uniform(self.config["pos"]["x_min"], self.config["pos"]["x_max"]),
            random.uniform(self.config["pos"]["y_min"], self.config["pos"]["y_max"]),
            random.uniform(self.config["pos"]["z_min"], self.config["pos"]["z_max"]) + self.z_offset
        ])
        mod_v_linear = utils.get_noisy_vect(
            v=self.config["velocity"]["lin_velocity"],
            mean=self.config["velocity"]["lin_noise_mean"],
            std=self.config["velocity"]["lin_noise_std"]
        )
        bowling_mod = {"mod_t": mod_t, "mod_v_linear": mod_v_linear}
        self.bowling_ball = self.add_object_to_scene(self.bowling_ball_info_mesh, False, **bowling_mod)
        self.bowling_ball = self.update_object_height(cur_obj=self.bowling_ball, objs=[self.table])
        self.bowling_ball_loaded = True

    def setup_cameras_(self):
        """
        SCENARIO-SPECIFIC
        """
        self.cameras = [
            self.update_camera_height(camera=cam, objs=[self.table]) for cam in self.cameras
        ]

    def simulate(self):
        # add bowling ball after preparation time to ensure that the object tower stands still
        if self.sim_t > self.prep_time and not self.bowling_ball_loaded:
            self.add_bowling_ball()
        super(BowlingScenario, self).simulate()
