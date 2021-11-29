"""
 Billiards Scenario: A ball smashes through a bunch of objects placed in a billiards-triangle manner
"""
import torch
import random
from copy import deepcopy

import ycb_dynamic.utils.utils as utils
from ycb_dynamic.CONFIG import CONFIG
import ycb_dynamic.CONSTANTS as CONSTANTS
from ycb_dynamic.scenarios.scenario import Scenario


class BillardsScenario(Scenario):
    def __init__(self, cfg, scene):
        self.name = "Billards"
        self.config = CONFIG["scenes"]["billards"]
        self.prep_time = 0.002  # during this time (in s), the scene will not be rendered
        super(BillardsScenario, self).__init__(cfg, scene)   # also calls reset_sim()

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
        self.mesh_loader.load_meshes(CONSTANTS.BILLIARDS_OBJECTS)

    def setup_objects_(self):
        """
        SCENARIO-SPECIFIC
        """
        table_info_mesh, bowling_ball_info_mesh, billards_obj_info_mesh = self.mesh_loader.get_meshes()

        # place table
        table_mod = {"mod_pose": CONSTANTS.TABLE_POSE}
        self.table = self.add_object_to_scene(table_info_mesh, True, **table_mod)
        self.table = self.update_object_height(cur_obj=self.table)

        # assemble several objects in a triangle-like shape
        N = len(CONSTANTS.BILLIARDS_TRIANLGE_POSES)
        obj_poses = deepcopy(CONSTANTS.BILLIARDS_TRIANLGE_POSES)
        for i, obj_info_mesh in enumerate(random.choices(billards_obj_info_mesh, k=N)):
            mod_pose = obj_poses[i]
            obj_mod = {"mod_pose": mod_pose}
            obj = self.add_object_to_scene(obj_info_mesh, False, **obj_mod)
            obj = self.update_object_height(cur_obj=obj, objs=[self.table])
            if self.is_there_collision():
                self.remove_obj_from_scene(obj)

        # add the bowling_ball with custom position and velocity
        mod_t = torch.tensor([
            random.uniform(self.config["pos"]["x_min"], self.config["pos"]["x_max"]),
            random.uniform(self.config["pos"]["y_min"], self.config["pos"]["y_max"]),
            random.uniform(self.config["pos"]["z_min"], self.config["pos"]["z_max"])
        ])
        mod_v_linear = utils.get_noisy_vect(
                v=self.config["velocity"]["lin_velocity"],
                mean=self.config["velocity"]["lin_noise_mean"],
                std=self.config["velocity"]["lin_noise_std"]
            )
        bowling_mod = {"mod_t": mod_t, "mod_v_linear": mod_v_linear}
        bowling_ball = self.add_object_to_scene(bowling_ball_info_mesh, False, **bowling_mod)
        _ = self.update_object_height(cur_obj=bowling_ball, objs=[self.table])

    def setup_cameras_(self):
        """
        SCENARIO-SPECIFIC
        """
        self.cameras = [
            self.update_camera_height(camera=cam, objs=[self.table]) for cam in self.cameras
        ]
