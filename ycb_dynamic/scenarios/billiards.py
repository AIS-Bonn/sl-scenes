"""
 Billiards Scenario: A ball smashes through a bunch of objects placed in a billiards-triangle manner
"""
import torch
import random
from copy import deepcopy

import ycb_dynamic.utils.utils as utils
from ycb_dynamic.CONFIG import CONFIG
import ycb_dynamic.CONSTANTS as CONSTANTS
from ycb_dynamic.camera import Camera
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
        self.mesh_loader.load_meshes(CONSTANTS.TABLE),
        self.mesh_loader.load_meshes(CONSTANTS.BOWLING_BALL),
        self.mesh_loader.load_meshes(CONSTANTS.BILLIARDS_OBJECTS),

    def setup_objects_(self):
        """
        SCENARIO-SPECIFIC
        """
        table_info_mesh, bowling_ball_info_mesh, billards_obj_info_mesh = self.mesh_loader.get_meshes()

        # place table
        table_mod = {"mod_pose": CONSTANTS.TABLE_POSE}
        self.table = self.add_object_to_scene(table_info_mesh, True, **table_mod)
        self.z_offset = self.table.pose()[2, -1]

        # assemble several objects in a triangle-like shape
        N = len(CONSTANTS.BILLIARDS_TRIANLGE_POSES)
        obj_poses = deepcopy(CONSTANTS.BILLIARDS_TRIANLGE_POSES)
        for i, obj_info_mesh in enumerate(random.choices(billards_obj_info_mesh, k=N)):
            mod_pose = obj_poses[i]
            mod_pose[2, -1] += self.z_offset
            obj_mod = {"mod_pose": mod_pose}
            obj = self.add_object_to_scene(obj_info_mesh, False, **obj_mod)
            if self.is_there_collision():  # removing last object if colliding with anything else
                self.remove_obj_from_scene(obj)

        # add the bowling_ball with custom position and velocity
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
        self.bowling_ball = self.add_object_to_scene(bowling_ball_info_mesh, False, **bowling_mod)

    def setup_cameras(self):
        print("camera setup...")
        self.cameras = []
        self.cameras.append(Camera("main", CONSTANTS.CAM_POS, CONSTANTS.CAM_LOOKAT, moving=False))

    def simulate(self, dt):
        self.scene.simulate(dt)
        self.sim_t += dt
