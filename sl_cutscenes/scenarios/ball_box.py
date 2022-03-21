"""
BallBoxScenario: Several Balls thrown into a big box, bouncing a bit.
"""
import numpy as np
import torch
import random
from copy import deepcopy

from ycb_dynamic.CONFIG import CONFIG
import ycb_dynamic.CONSTANTS as CONSTANTS
from ycb_dynamic.scenarios.scenario import Scenario


class BallBoxScenario(Scenario):
    def __init__(self, cfg, scene):
        self.name = "Ball Box"
        self.config = CONFIG["scenes"]["ball_box"]
        self.prep_time = 0.0  # during this time (in s), the scene will not be rendered
        super(BallBoxScenario, self).__init__(cfg, scene)   # also calls reset_sim()

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
        self.mesh_loader.load_meshes(CONSTANTS.BALL_BOX)
        self.mesh_loader.load_meshes(CONSTANTS.YCB_SMALL_BALLS)

    def setup_objects_(self):
        """
        SCENARIO-SPECIFIC
        """
        table_info_mesh, wooden_box_info_mesh, balls_info_mesh = self.mesh_loader.get_meshes()

        # place table
        table_mod = {"mod_pose": CONSTANTS.TABLE_POSE}
        self.table = self.add_object_to_scene(table_info_mesh, True, **table_mod)
        self.table = self.update_object_height(cur_obj=self.table)

        # place box
        box_pose = deepcopy(CONSTANTS.BALL_BOX_POSE)
        box_pose[2, -1] += self.z_offset
        box_mod = {"mod_pose": box_pose}
        box = self.add_object_to_scene(wooden_box_info_mesh, True, **box_mod)
        self.box = self.update_object_height(cur_obj=box, objs=[self.table])

        # spawn several balls at random positions in the box
        k = random.randint(self.config["other"]["min_objs"], self.config["other"]["max_objs"])
        obj_placement_angles = np.linspace(0, 2*np.pi, num=self.config["other"]["max_objs"] + 1).tolist()[:-1]
        obj_placement_angles = random.sample(obj_placement_angles, k=k)  # no duplicates
        balls_info_mesh = random.choices(balls_info_mesh, k=k)  # duplicates OK
        for angle, ball_info_mesh in zip(obj_placement_angles, balls_info_mesh):
            ball_pose = deepcopy(CONSTANTS.BALL_BOX_BALL_INIT_POSE)
            ball_x, ball_y = np.sin(angle), np.cos(angle)
            ball_pose[:2, -1] = 0.33 * torch.tensor([ball_x, ball_y])  # assign x and y coordiantes
            ball_v_linear = -1.5 * torch.tensor([ball_x, ball_y, 0.5])
            ball_v_angular = torch.randn(3) * 10
            ball_mod = {"mod_pose": ball_pose, "mod_v_linear": ball_v_linear,
                        "mod_v_angular": ball_v_angular}
            ball = self.add_object_to_scene(ball_info_mesh, False, **ball_mod)
            ball = self.update_object_height(cur_obj=ball, objs=[self.table, self.box], scales=[1.0, 0.35])
            if self.is_there_collision():
                self.remove_obj_from_scene(ball)

    def setup_cameras_(self):
        """
        SCENARIO-SPECIFIC
        """
        self.cameras = [
            self.update_camera_height(camera=cam, objs=[self.table, self.box], scales=[1.0, 0.0])
            for cam in self.cameras
        ]