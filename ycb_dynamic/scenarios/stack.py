"""
Stacked Scenario: Objects with flat surfaces assemble pyramids, which max fall due to lack of support
"""
import random
import numpy as np
import torch

import ycb_dynamic.CONSTANTS as CONSTANTS
from ycb_dynamic.CONFIG import CONFIG
from ycb_dynamic.scenarios.scenario import Scenario


class StackScenario(Scenario):
    def __init__(self, cfg, scene):
        self.name = "Stack"
        self.prep_time = 0.002  # during this time (in s), the scene will not be rendered
        self.config = CONFIG["scenes"]["stack"]
        super(StackScenario, self).__init__(cfg, scene)   # also calls reset_sim()

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
        self.mesh_loader.load_meshes(CONSTANTS.STACK_OBJECTS),

    def setup_objects_(self):
        """
        SCENARIO-SPECIFIC
        """
        table_info_mesh, stack_info_meshes = self.mesh_loader.get_meshes()

        # place table
        table_mod = {"mod_pose": CONSTANTS.TABLE_POSE}
        self.table = self.add_object_to_scene(table_info_mesh, True, **table_mod)
        self.table = self.update_object_height(cur_obj=self.table)

        # randomly sampling the number of stacks, and selecting mean (x,y) center positions
        N_poses = len(CONSTANTS.STACK_PYRAMID_POSES)
        N_stacks = np.random.randint(
                low=self.config["other"]["stacks_min"],
                high=self.config["other"]["stacks_max"] + 1
            )
        x_coords = random.uniform(self.config["pos"]["x_min"], self.config["pos"]["x_max"])
        y_coords = random.uniform(self.config["pos"]["y_min"], self.config["pos"]["y_max"])
        pyramid_centers = torch.zeros(N_stacks, 4, 4)
        pyramid_centers[:, 0,  -1] = x_coords
        pyramid_centers[:, 1, -1] = y_coords

        # assemble the pyramids of stacked objects. Initial poses are noisy, which might lead to pyramid falling
        for n in range(N_stacks):
            for i, obj_info_mesh in enumerate(random.choices(stack_info_meshes, k=N_poses)):
                base_pose = CONSTANTS.STACK_PYRAMID_POSES[i]
                obj_mod = {"mod_pose": base_pose + pyramid_centers[n]}
                obj = self.add_object_to_scene(obj_info_mesh, False, **obj_mod)
                obj = self.update_object_height(cur_obj=obj, objs=[self.table])
                if self.is_there_collision():
                    self.remove_obj_from_scene(obj)

    def setup_cameras_(self):
        """
        SCENARIO-SPECIFIC
        """
        self.cameras = [
            self.update_camera_height(camera=cam, objs=[self.table]) for cam in self.cameras
        ]

    def simulate(self, dt):
        self.scene.simulate(dt)
        self.sim_t += dt
