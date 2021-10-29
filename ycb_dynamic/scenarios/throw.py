"""
Throw Scenario: Objects are thrown from the side of the table.
Same as tabletop, but with intial linear velocity
"""
import random
import torch

import ycb_dynamic.utils.utils as utils
import ycb_dynamic.CONSTANTS as CONSTANTS
from ycb_dynamic.CONFIG import CONFIG
from ycb_dynamic.camera import Camera
from ycb_dynamic.scenarios.scenario import Scenario


class ThrowScenario(Scenario):
    def __init__(self, cfg, scene):
        self.name = "Throw"
        self.config = CONFIG["scenes"]["throw"]
        self.prep_time = 0  # during this time (in s), the scene will not be rendered
        self.meshes_loaded = False
        self.bowling_ball_loaded = False
        super(ThrowScenario, self).__init__(cfg, scene)   # also calls reset_sim()

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
        self.mesh_loader.load_meshes(CONSTANTS.YCBV_OBJECTS),

    def setup_objects_(self):
        """
        SCENARIO-SPECIFIC
        """
        table_info_mesh, ycbv_info_meshes = self.mesh_loader.get_meshes()

        # place table
        table_mod = {"mod_pose": CONSTANTS.TABLE_POSE}
        self.table = self.add_object_to_scene(table_info_mesh, True, **table_mod)

        # throw some random YCB-Video objects onto the table, from the side
        N_objs = random.randint(self.config["other"]["min_objs"], self.config["other"]["max_objs"] + 1)
        for obj_info_mesh in random.choices(ycbv_info_meshes, k=N_objs):
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
            mod_v_angular = utils.get_noisy_vect(
                    v=self.config["velocity"]["ang_velocity"],
                    mean=self.config["velocity"]["ang_noise_mean"],
                    std=self.config["velocity"]["ang_noise_std"]
            )
            obj_mod = {"mod_t": mod_t, "mod_v_linear": mod_v_linear, "mod_v_angular": mod_v_angular}
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
