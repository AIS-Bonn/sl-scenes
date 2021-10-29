"""
DEPRECATED
Bowl Scenario: Several balls/fruits are placed inside a bowl, approaching and colliding with each other.
"""
import random
import torch
from copy import deepcopy

import ycb_dynamic.CONSTANTS as CONSTANTS
from ycb_dynamic.CONFIG import CONFIG
from ycb_dynamic.camera import Camera
from ycb_dynamic.scenarios.scenario import Scenario


class CollisionScenario(Scenario):
    def __init__(self, cfg, scene):
        self.name = "Collision"
        self.config = CONFIG["scenes"].get("collision", {})
        self.prep_time = 0.002  # during this time (in s), the scene will not be rendered
        super(CollisionScenario, self).__init__(cfg, scene)

    def can_render(self):
        """
        :return: True if scene has been prepared and can be rendered, False otherwise.
        """
        return self.sim_t > self.prep_time

    def load_meshes_(self):
        """
        SCENARIO-SPECIFIC
        """
        self.mesh_loader.load_meshes(CONSTANTS.BOWL),
        self.mesh_loader.load_meshes(CONSTANTS.YCBV_OBJECTS),

    def setup_objects_(self):
        """
        SCENARIO-SPECIFIC
        """
        bowl_info_mesh, ycbv_info_meshes = self.mesh_loader.get_meshes()

        # place bowl
        bowl_pose = deepcopy(CONSTANTS.BOWL_POSE)
        bowl_pose[2, -1] += self.z_offset
        bowl_mod = {"mod_pose": bowl_pose}
        self.bowl = self.add_object_to_scene(bowl_info_mesh, True, **bowl_mod)

        # drop 10 random YCB-Video objects onto the table
        for ycbv_info_mesh in random.choices(ycbv_info_meshes, k=10):
            mod_t = torch.tensor([
                random.uniform(CONSTANTS.DROP_LIMITS["x_min"], CONSTANTS.DROP_LIMITS["x_max"]),
                random.uniform(CONSTANTS.DROP_LIMITS["y_min"], CONSTANTS.DROP_LIMITS["y_max"]),
                random.uniform(CONSTANTS.DROP_LIMITS["z_min"], CONSTANTS.DROP_LIMITS["z_max"])
            ])
            obj_mod = {"mod_t": mod_t}
            obj = self.add_object_to_scene(ycbv_info_mesh, False, **obj_mod)

            # removing last object if colliding with anything else
            if self.is_there_collision():
                self.remove_obj_from_scene(obj)

    def _add_pose_noise(self, pose):
        """ Sampling a noise matrix to add to the pose """
        # TODO: Add rotation noise
        pos_noise = torch.rand(size=(3,)) * self.config["pos_noise_mean"] + self.config["pos_noise_std"]
        pose[:3, -1] += pos_noise
        return pose

    def setup_cameras(self):
        print("camera setup...")
        self.cameras = []
        self.cameras.append(Camera("main", CONSTANTS.CAM_POS, CONSTANTS.CAM_LOOKAT, moving=False))

    def simulate(self, dt):
        self.scene.simulate(dt)
        self.sim_t += dt
