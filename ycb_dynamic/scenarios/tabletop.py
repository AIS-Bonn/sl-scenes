"""
Tabletop Scenario: Objects are left to free-fall on top of a table
"""
import random
import torch

import ycb_dynamic.CONSTANTS as CONSTANTS
from ycb_dynamic.camera import Camera
from ycb_dynamic.scenarios.scenario import Scenario


# NOTE: Currently not being used
class TabletopScenario(Scenario):
    def __init__(self, cfg, scene):
        self.name = "Tabletop"
        self.prep_time = 0.002  # during this time (in s), the scene will not be rendered
        super(TabletopScenario, self).__init__(cfg, scene)   # also calls reset_sim()

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
        self.z_offset = self.table.pose()[2, -1]

        # drop 10 random YCB-Video objects onto the table
        for obj_info_mesh in random.choices(ycbv_info_meshes, k=10):
            mod_t = torch.tensor([
                random.uniform(self.config["pos"]["x_min"], self.config["pos"]["x_max"]),
                random.uniform(self.config["pos"]["y_min"], self.config["pos"]["y_max"]),
                random.uniform(self.config["pos"]["z_min"], self.config["pos"]["z_max"]) + self.z_offset
            ])
            obj_mod = {"mod_t": mod_t}
            obj = self.add_object_to_scene(obj_info_mesh, False, **obj_mod)

            # removing last object if colliding with anything else
            if self.is_there_collision():
                self.remove_obj_from_scene(obj)

    def setup_cameras(self):
        print("camera setup...")
        self.cameras = []
        self.cameras.append(Camera("main", CONSTANTS.CAM_POS, CONSTANTS.CAM_LOOKAT, moving=False))

    def simulate(self, dt):
        self.scene.simulate(dt)
        self.sim_t += dt
