"""
Tidy Scenario: A robot uses its hand to tidy up the table, pushing the objects into a bin/cart etc.
"""
import torch
import random
from copy import deepcopy
import stillleben as sl

import ycb_dynamic.utils.utils as utils
from ycb_dynamic.CONFIG import CONFIG
import ycb_dynamic.CONSTANTS as CONSTANTS
from ycb_dynamic.scenarios.scenario import Scenario


class TidyScenario(Scenario):
    def __init__(self, cfg, scene):
        self.name = "Tidy"
        self.config = CONFIG["scenes"]["tidy"]
        self.prep_time = 1.000  # during this time (in s), the scene will not be rendered
        self.robot_sim = None
        super(TidyScenario, self).__init__(cfg, scene)   # also calls reset_sim()

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
        self.mesh_loader.load_meshes(CONSTANTS.YCBV_OBJECTS)
        self.mesh_loader.load_meshes(CONSTANTS.SUCTION_GRIPPER)

    def setup_objects_(self):
        """
        SCENARIO-SPECIFIC
        """
        table_info_mesh, ycbv_info_meshes, self.end_effector_mesh = self.mesh_loader.get_meshes()

        # place table
        table_mod = {"mod_pose": CONSTANTS.TABLE_POSE}
        self.table = self.add_object_to_scene(table_info_mesh, True, **table_mod)
        self.table = self.update_object_height(cur_obj=self.table)
        self.z_offset = self.table.pose()[2, -1]

        # drop 10 random YCB-Video objects onto the table
        for obj_info_mesh in random.choices(ycbv_info_meshes, k=3):
            mod_t = torch.tensor([
                random.uniform(self.config["pos"]["x_min"], self.config["pos"]["x_max"]),
                random.uniform(self.config["pos"]["y_min"], self.config["pos"]["y_max"]),
                random.uniform(self.config["pos"]["z_min"], self.config["pos"]["z_max"])
            ])
            obj_mod = {"mod_t": mod_t}
            obj = self.add_object_to_scene(obj_info_mesh, False, **obj_mod)
            obj = self.update_object_height(cur_obj=obj, objs=[self.table])

            # removing last object if colliding with anything else
            if self.is_there_collision():
                self.remove_obj_from_scene(obj)

    def setup_robot_sim(self):
        if not self.objects_loaded:
            self.setup_objects()

        end_effector_mod = {"mod_pose": torch.eye(4)}
        self.end_effector = self.add_object_to_scene(self.end_effector_mesh, is_static=False, **end_effector_mod)
        self.robot_sim = sl.ManipulationSim(self.scene, self.end_effector, self.end_effector.pose)
        self.robot_sim.set_spring_parameters(1000.0, 1.0, 30.0)  # stiffness, damping, force_limit

    def setup_cameras_(self):
        """
        SCENARIO-SPECIFIC
        """
        # TODO set camera to robot
        self.cameras = [
            self.update_camera_height(camera=cam, objs=[self.table]) for cam in self.cameras
        ]

    def simulate(self, dt):
        # add robot after preparation time to ensure that the objects are not falling anymore
        if self.sim_t > self.prep_time and self.robot_sim is None:
            self.setup_robot_sim()

        if self.robot_sim is None:
            self.scene.simulate(dt)
        else:
            self.robot_sim.step(self.end_effector.pose, dt)
        self.sim_t += dt

