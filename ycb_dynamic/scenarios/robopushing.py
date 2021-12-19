"""
Robopushing Scenario: Objects free-fall on top of a table, two robots push them around
"""
import random
import torch
import nimblephysics as nimble
import pathlib
import stillleben as sl
from scipy.spatial.transform import Rotation as R
from itertools import chain

import ycb_dynamic.CONSTANTS as CONSTANTS
from ycb_dynamic.CONFIG import CONFIG
import ycb_dynamic.OBJECT_INFO as OBJECT_INFO
from ycb_dynamic.scenarios.robot_scenario import RobotScenario
import ycb_dynamic.utils.utils as utils


class RobopushingScenario(RobotScenario):
    def __init__(self, cfg, scene):
        self.name = "Robopushing"
        self.config = CONFIG["scenes"]["robopushing"]
        self.prep_time = 0.000  # during this time (in s), the scene will not be rendered
        super(RobopushingScenario, self).__init__(cfg, scene)   # also calls reset_sim()

    def can_render(self):
        """
        :return: True if scene has been prepared and can be rendered, False otherwise.
        """
        return self.sim_t > self.prep_time

    def load_meshes_(self):
        """
        SCENARIO-SPECIFIC
        """
        self.mesh_loader.load_meshes(CONSTANTS.NO_POOL_TABLE)
        self.mesh_loader.load_meshes(CONSTANTS.YCBV_OBJECTS)

    def setup_objects_(self):
        """
        SCENARIO-SPECIFIC
        """
        table_info_mesh, ycbv_info_meshes = self.mesh_loader.get_meshes()

        # place table
        table_mod = {"mod_pose": CONSTANTS.TABLE_POSE}
        self.table = self.add_object_to_scene(table_info_mesh, True, **table_mod)
        self.table = self.update_object_height(cur_obj=self.table)
        self.z_offset = self.table.pose()[2, -1]
        
        # add table and other static sl objects to nimble world - must be called
        self.add_static_sl_to_nimble()
        
        # place two robots on the table
        robot1 = self.nimble_world.loadSkeleton('/assets/KR5/KR5.urdf')
        #robot1.enableSelfCollisionCheck()
        x_1 = 0.9 * self.config["pos"]["x_min"] + 0.1 * self.config["pos"]["x_max"]
        y_1 = 0.5 * self.config["pos"]["y_min"] + 0.5 * self.config["pos"]["y_max"]
        utils.set_root_offset(robot1, [x_1, self.z_offset+0.39, -y_1])
        robot1.setPositions([0, 140*(3.1415/180), -115*(3.1415/180), 0, 0, 0])
        
        robot2 = self.nimble_world.loadSkeleton('/assets/KR5/KR5.urdf')
        #robot2.enableSelfCollisionCheck()
        x_2 = 0.1 * self.config["pos"]["x_min"] + 0.9 * self.config["pos"]["x_max"]
        y_2 = 0.5 * self.config["pos"]["y_min"] + 0.5 * self.config["pos"]["y_max"]
        utils.set_root_offset(robot2, [x_2, self.z_offset+0.39, -y_2])
        robot2.setPositions([150*(3.1415/180), 140*(3.1415/180), -115*(3.1415/180), 0, 0, 0])
        
        # self.robots must be populated before calling add_robots_nimble_to_sl()
        self.robots = [robot1, robot2]
        
        # must be called! must be called after add_static_sl_to_nimble()
        self.add_robots_nimble_to_sl()
        
        # drop some random YCB-Video objects onto the table
        for obj_info_mesh in random.choices(ycbv_info_meshes, k=16):
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
            else:
                # must be populated
                self.prop_sl_objects.append(obj)
                
        # must be called! must be called after add_robots_nimble_to_sl()    
        self.add_prop_objects_sl_to_nimble()
        
    def get_action(self):
        """
        SCENARIO-SPECIFIC: Actions are control forces applied on each DOF
        """
        sign = 1
        for robot in self.robots:
            # difficult to determine control forces
            # lazy fix - set velocity state directly
            robot.setVelocities([sign*180*(3.1415/180), 0, 0, 0, 0, 0])
            self.nimble_state = torch.from_numpy(self.nimble_world.getState())
            sign *= -1
        return torch.zeros(self.nimble_world.getActionSize())
    
    def setup_cameras_(self):
        """
        SCENARIO-SPECIFIC
        """
        self.cameras = [
            self.update_camera_height(camera=cam, objs=[self.table]) for cam in self.cameras
        ]
