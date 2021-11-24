"""
Tidy Scenario: A robot uses its hand to tidy up the table, pushing the objects into a bin/cart etc.
EE = 'End Effector', which is the gripper / suction cup
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
        self.remaining_pause = 0.000  # pause time remaining for the gripper
        self.max_waypoint_deviation = 0.02  # in m
        self.max_velocity = 0.6  # in m/s
        self.acceleration = 1.5  # in m/s²
        self.ee = None
        self.robot_sim = None
        super(TidyScenario, self).__init__(cfg, scene)   # also calls reset_sim()

    @property
    def ee_pose(self):
        return self.ee.pose()

    @property
    def ee_t(self):
        return self.ee.pose()[:3, 3]

    def can_render(self):
        """
        :return: True if scene has been prepared and can be rendered, False otherwise.
        """
        return self.sim_t > self.prep_time and self.ee is not None

    def load_meshes_(self):
        """
        SCENARIO-SPECIFIC
        """
        self.mesh_loader.load_meshes(CONSTANTS.NO_POOL_TABLE)
        self.mesh_loader.load_meshes(CONSTANTS.YCBV_OBJECTS)
        self.mesh_loader.load_meshes(CONSTANTS.SUCTION_GRIPPER)

    def setup_objects_(self):
        """
        SCENARIO-SPECIFIC
        """
        table_info_mesh, ycbv_info_meshes, self.ee_mesh = self.mesh_loader.get_meshes()

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

        # set up end effector (ee)
        ee_pose = CONSTANTS.END_EFFECTOR_POSE
        init_z = random.uniform(self.config["endeffector_pos"]["z_min"], self.config["endeffector_pos"]["z_max"])
        ee_t = torch.tensor([
            self.config["endeffector_pos"]["x"],
            self.config["endeffector_pos"]["y_1"] if random.random() < 0.5
                else self.config["endeffector_pos"]["y_2"],
            init_z
        ])
        ee_pose[:3, 3] = ee_t
        ee_mod = {"mod_pose": ee_pose}
        self.ee = self.add_object_to_scene(self.ee_mesh, is_static=True, **ee_mod)
        self.ee = self.update_object_height(cur_obj=self.ee, objs=[self.table])
        self.table_height = self.ee.pose()[2, 3] - init_z
        self.ee_velocity = 0.0

        # set up the waypoints the ee has to reach
        self.waypoints = [
            torch.tensor([
                random.uniform(self.config["waypoint_pos"]["x_min"], self.config["waypoint_pos"]["x_max"]),
                self.ee_t[1],
                random.uniform(self.config["waypoint_pos"]["z_min"], self.config["waypoint_pos"]["z_max"])
                + self.table_height,
            ]),
            torch.tensor([
                random.uniform(self.config["waypoint_pos"]["x_min"], self.config["waypoint_pos"]["x_max"]),
                self.ee_t[1] * -1,
                random.uniform(self.config["waypoint_pos"]["z_min"], self.config["waypoint_pos"]["z_max"])
                + self.table_height,
            ]),
        ]

        # set up the robot simulation
        #self.robot_sim = sl.ManipulationSim(self.scene, self.ee, self.ee_pose)
        #self.robot_sim.set_spring_parameters(1000.0, 1.0, 30.0)  # stiffness, damping, force_limit

    def setup_cameras_(self):
        """
        SCENARIO-SPECIFIC
        """
        # TODO set camera to robot
        self.cameras = [
            self.update_camera_height(camera=cam, objs=[self.table]) for cam in self.cameras
        ]

    def simulate(self, dt):

        self.sim_t += dt
        # add robot after preparation time to ensure that the objects are not falling anymore
        if self.sim_t > self.prep_time and self.ee is None:
            self.setup_robot_sim()

        #print(self.remaining_pause)

        # simulate pbject physics without gripper
        if self.ee is None or self.remaining_pause > 0 or len(self.waypoints) < 1:
            self.scene.simulate(dt)
            if self.remaining_pause > 0:
                self.remaining_pause -= dt

        # move gripper to next waypoint of there is
        else:
            cur_waypoint = self.waypoints[0]
            pose_delta = cur_waypoint - self.ee_t
            pose_delta_norm = torch.linalg.norm(pose_delta)
            pose_delta_normalized = pose_delta / pose_delta_norm

            # reached current waypoint -> pop it and stop briefly
            if pose_delta_norm < self.max_waypoint_deviation:
                #print("reached a waypoint")
                _ = self.waypoints.pop(0)
                self.ee_velocity = 0.0
                self.remaining_pause += 0.300  # pause for a bit
            # else: adjust movement velocity according to distance to waypoint
            else:
                ideal_velocity = pose_delta_norm * 2.0
                acceleration = self.acceleration * dt
                if self.ee_velocity < ideal_velocity:
                    self.ee_velocity = min(self.ee_velocity + acceleration, self.max_velocity)
                elif self.ee_velocity >= ideal_velocity:
                    self.ee_velocity = max(self.ee_velocity - acceleration, 0.0)
                #print(self.ee_velocity)

            # calculate new gripper pose with calculated delta vector and velocity
            ee_pose = self.ee_pose
            ee_pose[:3, 3] += self.ee_velocity * dt * pose_delta_normalized
            #print(f"new pos: {ee_pose[:3, 3]}")
            #self.robot_sim.step(ee_pose, dt)  # TODO doesn't work!
            self.ee.set_pose(ee_pose)
            # print(f"after sim: {self.ee_pose[:3, 3]}")
