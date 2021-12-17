"""
Tabletop Scenario: Objects are left to free-fall on top of a table
"""
import random
import torch
import nimblephysics as nimble
import pathlib
import stillleben as sl

import ycb_dynamic.CONSTANTS as CONSTANTS
from ycb_dynamic.CONFIG import CONFIG
import ycb_dynamic.OBJECT_INFO as OBJECT_INFO
from ycb_dynamic.scenarios.scenario import Scenario
import ycb_dynamic.utils.utils as utils


class RobopushingScenario(Scenario):
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
        self.mesh_loader.load_meshes(CONSTANTS.TABLE)
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
        
        # create nimble world
        self.nimble_world = nimble.simulation.World()
        self.nimble_world.setTimeStep(self.sim_dt)
        
        # place two robots on the table
        robot1 = self.nimble_world.loadSkeleton('/assets/KR5/KR5.urdf')
        robot1.setName("Robot1")
        x_1 = 0.8 * self.config["pos"]["x_min"] + 0.2 * self.config["pos"]["x_max"]
        y_1 = 0.5 * self.config["pos"]["y_min"] + 0.5 * self.config["pos"]["y_max"]
        utils.set_root_offset(robot1, [x_1, self.z_offset+0.39, -y_1])
        
        robot2 = self.nimble_world.loadSkeleton('/assets/KR5/KR5.urdf')
        robot2.setName("Robot2")
        x_2 = 0.2 * self.config["pos"]["x_min"] + 0.8 * self.config["pos"]["x_max"]
        y_2 = 0.5 * self.config["pos"]["y_min"] + 0.5 * self.config["pos"]["y_max"]
        utils.set_root_offset(robot2, [x_2, self.z_offset+0.39, -y_2])
        self.robots = [robot1, robot2]
        
        
        # add robot meshes to sl scene
        for robot in self.robots:
            for part in robot.getBodyNodes():
                if part.getShapeNode(0) is not None:
                    # Transfer Mesh
                    mesh_path = pathlib.Path(part.getShapeNode(0).getShape().getMeshPath())
                    obj_path = utils.stl_to_obj(mesh_path)
                    mesh = sl.Mesh(obj_path)
                    obj = sl.Object(mesh)
                    obj.metallic = 1.0
                    obj.roughness = 0.4
                    #Transfer Pose
                    pose_mat = part.getWorldTransform().matrix()
                    p = obj.pose()
                    p[:3,3] = utils.P @ pose_mat[:3,3]  #P @ pose[-3:] + torch.tensor([0,0,0.5])
                    p[:3,:3] = utils.P @ pose_mat[:3,:3] #P @ torch.from_numpy(R.from_rotvec(pose[:3].numpy()).as_matrix())
                    obj.set_pose(p)
                    self.scene.add_object(obj)

        # drop 10 random YCB-Video objects onto the table
        for obj_info_mesh in random.choices(ycbv_info_meshes, k=10):
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
                
        # sync nimble with sl scene objects
        #self.nimble_loaded = True
                
    def setup_nimble_(self):
        '''
        Creates a clone of the current stillleben scene for nimblephysics, enabling physics simulation there.
        '''
        print("initializing nimble scene from sl...")
        # utils.dump_sl_scene_to_urdf(self.scene, "scene.urdf")
        self.nimble_world = nimble.simulation.World()
        self.nimble_world.setTimeStep(self.sim_dt)
        positions, velocities = [], []
        for obj in self.scene.objects:
            obj_info = OBJECT_INFO.get_object_by_class_id(obj.mesh.class_index)
            skel, pos, vel = utils.sl_object_to_nimble(obj, obj_info, debug_mode=self.nimble_debug)
            self.nimble_world.addSkeleton(skel)
            positions.extend(pos)
            velocities.extend(vel)
        self.nimble_states = [torch.cat(positions + velocities)]
        self.nimble_loaded = True
        
    def simulate_nimble_(self, action=None):
        '''
        Simulates a timestep in nimblephysics.
        '''
        # simulate timestep in nimble
        if action is None:
            action = torch.zeros(self.nimble_world.getNumDofs())
        new_state = nimble.timestep(self.nimble_world, self.nimble_states[-1], action)
        self.nimble_states.append(new_state)
        self.nimble_world.setState(new_state)

        # transfer object state back into the stillleben context
        obj_pos, obj_vel = torch.chunk(new_state.clone(), 2)
        obj_pos = torch.chunk(obj_pos, obj_pos.shape[0] // 6)
        obj_vel = torch.chunk(obj_vel, obj_vel.shape[0] // 6)
        for obj, pos, vel in zip(self.scene.objects, obj_pos, obj_vel):
            obj_pose = obj.pose()
            obj_rpy, obj_t = pos.split([3, 3])
            obj_pose[:3, :3] = utils.get_mat_from_rpy(obj_rpy)
            obj_pose[:3,  3] = obj_t
            obj.set_pose(obj_pose)
            angular_velocity, obj.linear_velocity = vel.split([3, 3])
            obj.angular_velocity = angular_velocity.flip(0)  # flip back from ZYX convention

    def setup_cameras_(self):
        """
        SCENARIO-SPECIFIC
        """
        self.cameras = [
            self.update_camera_height(camera=cam, objs=[self.table]) for cam in self.cameras
        ]
