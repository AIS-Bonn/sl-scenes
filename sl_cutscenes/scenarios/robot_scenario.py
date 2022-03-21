"""
Abstract scenario subclass for defining robot scenarios
"""
import random
import torch
import nimblephysics as nimble
import pathlib
import stillleben as sl
from itertools import chain

import ycb_dynamic.CONSTANTS as CONSTANTS
from ycb_dynamic.CONFIG import CONFIG
import ycb_dynamic.OBJECT_INFO as OBJECT_INFO
from ycb_dynamic.scenarios.scenario import Scenario
import ycb_dynamic.utils.utils as utils


class RobotScenario(Scenario):
    def __init__(self, cfg, scene):
        assert cfg.physics_engine == 'nimble', "Robot scenarios require nimblephysics sim"
        self.nimble_world = nimble.simulation.World()
        self.nimble_world.setGravity([0, -9.81, 0])
        self.nimble_world.setTimeStep(cfg.sim_dt)
        self.prop_sl_objects = []
        self.sim_steps_per_frame = cfg.sim_steps_per_frame
        self.num_sim_steps = 0
        super(RobotScenario, self).__init__(cfg, scene)   # also calls reset_sim()

    def add_static_sl_to_nimble(self):
        for obj in self.scene.objects:
            obj_info = OBJECT_INFO.get_object_by_class_id(obj.mesh.class_index)
            skel, pos, vel = utils.sl_object_to_nimble(obj, obj_info, debug_mode=self.nimble_debug)
            skel.setPositions(torch.cat(pos))
            self.nimble_world.addSkeleton(skel)
        self.nimble_robot_offset = self.nimble_world.getNumDofs()
        
    def add_robots_nimble_to_sl(self):
        self.robot_sl_objects = [[] for robot in self.robots]
        # add robot meshes to sl scene
        for i, robot in enumerate(self.robots):
            for part in robot.getBodyNodes():
                if part.getShapeNode(0) is not None:
                    # transfer mesh
                    mesh_path = pathlib.Path(part.getShapeNode(0).getShape().getMeshPath())
                    obj_path = utils.stl_to_obj(mesh_path)
                    mesh = sl.Mesh(obj_path)
                    obj = sl.Object(mesh)
                    obj.metallic = 1.0
                    obj.roughness = 0.4
                    # transfer pose
                    pose_mat = part.getWorldTransform().matrix()
                    p = obj.pose()
                    p[:3,3] = utils.P @ pose_mat[:3,3]
                    p[:3,:3] = utils.P @ pose_mat[:3,:3]
                    obj.set_pose(p)
                    self.scene.add_object(obj)
                    self.robot_sl_objects[i].append(obj)
        # the prop descriptions start at this index in the state vector
        self.nimble_prop_offset = self.nimble_world.getNumDofs()
        
    def sync_robots_nimble_to_sl(self):
        for i, robot in enumerate(self.robots):
            for j, part in enumerate(robot.getBodyNodes()):
                # obtain corresponding object
                obj = self.robot_sl_objects[i][j]
                # transfer pose
                pose_mat = part.getWorldTransform().matrix()
                p = obj.pose()
                p[:3,3] = utils.P @ pose_mat[:3,3]
                p[:3,:3] = utils.P @ pose_mat[:3,:3]
                obj.set_pose(p)
        
    def restrict_action_space_to_robots(self):
        # ensure the action space contains only those actions related to the robots
        for dof in chain(range(self.nimble_robot_offset), range(self.nimble_prop_offset, self.nimble_world.getNumDofs())):
            self.nimble_world.removeDofFromActionSpace(dof)
        
    def add_prop_objects_sl_to_nimble(self):
        for obj in self.prop_sl_objects:
            obj_info = OBJECT_INFO.get_object_by_class_id(obj.mesh.class_index)
            skel, pos, vel = utils.sl_object_to_nimble(obj, obj_info, debug_mode=self.nimble_debug)
            skel.setPositions(torch.cat(pos))
            self.nimble_world.addSkeleton(skel)
    
    def sync_prop_objects_nimble_to_sl(self):
        obj_pos, obj_vel = torch.chunk(self.nimble_state.clone(), 2)
        obj_pos = obj_pos[self.nimble_prop_offset:]
        obj_pos = torch.chunk(obj_pos, obj_pos.shape[0] // 6)
        for obj, pos in zip(self.prop_sl_objects, obj_pos):
            obj_pose = obj.pose()
            obj_rot, obj_t = pos.split([3, 3])
            obj_pose[:3, :3] = utils.nimble_to_sl_rot(obj_rot)
            obj_pose[:3,  3] = utils.P @ obj_t
            obj.set_pose(obj_pose)
            
    def reset_sim(self):
        self.meshes_loaded, self.objects_loaded, self.cameras_loaded = False, False, False
        if self.physics_engine == "nimble":
            self.nimble_loaded = False
        self.sim_t = 0
        self.setup_scene()
        self.setup_objects()
        self.restrict_action_space_to_robots()
        self.nimble_state = torch.from_numpy(self.nimble_world.getState())
        self.nimble_loaded = True
        self.setup_cameras()
        # would need to be swapable to front
        # to enable interaction with decoration
        self.decorate_scene()
        return
                
    def get_action(self):
        """ Default action, can be specified by scenario"""
        return torch.zeros(self.nimble_world.getActionSize())
        
    def simulate_nimble_(self, action=None):
        '''
        Simulates a timestep in nimblephysics.
        '''
        # simulate timestep in nimble
        action = self.get_action()
        self.nimble_state = nimble.timestep(self.nimble_world, self.nimble_state, action)
        
        self.num_sim_steps += 1
        if self.num_sim_steps % self.sim_steps_per_frame == 0:
            #transfer robot components to sl context
            self.sync_robots_nimble_to_sl()
            # transfer prop objects to sl context
            self.sync_prop_objects_nimble_to_sl()
            

