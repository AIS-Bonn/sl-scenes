"""
Abstract class for defining scenarios
"""
import random
from typing import Tuple
import numpy as np
from copy import deepcopy
import torch
import stillleben as sl
import nimblephysics as nimble

from ycb_dynamic.room_models import RoomAssembler
from ycb_dynamic.objects.mesh_loader import MeshLoader
from ycb_dynamic.objects.object_loader import ObjectLoader
from ycb_dynamic.objects.decorator_loader import DecoratorLoader
from ycb_dynamic.lighting import get_lightmap
from ycb_dynamic.camera import Camera
import ycb_dynamic.utils.utils as utils
import ycb_dynamic.CONSTANTS as CONSTANTS
import ycb_dynamic.OBJECT_INFO as OBJECT_INFO


class Scenario(object):
    """ Abstract class for defining scenarios """

    config = dict()
    name = 'scenario'

    def __init__(self, cfg, scene: sl.Scene, randomize=True):
        self.device = cfg.device
        self.viewer_mode = cfg.viewer
        self.scene = scene
        if randomize:
            utils.randomize()

        self.mesh_loader = MeshLoader()
        self.object_loader = ObjectLoader(scenario_reset=True)
        self.room_assembler = RoomAssembler(scene=self.scene)
        self.decorator_loader = DecoratorLoader(scene=self.scene)

        self.meshes_loaded, self.objects_loaded = False, False
        self.z_offset = 0.
        self.lightmap = cfg.lightmap

        if getattr(self, "allow_multiple_cameras", True):
            self.n_cameras = cfg.cameras
        else:
            print(f"scenario '{self.name}' supports only 1 camera -> ignoring n_cameras...")
            self.n_cameras = 1
        self.coplanar_stereo = cfg.coplanar_stereo
        self.coplanar_stereo_dist = cfg.coplanar_stereo_dist
        self.cam_movement_complexity = cfg.cam_movement_complexity
        self.sim_dt = cfg.sim_dt
        self.cam_dt = cfg.cam_dt
        self.physics_engine = cfg.physics_engine
        self.nimble_debug = cfg.nimble_debug
        self.reset_sim()
        return

    def reset_sim(self):
        self.meshes_loaded, self.objects_loaded, self.cameras_loaded = False, False, False
        if self.physics_engine == "nimble":
            self.nimble_loaded = False
        self.sim_t = 0
        self.setup_scene()
        self.setup_objects()
        self.setup_cameras()
        self.decorate_scene()
        return

    @property
    def all_objects(self):
        return self.object_loader.all_objects

    @property
    def static_objects(self):
        return self.object_loader.static_objects

    @property
    def dynamic_objects(self):
        return self.object_loader.dynamic_objects

    def set_camera_look_at(self, pos, lookat):
        self.scene.set_camera_look_at(position=pos, look_at=lookat)

    def can_render(self):
        raise NotImplementedError

    def decorate_scene(self):
        self.room_assembler.add_wall_furniture()
        self.decorator_loader.decorate_scene(object_loader=self.object_loader)
        return

    def setup_scene(self):
        """ Default setup_scene lighting and camera. Can be overriden from specific scenes """
        self.scene.ambient_light = torch.tensor([0.1, 0.1, 0.1])
        _ = self.room_assembler.make_room()
        self.scene.light_map = get_lightmap(self.lightmap)
        # self.scene.choose_random_light_position()
        return

    def get_separations(self):
        # assert len(self.dynamic_objects) > 0, "Objects must be added to dynamic_objects before computing collisions"
        self.scene.check_collisions()
        separations = [obj.separation for obj in self.dynamic_objects if hasattr(obj, "separation")]
        return separations

    def is_there_collision(self):
        separations = self.get_separations()
        collision = True if np.sum(separations) < 0 else False
        return collision

    def load_meshes(self):
        """ """
        if self.meshes_loaded:
            return
        print("mesh setup...")
        self.load_meshes_()
        self.meshes_loaded = True

    def load_meshes_(self):
        """
        Scenario-specific logic
        """
        raise NotImplementedError

    def setup_objects(self):
        """ """
        if self.objects_loaded:
            return
        print("object setup...")
        if not self.meshes_loaded:
            self.load_meshes()  # if objects have not been loaded yet, load them
        self.setup_objects_()
        self.objects_loaded = True
        return

    def setup_objects_(self):
        """
        Scenario-specific logic
        """
        raise NotImplementedError

    def setup_cameras(self):
        if self.cameras_loaded:
            return
        print("camera setup...")
        self.cameras = []
        self.camera_objs = []
        cam_config = self.config["camera"]
        base_lookat = cam_config["base_lookat"]

        # pick default ori. angle and (n_cameras-1) other angles from a linspace of angles that are 5 degrees apart
        default_ori_angle = cam_config["orientation_angle_default"]
        cam_ori_angles = [0] + random.sample(np.linspace(0, 360, 72+1).tolist()[1:-1], k=self.n_cameras-1)
        cam_ori_angles = [(angle + default_ori_angle) % 360 for angle in cam_ori_angles]
        # TODO parameters 'orientation_angle_min/max' are not yet used!

        for i, cam_ori_angle in enumerate(cam_ori_angles):
            cam_elev_angle = random.uniform(cam_config["elevation_angle_min"], cam_config["elevation_angle_max"])
            cam_dist = random.uniform(cam_config["distance_min"], cam_config["distance_max"])
            cam_lookat = deepcopy(base_lookat)
            cam_name = f"cam_{str(i).zfill(2)}"
            cam_stereo_positions = ["left", "right"] if self.coplanar_stereo else ["mono"]
            self.cameras.append(Camera(cam_name, self.cam_dt, cam_elev_angle, cam_ori_angle, cam_dist, cam_lookat,
                                       self.coplanar_stereo_dist, cam_stereo_positions, self.cam_movement_complexity))
        self.setup_cameras_()  # e.g. scenario-specific height adjustment
        self.setup_camera_objs()
        self.cameras_loaded = True

    def setup_camera_objs(self):
        """
        Setting an object for each of the cameras.
         - Viewer mode: A full mesh is displayed at the position and with the pose of the camera
         - Normal mode: A tiny-dummy obj is place on the location of the camera to fill the occ-matrix cell
        """
        camera_mesh = CONSTANTS.CAMERA_OBJ if self.viewer_mode else CONSTANTS.DUMMY_CAMERA_OBJ
        for camera_id, camera in enumerate(self.cameras):
            self.mesh_loader.load_meshes(camera_mesh)
            camera_pos = camera.get_pos()
            camera_info_mesh = self.mesh_loader.get_meshes()[-1]
            self.camera_objs.append(self.add_object_to_scene(camera_info_mesh, is_static=True))
            pose = torch.eye(4)
            pose[:2, -1] = camera_pos[:2]
            pose[2, -1] = camera_pos[-1] + self.camera_objs[-1].mesh.bbox.min[-1]
            pose[:3, :3] = utils.get_rot_matrix(
                    yaw=torch.tensor(camera.ori_angle * np.pi / 180),
                    pitch=torch.tensor(-1 * camera.elev_angle * np.pi / 180),
                    roll=torch.tensor(0.)
                )
            self.camera_objs[-1].set_pose(pose)
            self.scene.add_object(self.camera_objs[-1])
        return

    def setup_cameras_(self):
        """
        Scenario-specific logic, e.g. height adjustment
        """
        raise NotImplementedError

    def simulate(self):
        '''
        Can be overwritten by scenario-specific logic
        '''
        self.sim_t += self.sim_dt
        self.sim_step_()

    def sim_step_(self):
        '''
        Just calls the appropriate simulator; assumes that all other things have been taken care of.
        '''
        if self.physics_engine == "physx":
            self.scene.simulate(self.sim_dt)
        elif self.physics_engine == "physx_manipulation_sim":
            raise NotImplementedError  # TODO implement for gripper sim
        elif self.physics_engine == "nimble":
            if not self.nimble_loaded:
                self.setup_nimble_()
            self.simulate_nimble_()
        else:
            raise ValueError(f"invalid physics_engine parameter: {self.physics_engine}")

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

    def add_object_to_scene(self, obj_info_mesh: Tuple[OBJECT_INFO.ObjectInfo, sl.Mesh], is_static: bool, **obj_mod):
        obj_info, obj_mesh = obj_info_mesh
        obj = self.object_loader.create_object(obj_info, obj_mesh, is_static, **obj_mod)
        self.scene.add_object(obj)
        return obj

    def remove_obj_from_scene(self, obj: sl.Object, decrement_ins_idx: bool=True):
        self.scene.remove_object(obj)
        self.object_loader.remove_object(obj.instance_index, decrement_ins_idx=decrement_ins_idx)

    def update_object_height(self, cur_obj, objs=[], scales=None):
        """ Updating an object z-position given a list of supporting objects"""
        scales = [1.0] * len(objs) if scales is None else scales
        assert len(objs) == len(scales), "provided non-matching scales for update_camera_height"
        cur_obj_pose = cur_obj.pose()
        z_pose = self.get_obj_z_offset(cur_obj)
        for obj, scale in zip(objs, scales):
            z_pose += self.get_obj_z_offset(obj) * scale
        cur_obj_pose[2, -1] = z_pose
        cur_obj.set_pose(cur_obj_pose)
        return cur_obj

    def update_camera_height(self, camera, objs=[], scales=None):
        """ Updating the camera position, camera-object position and the look-at parameter"""
        scales = [1.0] * len(objs) if scales is None else scales
        assert len(objs) == len(scales), "provided non-matching scales for update_camera_height"
        z_lookat = deepcopy(camera.start_base_lookat[-1])
        for obj, scale in zip(objs, scales):
            z_lookat += self.get_obj_z_offset(obj) * scale
        camera.start_base_lookat[-1] = z_lookat
        return camera

    def get_obj_z_offset(self, obj):
        """ Obtaining the z_offset (z-pos + height) for a given object"""
        obj_pose = obj.pose()
        z_offset = obj_pose[2, -1] + (obj.mesh.bbox.max[-1] - obj.mesh.bbox.min[-1]) / 2
        return z_offset

    def get_obj_offset(self, obj):
        """ Obtaining the bbox boundaries (pos + size for x,y,z) for a given object"""
        obj_pose = obj.pose()
        offset_x, offset_y, offset_z = obj_pose[:3, -1] + obj.mesh.bbox.max
        offset = torch.Tensor([-offset_x, -offset_y, offset_z])
        return offset
