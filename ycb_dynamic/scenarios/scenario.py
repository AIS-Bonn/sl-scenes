"""
Abstract class for defining scenarios
"""
import random
from typing import Tuple
import numpy as np
from copy import deepcopy
import torch
import stillleben as sl

from ycb_dynamic.room_models import RoomAssembler
from ycb_dynamic.object_models import MeshLoader, ObjectLoader, DecoratorLoader
from ycb_dynamic.lighting import get_lightmap
from ycb_dynamic.camera import Camera
import ycb_dynamic.OBJECT_INFO as OBJECT_INFO


class Scenario(object):
    """ Abstract class for defining scenarios """

    config = dict()

    def __init__(self, cfg, scene: sl.Scene):
        self.scene = scene
        self.mesh_loader = MeshLoader()
        self.room_assembler = RoomAssembler(scene=self.scene)
        self.object_loader = ObjectLoader()
        self.decorator_loader = DecoratorLoader(scene=self.scene)

        self.meshes_loaded, self.objects_loaded = False, False
        self.z_offset = 0.
        self.lightmap = cfg.lightmap
        self.n_cameras = cfg.cameras
        self.coplanar_stereo = cfg.coplanar_stereo
        self.coplanar_stereo_dist = cfg.coplanar_stereo_dist
        self.cam_movement_complexity = cfg.cam_movement_complexity
        self.reset_sim()
        return

    def reset_sim(self):
        self.meshes_loaded, self.objects_loaded, self.cameras_loaded = False, False, False
        self.sim_t = 0
        self.setup_scene()
        self.setup_objects()
        self.setup_cameras()
        self.decorate_scene()


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
        self.decorator_loader.decorate_scene(object_loader=self.object_loader)
        return

    def setup_scene(self):
        """ Default setup_scene lighting and camera. Can be overriden from specific scenes """
        self.scene.ambient_light = torch.tensor([0.7, 0.7, 0.7])
        _ = self.room_assembler.make_room()
        # self.scene.light_map = get_lightmap(self.lightmap)
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
        cam_config = self.config["camera"]
        conf_cam_lookat = cam_config["lookat"]

        # pick default ori. angle and (n_cameras-1) other angles from a linspace of angles that are 5 degrees apart
        default_ori_angle = cam_config["orientation_angle_default"]
        cam_ori_angles = [0] + random.sample(np.linspace(0, 360, 72+1).tolist()[1:-1], k=self.n_cameras-1)
        cam_ori_angles = [(angle + default_ori_angle) % 360 for angle in cam_ori_angles]
        # TODO parameters 'orientation_angle_min/max' are not yet used!

        for i, cam_ori_angle in enumerate(cam_ori_angles):
            cam_elev_angle = random.uniform(cam_config["elevation_angle_min"], cam_config["elevation_angle_max"])
            cam_dist = random.uniform(cam_config["distance_min"], cam_config["distance_max"])
            cam_lookat = deepcopy(conf_cam_lookat)
            cam_name = f"cam_{str(i).zfill(2)}"
            cam_stereo_positions = ["left", "right"] if self.coplanar_stereo else ["mono"]
            self.cameras.append(Camera(cam_name, cam_elev_angle, cam_ori_angle, cam_dist, cam_lookat,
                                       self.coplanar_stereo_dist, cam_stereo_positions, self.cam_movement_complexity))
        self.setup_cameras_()  # e.g. scenario-specific height adjustment
        self.cameras_loaded = True

    def setup_cameras_(self):
        """
        Scenario-specific logic, e.g. height adjustment
        """
        raise NotImplementedError

    def simulate(self, dt):
        raise NotImplementedError

    def add_object_to_scene(self, obj_info_mesh: Tuple[OBJECT_INFO.ObjectInfo, sl.Mesh], is_static: bool, **obj_mod):
        obj_info, obj_mesh = obj_info_mesh
        obj = self.object_loader.create_object(obj_info, obj_mesh, is_static, **obj_mod)
        self.scene.add_object(obj)
        return obj

    def remove_obj_from_scene(self, obj: sl.Object, decrement_ins_idx: bool=True):
        self.scene.remove_object(obj)
        self.object_loader.remove_object(obj.instance_index, decrement_ins_idx=decrement_ins_idx)

    def update_object_height(self, cur_obj, objs=[]):
        """ Updating an object z-position given a list of supporting objects"""
        cur_obj_pose = cur_obj.pose()
        z_pose = self.get_obj_z_offset(cur_obj)
        for obj in objs:
            z_pose = z_pose + self.get_obj_z_offset(obj)
        cur_obj_pose[2, -1] = z_pose
        cur_obj.set_pose(cur_obj_pose)
        return cur_obj

    def update_camera_height(self, camera, objs=[]):
        """ Updating the camera position and the look-at parameter"""
        pos = camera.start_base_pos
        z_lookat = camera.start_base_lookat[-1]
        for obj in objs:
            pos += self.get_obj_offset(obj)
            pos[-1] += 0.2  # NOTE: dirty hack. Should be a param
            z_lookat += self.get_obj_z_offset(obj)
        camera.start_base_pos = pos
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
