"""
Abstract class for defining scenarios
"""
from typing import Tuple
import numpy as np
import torch
import stillleben as sl
from ycb_dynamic.object_models import MeshLoader, ObjectLoader
from ycb_dynamic.lighting import get_lightmap
import ycb_dynamic.OBJECT_INFO as OBJECT_INFO


class Scenario(object):
    """ Abstract class for defining scenarios """

    def __init__(self, cfg, scene: sl.Scene):
        self.scene = scene
        self.mesh_loader = MeshLoader()
        self.object_loader = ObjectLoader()
        self.meshes_loaded, self.objects_loaded = False, False
        self.z_offset = 0.
        self.lightmap = cfg.lightmap
        self.reset_sim()

    def reset_sim(self):
        self.sim_t = 0
        self.setup_scene()
        self.setup_objects()
        self.setup_cameras()

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

    def setup_scene(self):
        """ Default setup_scene lighting and camera. Can be overriden from specific scenes """
        self.scene.ambient_light = torch.tensor([0.7, 0.7, 0.7])
        self.scene.light_map = get_lightmap(self.lightmap)
        self.scene.choose_random_light_position()

    def get_separations(self):
        assert len(self.dynamic_objects) > 0, "Objects must be added to dynamic_objects before computing collisions"
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

    def update_object_height(self, cur_obj, objs):
        """ Updating an object z-position given a list of supporting objects"""
        cur_obj_pose = cur_obj.pose()
        z_pose = self.get_obj_z_offset(cur_obj)
        for obj in objs:
            z_pose = z_pose + self.get_obj_z_offset(obj)

        cur_obj_pose[2, -1] = z_pose
        cur_obj.set_pose(cur_obj_pose)
        return cur_obj

    def update_camera_height(self, camera, objs):
        """ Updating the camera z-position """

        z_pos = camera.start_pos[-1]
        z_lookat = camera.start_lookat[-1]
        for obj in objs:
            z_pos = z_pos + self.get_obj_z_offset(obj)
            z_lookat = z_lookat + self.get_obj_z_offset(obj)
        camera.start_pos[-1] = z_pos
        camera.start_lookat[-1] = z_lookat
        print(camera.start_pos)
        print(camera.start_lookat)
        camera.reset_cam()
        return camera

    def get_obj_z_offset(self, obj):
        """ Obtaining the z_offset (z-pos + height) for a given object"""
        obj_pose = obj.pose()
        z_offset = obj_pose[2, -1] + obj.mesh.bbox.max[-1]
        return z_offset
