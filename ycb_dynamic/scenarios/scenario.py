"""
Abstract class for defining scenarios
"""

import numpy as np
import torch
import stillleben as sl
from ycb_dynamic.lighting import get_lightmap


class Scenario(object):
    """ Abstract class for defining scenarios """

    def __init__(self, cfg, scene):
        self.scene = scene
        self.meshes_loaded = False
        self.lightmap = cfg.lightmap
        self.reset_sim()

    def reset_sim(self):
        self.sim_t = 0
        self.setup_scene()
        self.setup_objects()
        self.setup_cameras()

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
        assert self.dynamic_objects is not None, "Objects must be added to dynamic_objects before computing collisions"
        self.scene.check_collisions()
        separations = [object.separation for object in self.dynamic_objects]
        return separations

    def is_there_collision(self):
        separations = self.get_separations()
        collision = True if np.sum(separations) < 0 else False
        return collision

    def load_meshes(self):
        raise NotImplementedError

    def setup_objects(self):
        raise NotImplementedError

    def setup_cameras(self):
        raise NotImplementedError

    def simulate(self, dt):
        raise NotImplementedError


def add_obj_to_scene(scene: sl.Scene, obj: sl.Object):
    obj.instance_index = len(scene.objects) + 1
    scene.add_object(obj)

def remove_obj_from_scene(scene: sl.Scene, obj: sl.Object):
    obj.instance_index = len(scene.objects) - 1
    scene.remove_object(obj)
