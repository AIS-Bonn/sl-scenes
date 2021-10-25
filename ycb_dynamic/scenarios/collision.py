import stillleben as sl
import random
import numpy as np
import torch

import ycb_dynamic.CONSTANTS as CONSTANTS
from ycb_dynamic.CONFIG import CONFIG
from ycb_dynamic.object_models import load_collision
from ycb_dynamic.camera import Camera
from ycb_dynamic.scenarios.scenario import Scenario, add_obj_to_scene, remove_obj_from_scene


class CollisionScenario(Scenario):
    def __init__(self, cfg, scene):
        self.name = "Stack"
        self.prep_time = 0.002  # during this time (in s), the scene will not be rendered
        self.config = CONFIG["scenes"]["stack"]
        super(CollisionScenario, self).__init__(cfg, scene)

    def can_render(self):
        """
        :return: True if scene has been prepared and can be rendered, False otherwise.
        """
        return self.sim_t > self.prep_time

    def load_meshes(self):
        loaded_meshes, loaded_mesh_weights = load_collision()
        self.bowl_mesh, self.obj_meshes = loaded_meshes
        self.bowl_weight, self.obj_weights = loaded_mesh_weights
        self.meshes_loaded = True

    def setup_objects(self):
        print("object setup...")
        self.static_objects, self.dynamic_objects = [], []
        if not self.meshes_loaded:
            self.load_meshes()  # if objects have not been loaded yet, load them

        # place the static objects (bowl) into the scene
        bowl = sl.Object(self.bowl_mesh)
        bowl.set_pose(CONSTANTS.BOWL_POSE)
        bowl.mass = self.bowl_weight
        bowl.static = True
        add_obj_to_scene(self.scene, bowl)
        self.static_objects.append(bowl)

        # drop 10 random YCB-Video objects onto the table
        for (mesh, weight) in random.choices(list(zip(self.obj_meshes, self.obj_weights)), k=10):
            obj = sl.Object(mesh)
            p = obj.pose()
            x = random.uniform(CONSTANTS.DROP_LIMITS["x_min"], CONSTANTS.DROP_LIMITS["x_max"])
            y = random.uniform(CONSTANTS.DROP_LIMITS["y_min"], CONSTANTS.DROP_LIMITS["y_max"])
            z = random.uniform(CONSTANTS.DROP_LIMITS["z_min"], CONSTANTS.DROP_LIMITS["z_max"])
            p[:3, 3] = torch.tensor([x, y, z])
            obj.set_pose(p)
            obj.mass = weight
            add_obj_to_scene(self.scene, obj)
            if(self.is_there_collision()):  # removing last object if colliding with anything else
                remove_obj_from_scene(self.scene, obj)
            else:
                self.dynamic_objects.append(obj)

        return

    def _add_pose_noise(self, pose):
        """ Sampling a noise matrix to add to the pose """
        # TODO: Add rotation noise
        pos_noise = torch.rand(size=(3,)) * self.config["pos_noise_mean"] + self.config["pos_noise_std"]
        pose[:3, -1] += pos_noise
        return pose

    def setup_cameras(self):
        print("camera setup...")
        self.cameras = []
        self.cameras.append(Camera("main", CONSTANTS.CAM_POS, CONSTANTS.CAM_LOOKAT, moving=False))

    def simulate(self, dt):
        self.scene.simulate(dt)
        self.sim_t += dt

#
