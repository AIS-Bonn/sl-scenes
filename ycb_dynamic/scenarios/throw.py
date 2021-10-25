import stillleben as sl
import random
import torch
import numpy as np

import ycb_dynamic.CONSTANTS as CONSTANTS
from ycb_dynamic.CONFIG import CONFIG
from ycb_dynamic.object_models import load_table_and_ycbv
from ycb_dynamic.camera import Camera
from ycb_dynamic.scenarios.scenario import Scenario, add_obj_to_scene, remove_obj_from_scene


class ThrowScenario(Scenario):
    def __init__(self, cfg, scene):
        self.name = "Throw"
        self.config = CONFIG["scenes"]["throw"]
        self.prep_time = 0  # during this time (in s), the scene will not be rendered
        self.meshes_loaded = False
        self.bowling_ball_loaded = False
        super(ThrowScenario, self).__init__(cfg, scene)   # also calls reset_sim()

    def can_render(self):
        """
        :return: True if scene has been prepared and can be rendered, False otherwise.
        """
        return self.sim_t > self.prep_time

    def load_meshes(self):
        loaded_meshes, loaded_mesh_weights = load_table_and_ycbv()
        self.table_mesh, self.obj_meshes = loaded_meshes
        self.table_weight, self.obj_weights = loaded_mesh_weights
        self.meshes_loaded = True

    def setup_objects(self):
        print("object setup...")
        self.static_objects, self.dynamic_objects = [], []
        if not self.meshes_loaded:
            self.load_meshes()  # if objects have not been loaded yet, load them

        # place the static objects (table) into the scene
        table = sl.Object(self.table_mesh)
        table.set_pose(CONSTANTS.TABLE_POSE)
        table.mass = self.table_weight
        table.static = True
        add_obj_to_scene(self.scene, table)
        self.static_objects.append(table)

        # throw 10 random YCB-Video objects onto the table, from the side
        for (mesh, weight) in random.choices(list(zip(self.obj_meshes, self.obj_weights)), k=10):
            obj = sl.Object(mesh)
            p = obj.pose()
            x = random.uniform(CONSTANTS.DROP_LIMITS["x_min"], CONSTANTS.DROP_LIMITS["x_max"])
            y = -1.0
            z = random.uniform(CONSTANTS.DROP_LIMITS["z_min"], CONSTANTS.DROP_LIMITS["z_max"])
            p[:3, 3] = torch.tensor([x, y, z])
            obj.set_pose(p)
            obj.mass = weight
            linear_noise = self.config["linear_noise_std"] * torch.randn(3,) + self.config["linear_noise_mean"]
            angular_noise = self.config["angular_noise_std"] * torch.randn(3,) + self.config["angular_noise_mean"]
            obj.linear_velocity = self.config["linear_velocity"] + linear_noise
            obj.angular_velocity = self.config["angular_velocity"] + angular_noise
            # print(obj.linear_velocity, obj.angular_velocity)
            add_obj_to_scene(self.scene, obj)
            if(self.is_there_collision()):  # removing last object if colliding with anything else
                remove_obj_from_scene(self.scene, obj)
            else:
                self.dynamic_objects.append(obj)


    def setup_cameras(self):
        print("camera setup...")
        self.cameras = []
        self.cameras.append(Camera("main", CONSTANTS.BOWLING_CAM_POS, CONSTANTS.CAM_LOOKAT, moving=False))

    def simulate(self, dt):
        self.scene.simulate(dt)
        self.sim_t += dt
