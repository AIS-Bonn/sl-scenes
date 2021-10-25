import math

import numpy as np
import stillleben as sl
import torch
import random

import ycb_dynamic.CONSTANTS as CONSTANTS
from ycb_dynamic.lighting import get_default_light_map
from ycb_dynamic.object_models import load_bowl
from ycb_dynamic.camera import Camera
from ycb_dynamic.scenarios.scenario import Scenario, add_obj_to_scene


class BowlScenario(Scenario):
    def __init__(self, cfg, scene):
        self.name = "Bowl"
        self.prep_time = 0.0  # during this time (in s), the scene will not be rendered
        super(BowlScenario, self).__init__(cfg, scene)   # also calls reset_sim()

    def can_render(self):
        """
        :return: True if scene has been prepared and can be rendered, False otherwise.
        """
        return self.sim_t > self.prep_time

    def setup_scene(self):
        print("scene setup...")
        self.scene.ambient_light = torch.tensor([0.7, 0.7, 0.7])
        self.scene.light_map = get_default_light_map()
        self.scene.choose_random_light_position()

    def load_meshes(self):
        loaded_meshes, loaded_weights = load_bowl()
        self.table_mesh, self.wooden_bowl_mesh, self.bowling_mesh = loaded_meshes
        self.table_weight, self.wooden_bowl_weight, self.bowling_weight = loaded_weights

    def setup_objects(self):
        print("object setup...")
        self.static_objects, self.dynamic_objects = [], []
        if not self.meshes_loaded:
            self.load_meshes() # if objects have not been loaded yet, load them

        # place the static objects (table, bowl) into the scene
        table = sl.Object(self.table_mesh)
        table.set_pose(CONSTANTS.TABLE_POSE)
        table.mass = self.table_weight
        table.static = True
        add_obj_to_scene(self.scene, table)
        self.static_objects.append(table)

        wooden_bowl = sl.Object(self.wooden_bowl_mesh)
        wooden_bowl.set_pose(CONSTANTS.WOODEN_BOWL_POSE)
        wooden_bowl.mass = self.wooden_bowl_weight
        wooden_bowl.static = True
        add_obj_to_scene(self.scene, wooden_bowl)
        self.static_objects.append(wooden_bowl)

        # spawn several balls at random positions in the bowl
        k = random.randint(1, 5)  # 1 to 5 balls in bowl
        ball_placement_angles = np.linspace(0, 2*np.pi, num=10).tolist()
        ball_placement_angles = random.sample(ball_placement_angles[:-1], k=k)
        for angle in ball_placement_angles:
            ball = sl.Object(self.bowling_mesh)
            p = CONSTANTS.BOWL_BOWLING_BALL_INIT_POS
            p[:2, 3] = 0.5 * torch.tensor([np.sin(angle), np.cos(angle)])  # assign x and y coordiantes
            ball.set_pose(p)
            ball.mass = self.bowling_weight
            add_obj_to_scene(self.scene, ball)
            self.dynamic_objects.append(ball)

    def setup_cameras(self):
        print("camera setup...")
        self.cameras = []
        self.cameras.append(Camera("main", CONSTANTS.BOWL_CAM_POS,
                                   CONSTANTS.BOWL_CAM_LOOKAT, moving=False))

    def simulate(self, dt):
        self.scene.simulate(dt)
        self.sim_t += dt