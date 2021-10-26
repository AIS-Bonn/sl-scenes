"""
Bowling Scenario: A ball smashes through a tower of wooden blocks
"""
import stillleben as sl
import torch
import random

import ycb_dynamic.utils.utils as utils
from ycb_dynamic.CONFIG import CONFIG
import ycb_dynamic.CONSTANTS as CONSTANTS
from ycb_dynamic.object_models import MeshLoader
from ycb_dynamic.camera import Camera
from ycb_dynamic.scenarios.scenario import Scenario, add_obj_to_scene, remove_obj_from_scene


class BowlingScenario(Scenario):
    def __init__(self, cfg, scene):
        self.name = "Bowling"
        self.config = CONFIG["scenes"]["bowling"]
        self.prep_time = 1.000  # during this time (in s), the scene will not be rendered
        self.bowling_ball_loaded = False
        super(BowlingScenario, self).__init__(cfg, scene)   # also calls reset_sim()

    def can_render(self):
        """
        :return: True if scene has been prepared and can be rendered, False otherwise.
        """
        return self.sim_t > self.prep_time

    def load_meshes(self):
        """ """
        meshLoader = MeshLoader()
        meshLoader.load_meshes(CONSTANTS.TABLE),
        meshLoader.load_meshes(CONSTANTS.BOWLING_BALL),
        meshLoader.load_meshes(CONSTANTS.WOOD_BLOCK),
        loaded_meshes, loaded_weights = meshLoader.get_meshes(), meshLoader.get_mesh_weights()

        self.table_mesh, self.bowling_mesh, self.wood_block_mesh = loaded_meshes
        self.table_weight, self.bowling_weight, self.wood_block_weight = loaded_weights
        self.meshes_loaded = True
        return

    def setup_objects(self):
        """ """
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

        # assemble a pyramid of wooden blocks
        for wb_pose in CONSTANTS.WOOD_BLOCK_POSES:
            wood_block = sl.Object(self.wood_block_mesh)
            wood_block.set_pose(wb_pose)
            wood_block.mass = self.wood_block_weight
            add_obj_to_scene(self.scene, wood_block)

            if(self.is_there_collision()):  # removing last object if colliding with anything else
                remove_obj_from_scene(self.scene, wood_block)
            else:
                self.dynamic_objects.append(wood_block)
        return

    def add_bowling_ball(self):
        """ """
        if not self.meshes_loaded:
            self.load_meshes()
        bowling_ball = sl.Object(self.bowling_mesh)
        bp = bowling_ball.pose()
        x = random.uniform(self.config["pos"]["x_min"], self.config["pos"]["x_max"])
        y = random.uniform(self.config["pos"]["y_min"], self.config["pos"]["y_max"])
        z = random.uniform(self.config["pos"]["z_min"], self.config["pos"]["z_max"])
        bp[:3, 3] = torch.tensor([x, y, z])
        bowling_ball.set_pose(bp)
        bowling_ball.mass = self.bowling_weight
        bowling_ball.linear_velocity = utils.get_noisy_vect(
                v=self.config["velocity"]["lin_velocity"],
                mean=self.config["velocity"]["lin_noise_mean"],
                std=self.config["velocity"]["lin_noise_std"]
            )
        add_obj_to_scene(self.scene, bowling_ball)
        self.dynamic_objects.append(bowling_ball)
        self.bowling_ball_loaded = True
        return

    def setup_cameras(self):
        print("camera setup...")
        self.cameras = []
        self.cameras.append(Camera("main", CONSTANTS.BOWLING_CAM_POS, CONSTANTS.CAM_LOOKAT, moving=False))

    def simulate(self, dt):
        # add bowling ball after preparation time to ensure that the object tower stands still
        if self.sim_t > self.prep_time and not self.bowling_ball_loaded:
            self.add_bowling_ball()
        self.scene.simulate(dt)
        self.sim_t += dt
