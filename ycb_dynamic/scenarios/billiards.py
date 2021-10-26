"""
 Billiards Scenario: A ball smashes through a bunch of objects placed in a billiards-triangle manner
"""
import stillleben as sl
import torch
import random
from copy import deepcopy

import ycb_dynamic.utils.utils as utils
from ycb_dynamic.CONFIG import CONFIG
import ycb_dynamic.CONSTANTS as CONSTANTS
from ycb_dynamic.object_models import MeshLoader
from ycb_dynamic.camera import Camera
from ycb_dynamic.scenarios.scenario import Scenario, add_obj_to_scene, remove_obj_from_scene


class BillardsScenario(Scenario):
    def __init__(self, cfg, scene):
        self.name = "Billards"
        self.config = CONFIG["scenes"]["billards"]
        self.prep_time = 0.002  # during this time (in s), the scene will not be rendered
        super(BillardsScenario, self).__init__(cfg, scene)   # also calls reset_sim()

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
        meshLoader.load_meshes(CONSTANTS.BILLIARDS_OBJECTS),
        loaded_meshes, loaded_weights = meshLoader.get_meshes(), meshLoader.get_mesh_weights()

        self.table_mesh, self.bowling_mesh, self.objects_triangle_mesh = loaded_meshes
        self.table_weight, self.bowling_weight, self.objects_triangle_weights = loaded_weights
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
        self.z_offset = table.pose()[2, -1]
        add_obj_to_scene(self.scene, table)
        self.static_objects.append(table)

        # assemble several objects in a triangle-like shape
        N = len(CONSTANTS.BILLIARDS_TRIANLGE_POSES)
        obj_poses = deepcopy(CONSTANTS.BILLIARDS_TRIANLGE_POSES)
        for i, (mesh, weight) in enumerate(
                random.choices(list(zip(self.objects_triangle_mesh, self.objects_triangle_weights)), k=N)):
            object = sl.Object(mesh)
            pose = obj_poses[i]
            pose[2, -1] += self.z_offset
            object.set_pose(pose)
            object.mass = weight
            add_obj_to_scene(self.scene, object)
            if(self.is_there_collision()):  # removing last object if colliding with anything else
                remove_obj_from_scene(self.scene, object)
            else:
                self.dynamic_objects.append(object)

        # adding the bowling_ball with custom position and velocity
        bowling_ball = sl.Object(self.bowling_mesh)
        bp = bowling_ball.pose()
        x = random.uniform(self.config["pos"]["x_min"], self.config["pos"]["x_max"])
        y = random.uniform(self.config["pos"]["y_min"], self.config["pos"]["y_max"])
        z = self.z_offset + random.uniform(self.config["pos"]["z_min"], self.config["pos"]["z_max"])
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
        return

    def setup_cameras(self):
        print("camera setup...")
        self.cameras = []
        self.cameras.append(Camera("main", CONSTANTS.CAM_POS, CONSTANTS.CAM_LOOKAT, moving=False))

    def simulate(self, dt):
        self.scene.simulate(dt)
        self.sim_t += dt
