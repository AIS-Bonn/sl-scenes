"""
Throw Scenario: Objects are thrown from the side of the table.
Same as tabletop, but with intial linear velocity
"""

import stillleben as sl
import random
import torch

import ycb_dynamic.utils.utils as utils
import ycb_dynamic.CONSTANTS as CONSTANTS
from ycb_dynamic.CONFIG import CONFIG
from ycb_dynamic.object_models import MeshLoader
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
        """ """
        meshLoader = MeshLoader()
        meshLoader.load_meshes(CONSTANTS.TABLE),
        meshLoader.load_meshes(CONSTANTS.YCBV_OBJECTS),
        loaded_meshes, loaded_mesh_weights = meshLoader.get_meshes(), meshLoader.get_mesh_weights()

        self.table_mesh, self.obj_meshes = loaded_meshes
        self.table_weight, self.obj_weights = loaded_mesh_weights
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

        # throw some random YCB-Video objects onto the table, from the side
        N_objs = random.randint(self.config["other"]["min_objs"], self.config["other"]["max_objs"] + 1)
        for (mesh, weight) in random.choices(list(zip(self.obj_meshes, self.obj_weights)), k=N_objs):
            obj = sl.Object(mesh)
            p = obj.pose()
            x = random.uniform(self.config["pos"]["x_min"], self.config["pos"]["x_max"])
            y = random.uniform(self.config["pos"]["y_min"], self.config["pos"]["y_max"])
            z = self.z_offset + random.uniform(self.config["pos"]["z_min"], self.config["pos"]["z_max"])
            p[:3, 3] = torch.tensor([x, y, z])
            obj.set_pose(p)
            obj.mass = weight
            obj.linear_velocity = utils.get_noisy_vect(
                    v=self.config["velocity"]["lin_velocity"],
                    mean=self.config["velocity"]["lin_noise_mean"],
                    std=self.config["velocity"]["lin_noise_std"]
            )
            obj.angular_velocity = utils.get_noisy_vect(
                    v=self.config["velocity"]["ang_velocity"],
                    mean=self.config["velocity"]["ang_noise_mean"],
                    std=self.config["velocity"]["ang_noise_std"]
            )
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
