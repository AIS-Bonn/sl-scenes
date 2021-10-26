"""
Dice Roll Scenario: A bunch of small and (almost) square objects are rolled on a surface
Same as tabletop, but with some linear velocity and high angular velocity
"""
import stillleben as sl
import random
import torch

import ycb_dynamic.CONSTANTS as CONSTANTS
from ycb_dynamic.CONFIG import CONFIG
from ycb_dynamic.object_models import MeshLoader
from ycb_dynamic.camera import Camera
from ycb_dynamic.scenarios.scenario import Scenario, add_obj_to_scene, remove_obj_from_scene


class DiceRollScenario(Scenario):
    def __init__(self, cfg, scene):
        self.name = "DiceRoll"
        self.config = CONFIG["scenes"].get("dice_roll", {})
        self.prep_time = 0.002  # during this time (in s), the scene will not be rendered
        super(DiceRollScenario, self).__init__(cfg, scene)

    def can_render(self):
        """
        :return: True if scene has been prepared and can be rendered, False otherwise.
        """
        return self.sim_t > self.prep_time

    def load_meshes(self):
        """ """
        meshLoader = MeshLoader()
        meshLoader.load_meshes(CONSTANTS.TABLE),
        meshLoader.load_meshes(CONSTANTS.DICE_OBJECTS),
        loaded_meshes, loaded_mesh_weights = meshLoader.get_meshes(), meshLoader.get_mesh_weights()

        self.table_mesh, self.obj_meshes = loaded_meshes
        self.table_weight, self.obj_weights = loaded_mesh_weights
        self.meshes_loaded = True
        return

    def setup_objects(self):
        print("object setup...")
        self.static_objects, self.dynamic_objects = [], []
        if not self.meshes_loaded:
            self.load_meshes()  # if objects have not been loaded yet, load them

        # place the static objects (bowl) into the scene
        table = sl.Object(self.table_mesh)
        table.set_pose(CONSTANTS.TABLE_POSE)
        table.mass = self.table_weight
        table.static = True
        add_obj_to_scene(self.scene, table)
        self.static_objects.append(table)

        # throw 5 random objects onto the table, from one of the table ends
        for (mesh, weight) in random.choices(list(zip(self.obj_meshes, self.obj_weights)), k=5):
            obj = sl.Object(mesh)
            p = obj.pose()
            x = -1.2  # starting at the beginning of the table
            y = random.uniform(CONSTANTS.DROP_LIMITS["y_min"], CONSTANTS.DROP_LIMITS["y_max"])
            z = random.uniform(CONSTANTS.DROP_LIMITS["z_min"], CONSTANTS.DROP_LIMITS["z_max"])
            p[:3, 3] = torch.tensor([x, y, z])
            obj.set_pose(p)
            obj.mass = weight
            linear_noise = self.config["linear_noise_std"] * torch.randn(3,) + self.config["linear_noise_mean"]
            angular_noise = self.config["angular_noise_std"] * torch.randn(3,) + self.config["angular_noise_mean"]
            obj.linear_velocity = self.config["linear_velocity"] + linear_noise
            obj.angular_velocity = self.config["angular_velocity"] + angular_noise
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
