"""
Bowl Scenario: Several balls/fruits are placed inside a bowl, approaching and colliding with each other.
"""
import numpy as np
import stillleben as sl
import torch
import random

import ycb_dynamic.CONSTANTS as CONSTANTS
from ycb_dynamic.object_models import MeshLoader
from ycb_dynamic.camera import Camera
from ycb_dynamic.scenarios.scenario import Scenario, add_obj_to_scene, remove_obj_from_scene


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

    def load_meshes(self):
        """ """
        meshLoader = MeshLoader()
        meshLoader.load_meshes(CONSTANTS.TABLE),
        meshLoader.load_meshes(CONSTANTS.WOODEN_BOWL)
        meshLoader.load_meshes(CONSTANTS.FRUIT_OBJECTS)
        loaded_meshes, loaded_weights = meshLoader.get_meshes(), meshLoader.get_mesh_weights()

        self.table_mesh, self.wooden_bowl_mesh, self.fruit_meshes = loaded_meshes
        self.table_weight, self.wooden_bowl_weight, self.fruit_weights = loaded_weights
        self.meshes_loaded = True
        return

    def setup_objects(self):
        print("object setup...")
        self.static_objects, self.dynamic_objects = [], []
        if not self.meshes_loaded:
            self.load_meshes()  # if objects have not been loaded yet, load them

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
        k = random.randint(1, 7)  # 1 to 5 balls in bowl
        obj_placement_angles = np.linspace(0, 2*np.pi, num=10).tolist()
        obj_placement_angles = random.sample(obj_placement_angles[:-1], k=k)
        meshes_and_weights = random.choices(list(zip(self.fruit_meshes, self.fruit_weights)), k=k)
        for angle, (mesh, weight) in zip(obj_placement_angles, meshes_and_weights):
            obj = sl.Object(mesh)
            p = CONSTANTS.BOWL_FRUIT_INIT_POS
            p[:2, 3] = 0.33 * torch.tensor([np.sin(angle), np.cos(angle)])  # assign x and y coordiantes
            obj.set_pose(p)
            obj.mass = weight
            add_obj_to_scene(self.scene, obj)
            if(self.is_there_collision()):  # removing last object if colliding with anything else
                remove_obj_from_scene(self.scene, obj)
            else:
                self.dynamic_objects.append(obj)

    def setup_cameras(self):
        print("camera setup...")
        self.cameras = []
        self.cameras.append(Camera("main", CONSTANTS.BOWL_CAM_POS,
                                   CONSTANTS.BOWL_CAM_LOOKAT, moving=False))

    def simulate(self, dt):
        self.scene.simulate(dt)
        self.sim_t += dt
