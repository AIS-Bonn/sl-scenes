import stillleben as sl
import random
import torch

import ycb_dynamic.CONSTANTS as CONSTANTS
from ycb_dynamic.lighting import get_default_light_map
from ycb_dynamic.object_models import load_table_and_ycbv
from ycb_dynamic.camera import Camera
from ycb_dynamic.scenarios.scenario import Scenario, add_obj_to_scene


class StackScenario(Scenario):
    def __init__(self, cfg, scene):
        self.name = "Stack"
        self.prep_time = 0.002  # during this time (in s), the scene will not be rendered
        super(StackScenario, self).__init__(cfg, scene)   # also calls reset_sim()

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


    def setup_objects(self):
        print("object setup...")
        self.static_objects, self.dynamic_objects = [], []
        loaded_meshes, loaded_mesh_weights = load_table_and_ycbv()
        table_mesh, obj_meshes = loaded_meshes
        table_weight, obj_weights = loaded_mesh_weights

        # place the static objects (table) into the scene
        table = sl.Object(table_mesh)
        table.set_pose(CONSTANTS.TABLE_POSE)
        table.mass = table_weight
        table.static = True
        add_obj_to_scene(self.scene, table)
        self.static_objects.append(table)

        # drop 10 random YCB-Video objects onto the table
        for (mesh, weight) in random.choices(list(zip(obj_meshes, obj_weights)), k=10):
            obj = sl.Object(mesh)
            p = obj.pose()
            x = random.uniform(CONSTANTS.DROP_LIMITS["x_min"], CONSTANTS.DROP_LIMITS["x_max"])
            y = random.uniform(CONSTANTS.DROP_LIMITS["y_min"], CONSTANTS.DROP_LIMITS["y_max"])
            z = random.uniform(CONSTANTS.DROP_LIMITS["z_min"], CONSTANTS.DROP_LIMITS["z_max"])
            p[:3, 3] = torch.tensor([x, y, z])
            obj.set_pose(p)
            obj.mass = weight
            add_obj_to_scene(self.scene, obj)
            self.dynamic_objects.append(obj)

    def setup_cameras(self):
        print("camera setup...")
        self.cameras = []
        self.cameras.append(Camera("main", CONSTANTS.CAM_POS, CONSTANTS.CAM_LOOKAT, moving=False))

    def simulate(self, dt):
        self.scene.simulate(dt)
        self.sim_t += dt