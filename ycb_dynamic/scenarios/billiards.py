import stillleben as sl
import torch
import random

import ycb_dynamic.CONSTANTS as CONSTANTS
from ycb_dynamic.lighting import get_default_light_map
from ycb_dynamic.object_models import load_billiards
from ycb_dynamic.camera import Camera
from ycb_dynamic.scenarios.scenario import Scenario, add_obj_to_scene


class BillardsScenario(Scenario):
    def __init__(self, cfg, scene):
        self.name = "Billards"
        self.prep_time = 0.002  # during this time (in s), the scene will not be rendered
        super(BillardsScenario, self).__init__(cfg, scene)   # also calls reset_sim()

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
        loaded_meshes, loaded_weights = load_billiards()
        self.table_mesh, self.bowling_mesh, self.objects_triangle_mesh = loaded_meshes
        self.table_weight, self.bowling_weight, self.objects_triangle_weights = loaded_weights

    def setup_objects(self):
        print("object setup...")
        self.static_objects, self.dynamic_objects = [], []
        if not self.meshes_loaded:
            self.load_meshes() # if objects have not been loaded yet, load them

        # place the static objects (table) into the scene
        table = sl.Object(self.table_mesh)
        table.set_pose(CONSTANTS.TABLE_POSE)
        table.mass = self.table_weight
        table.static = True
        add_obj_to_scene(self.scene, table)
        self.static_objects.append(table)

        # assemble several objects in a triangle-like shape
        N = len(CONSTANTS.BILLIARDS_TRIANLGE_POSES)
        for i, (mesh, weight) in enumerate(
                random.choices(list(zip(self.objects_triangle_mesh, self.objects_triangle_weights)), k=N)):
            object = sl.Object(mesh)
            object.set_pose(CONSTANTS.BILLIARDS_TRIANLGE_POSES[i])
            object.mass = weight
            add_obj_to_scene(self.scene, object)
            self.dynamic_objects.append(object)

        bowling_ball = sl.Object(self.bowling_mesh)
        bp = bowling_ball.pose()
        bp[:3, 3] = torch.tensor([-0.9, 0, 1.25])
        bowling_ball.set_pose(bp)
        bowling_ball.mass = self.bowling_weight
        bowling_ball.linear_velocity = torch.tensor([2.0, 0, 0])
        add_obj_to_scene(self.scene, bowling_ball)
        self.dynamic_objects.append(bowling_ball)

    def setup_cameras(self):
        print("camera setup...")
        self.cameras = []
        self.cameras.append(Camera("main", CONSTANTS.CAM_POS, CONSTANTS.CAM_LOOKAT, moving=False))

    def simulate(self, dt):
        self.scene.simulate(dt)
        self.sim_t += dt