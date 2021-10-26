"""
Stacked Scenario: Objects with flat surfaces assemble pyramids, which max fall due to lack of support
"""
import stillleben as sl
import random
import numpy as np
import torch

import ycb_dynamic.CONSTANTS as CONSTANTS
from ycb_dynamic.CONFIG import CONFIG
from ycb_dynamic.object_models import MeshLoader
from ycb_dynamic.camera import Camera
from ycb_dynamic.scenarios.scenario import Scenario, add_obj_to_scene, remove_obj_from_scene


class StackScenario(Scenario):
    def __init__(self, cfg, scene):
        self.name = "Stack"
        self.prep_time = 0.002  # during this time (in s), the scene will not be rendered
        self.config = CONFIG["scenes"]["stack"]
        super(StackScenario, self).__init__(cfg, scene)   # also calls reset_sim()

    def can_render(self):
        """
        :return: True if scene has been prepared and can be rendered, False otherwise.
        """
        return self.sim_t > self.prep_time

    def load_meshes(self):
        """ """
        meshLoader = MeshLoader()
        meshLoader.load_meshes(CONSTANTS.TABLE),
        meshLoader.load_meshes(CONSTANTS.STACK_OBJECTS),
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

        # place the static objects (table) into the scene
        table = sl.Object(self.table_mesh)
        table.set_pose(CONSTANTS.TABLE_POSE)
        table.mass = self.table_weight
        table.static = True
        add_obj_to_scene(self.scene, table)
        self.static_objects.append(table)

        # randomly sampling the number of stacks, and selecting mean (x,y) center positions
        N_poses = len(CONSTANTS.STACK_PYRAMID_POSES)
        N_pyramids = np.random.randint(low=self.config["stacks_min"], high=self.config["stacks_max"] + 1)

        x_coords = torch.arange(self.config["x_disp_min"], self.config["x_disp_max"], self.config["x_disp_step"])
        x_coords = x_coords[torch.randperm(len(x_coords))][:N_pyramids]
        y_coords = torch.arange(self.config["y_disp_min"], self.config["y_disp_max"], self.config["y_disp_step"])
        y_coords = y_coords[torch.randperm(len(y_coords))][:N_pyramids]

        pyramid_centers = torch.zeros(N_pyramids, 4, 4)
        pyramid_centers[:, 0,  -1] = x_coords.view(N_pyramids)
        pyramid_centers[:, 1, -1] = y_coords.view(N_pyramids)

        # assemble the pyramids of stacked objects. Initial poses are noisy, which might lead to pyramid falling
        for n in range(N_pyramids):
            for i, (mesh, weight) in enumerate(random.choices(list(zip(self.obj_meshes, self.obj_weights)), k=N_poses)):
                object = sl.Object(mesh)
                base_pose = CONSTANTS.STACK_PYRAMID_POSES[i]
                pose = base_pose + pyramid_centers[n]
                pose = self._add_pose_noise(pose)
                object.set_pose(pose)
                object.mass = weight
                add_obj_to_scene(self.scene, object)
                # removing last object if colliding with anything else
                if(self.is_there_collision()):
                    remove_obj_from_scene(self.scene, object)
                else:
                    self.dynamic_objects.append(object)

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
