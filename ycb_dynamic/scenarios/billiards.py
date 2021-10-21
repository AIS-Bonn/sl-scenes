import stillleben as sl
import torch
import random

import ycb_dynamic.CONSTANTS as CONSTANTS
from ycb_dynamic.lighting import get_default_light_map
from ycb_dynamic.object_models import load_billiards
from ycb_dynamic.camera import Camera
from ycb_dynamic.scenarios.scenario import Scenario, add_obj_to_scene


def setup_billiards_scene(cfg, scene):
    """
    """

    print("scene setup...")
    scene.ambient_light = torch.tensor([0.7, 0.7, 0.7])
    scene.light_map = get_default_light_map()
    scene.choose_random_light_position()
    # scene.background_plane_size = torch.tensor([10.0, 10.0])
    # scene.background_color = torch.tensor([0.1, 0.1, 0.1, 1.0])

    print("loading objects...")
    table_mesh, bowling_mesh, objects_triangle_mesh = load_billiards()

    # place the static objects (table) into the scene
    table = sl.Object(table_mesh)
    table.set_pose(CONSTANTS.TABLE_POSE)
    table.static = True
    add_obj_to_scene(scene, table)

    dynamic_objects = []

    # assemble several objects in a triangle-like shape
    N = len(CONSTANTS.BILLIARDS_TRIANLGE_POSES)
    for i, mesh in enumerate(random.choices(objects_triangle_mesh, k=N)):
        object = sl.Object(mesh)
        object.set_pose(CONSTANTS.BILLIARDS_TRIANLGE_POSES[i])
        dynamic_objects.append(object)
        add_obj_to_scene(scene, object)

    bowling_ball = sl.Object(bowling_mesh)
    bp = bowling_ball.pose()
    bp[:3, 3] = torch.tensor([-0.9, 0, 1.25])
    bowling_ball.set_pose(bp)
    bowling_ball.linear_velocity = torch.tensor([2.0, 0, 0])
    dynamic_objects.append(bowling_ball)
    add_obj_to_scene(scene, bowling_ball)

    main_cam = Camera("main", CONSTANTS.CAM_POS, CONSTANTS.CAM_LOOKAT, moving=False)
    billiards_scenario = Scenario(
        name="Billiards",
        scene=scene,
        cameras=[main_cam],
        static_objects=[table],
        dynamic_objects=dynamic_objects,
    )

    return billiards_scenario
