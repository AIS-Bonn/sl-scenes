import stillleben as sl
import random
import torch
import numpy as np

import ycb_dynamic.CONSTANTS as CONSTANTS
from ycb_dynamic.lighting import get_default_light_map
from ycb_dynamic.object_models import load_bowling
from ycb_dynamic.camera import Camera
from ycb_dynamic.scenarios.scenario import Scenario, add_obj_to_scene


def setup_bowling_scene(cfg, scene):
    """
    TODO Doc
    :param cfg:
    :param scene:
    :return:
    """

    print("scene setup...")
    scene.ambient_light = torch.tensor([0.7, 0.7, 0.7])
    scene.light_map = get_default_light_map()
    scene.choose_random_light_position()
    # scene.background_plane_size = torch.tensor([10.0, 10.0])
    # scene.background_color = torch.tensor([0.1, 0.1, 0.1, 1.0])

    print("loading objects...")
    loaded_meshes, loaded_weights = load_bowling()
    table_mesh, bowling_mesh, wood_block_mesh = loaded_meshes
    table_weight, bowling_weight, wood_block_weight = loaded_weights

    # place the static objects (table) into the scene
    table = sl.Object(table_mesh)
    table.set_pose(CONSTANTS.TABLE_POSE)
    table.mass = table_weight
    table.static = True
    add_obj_to_scene(scene, table)

    dynamic_objects = []

    # assemble a pyramid of wooden blocks
    for wb_pose in CONSTANTS.WOOD_BLOCK_POSES:
        wood_block = sl.Object(wood_block_mesh)
        wood_block.set_pose(wb_pose)
        wood_block.mass = wood_block_weight
        dynamic_objects.append(wood_block)
        add_obj_to_scene(scene, wood_block)

    bowling_ball = sl.Object(bowling_mesh)
    bp = bowling_ball.pose()
    bp[:3, 3] = torch.tensor([-0.9, 0, 1.25])
    bowling_ball.set_pose(bp)
    bowling_ball.mass = bowling_weight
    bowling_ball.linear_velocity = CONSTANTS.BOWLING_INITIAL_VELOCITY
    dynamic_objects.append(bowling_ball)
    add_obj_to_scene(scene, bowling_ball)

    main_cam = Camera("main", CONSTANTS.BOWLING_CAM_POS, CONSTANTS.CAM_LOOKAT, moving=False)
    bowling_scenario = Scenario(
        name="Bowling",
        scene=scene,
        cameras=[main_cam],
        static_objects=[table],
        dynamic_objects=dynamic_objects,
    )

    return bowling_scenario
