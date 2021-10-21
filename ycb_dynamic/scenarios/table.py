import stillleben as sl
import random
import torch

import ycb_dynamic.CONSTANTS as CONSTANTS
from ycb_dynamic.lighting import get_default_light_map
from ycb_dynamic.object_models import load_table_and_ycbv
from ycb_dynamic.camera import Camera
from ycb_dynamic.scenarios.scenario import Scenario, add_obj_to_scene


def setup_table_scene(cfg, scene):
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
    loaded_meshes, loaded_mesh_weights = load_table_and_ycbv()
    table_mesh, obj_meshes = loaded_meshes
    table_weight, obj_weights = loaded_mesh_weights

    # place the static objects (table) into the scene
    table = sl.Object(table_mesh)
    table.set_pose(CONSTANTS.TABLE_POSE)
    table.mass = table_weight
    table.static = True
    add_obj_to_scene(scene, table)

    # drop 10 random YCB-Video objects onto the table
    dynamic_objects = []
    for (mesh, weight) in random.choices(list(zip(obj_meshes, obj_weights)), k=10):
        obj = sl.Object(mesh)
        p = obj.pose()
        x = random.uniform(CONSTANTS.DROP_LIMITS["x_min"], CONSTANTS.DROP_LIMITS["x_max"])
        y = random.uniform(CONSTANTS.DROP_LIMITS["y_min"], CONSTANTS.DROP_LIMITS["y_max"])
        z = random.uniform(CONSTANTS.DROP_LIMITS["z_min"], CONSTANTS.DROP_LIMITS["z_max"])
        p[:3, 3] = torch.tensor([x, y, z])
        obj.set_pose(p)
        obj.mass = weight
        dynamic_objects.append(obj)
        add_obj_to_scene(scene, obj)

    main_cam = Camera("main", CONSTANTS.CAM_POS, CONSTANTS.CAM_LOOKAT, moving=False)
    table_scenario = Scenario(
        name="Table",
        scene=scene,
        cameras=[main_cam],
        static_objects=[table],
        dynamic_objects=dynamic_objects,
    )

    return table_scenario
