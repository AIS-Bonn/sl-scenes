import stillleben as sl
import random
import torch
from ycb_dynamic.lighting import get_default_light_map
from ycb_dynamic.object_models import load_table_scenario_meshes


AMBIENT_LIGHT = torch.tensor([0.7, 0.7, 0.7])
CAM_POS = torch.Tensor([0.5, 2.4, 1.5])
CAM_LOOKAT = torch.Tensor([0, 0, 1.1])
TABLE_POSE = torch.tensor([[0, 0, 1,    0],
                           [1, 0, 0,    0],
                           [0, 1, 0, 0.60],
                           [0, 0, 0,    1]])
DROP_LIMITS = {"x_min": -0.6, "x_max": 0.6,
               "y_min": -0.4, "y_max": 0.4,
               "z_min":  1.2, "z_max": 1.5}


def setup_table_scene(cfg, scene):
    '''
    TODO Doc
    :param cfg:
    :param scene:
    :return:
    '''

    print("scene setup...")
    scene.ambient_light = AMBIENT_LIGHT
    scene.light_map = get_default_light_map()
    scene.choose_random_light_position()
    scene.background_plane_size = torch.tensor([10.0, 10.0])
    scene.background_color = torch.tensor([0.1, 0.1, 0.1, 1.0])


    print("loading objects...")
    table_mesh, obj_meshes = load_table_scenario_meshes()

    # place the table into the scene
    table = sl.Object(table_mesh)
    table.set_pose(TABLE_POSE)
    scene.add_object(table)

    # drop 10 random YCB-Video objects onto the table
    for mesh in random.sample(obj_meshes, 10):
        obj = sl.Object(mesh)
        p = obj.pose()
        x = random.uniform(DROP_LIMITS["x_min"], DROP_LIMITS["x_max"])
        y = random.uniform(DROP_LIMITS["y_min"], DROP_LIMITS["y_max"])
        z = random.uniform(DROP_LIMITS["z_min"], DROP_LIMITS["z_max"])
        p[:3, 3] = torch.tensor([x, y, z])
        obj.set_pose(p)
        scene.add_object(obj)

    return scene, [(CAM_POS, CAM_LOOKAT)]
