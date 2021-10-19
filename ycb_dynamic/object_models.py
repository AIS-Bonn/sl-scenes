import stillleben as sl
from collections import namedtuple
from pathlib import Path
import torch

MESH_BASE_DIR = Path(".") / "external_data" / "object_models"
FLAG_CONCAVE = (1 << 0)
ObjectInfo = namedtuple('ObjectInfo', ['name', 'mesh_fp', 'weight', 'flags', 'metallic', 'roughness', 'scale'])

# source: http://www.ycbbenchmarks.com/wp-content/uploads/2015/09/object-list-Sheet1.pdf
# BOP models use millimeter units for some strange reason. sl is completely metric, so scale accordingly.
OBJECT_INFO = [
    ObjectInfo('002_master_chef_can',   'ycbv_models/models_fine/obj_000001.ply',         0.414, 0,             0.6, 0.2, 0.001),
    ObjectInfo('003_cracker_box',       'ycbv_models/models_fine/obj_000002.ply',         0.411, 0,             0.1, 0.5, 0.001),
    ObjectInfo('004_sugar_box',         'ycbv_models/models_fine/obj_000003.ply',         0.514, 0,             0.1, 0.5, 0.001),
    ObjectInfo('005_tomato_soup_can',   'ycbv_models/models_fine/obj_000004.ply',         0.349, 0,             0.1, 0.5, 0.001),
    ObjectInfo('006_mustard_bottle',    'ycbv_models/models_fine/obj_000005.ply',         0.603, 0,             0.3, 0.5, 0.001),
    ObjectInfo('007_tuna_fish_can',     'ycbv_models/models_fine/obj_000006.ply',         0.171, 0,             0.6, 0.2, 0.001),
    ObjectInfo('008_pudding_box',       'ycbv_models/models_fine/obj_000007.ply',         0.187, 0,             0.1, 0.5, 0.001),
    ObjectInfo('009_gelatin_box',       'ycbv_models/models_fine/obj_000008.ply',         0.097, 0,             0.1, 0.5, 0.001),
    ObjectInfo('010_potted_meat_can',   'ycbv_models/models_fine/obj_000009.ply',         0.370, 0,             0.6, 0.3, 0.001),
    ObjectInfo('011_banana',            'ycbv_models/models_fine/obj_000010.ply',         0.066, 0,             0.3, 0.3, 0.001),
    ObjectInfo('019_pitcher_base',      'ycbv_models/models_fine/obj_000011.ply', 0.178 + 0.066, 0,             0.1, 0.5, 0.001),
    ObjectInfo('021_bleach_cleanser',   'ycbv_models/models_fine/obj_000012.ply',         1.131, 0,             0.1, 0.5, 0.001),
    ObjectInfo('024_bowl',              'ycbv_models/models_fine/obj_000013.ply',         0.147, FLAG_CONCAVE,  0.6, 0.3, 0.001),
    ObjectInfo('025_mug',               'ycbv_models/models_fine/obj_000014.ply',         0.118, FLAG_CONCAVE,  0.6, 0.3, 0.001),
    ObjectInfo('035_power_drill',       'ycbv_models/models_fine/obj_000015.ply',         0.895, FLAG_CONCAVE,  0.1, 0.6, 0.001),
    ObjectInfo('036_wood_block',        'ycbv_models/models_fine/obj_000016.ply',         0.729, 0,             0.3, 0.5, 0.001),
    ObjectInfo('037_scissors',          'ycbv_models/models_fine/obj_000017.ply',         0.082, 0,             0.1, 0.5, 0.001),
    ObjectInfo('040_large_marker',      'ycbv_models/models_fine/obj_000018.ply',         0.016, 0,             0.1, 0.5, 0.001),
    ObjectInfo('051_large_clamp',       'ycbv_models/models_fine/obj_000019.ply',         0.125, 0,             0.1, 0.5, 0.001),
    ObjectInfo('052_extra_large_clamp', 'ycbv_models/models_fine/obj_000020.ply',         0.202, 0,             0.1, 0.5, 0.001),
    ObjectInfo('061_foam_brick',        'ycbv_models/models_fine/obj_000021.ply',         0.028, 0,             0.1, 0.7, 0.001),
    ObjectInfo('table',                 'other_models/table/table.obj',                  20.000, 0,             0.3, 0.5, 0.010)
    #ObjectInfo('062_dice',              'other_models/bowling_ball/ball.obj',            10.000, 0,             0.3, 0.1, 1.000)
]

# pre-defined object sets (TODO improve efficiency?)
YCBV_OBJECTS = [obj for obj in OBJECT_INFO if obj.name[0].isdigit()]
TABLE = [obj for obj in OBJECT_INFO if obj.name == "table"]

def mesh_flags(info : ObjectInfo):
    if info.flags >= FLAG_CONCAVE:
        return sl.Mesh.Flag.NONE
    else:
        return sl.Mesh.Flag.PHYSICS_FORCE_CONVEX_HULL


def load_meshes(objects, class_index_start=0):
    '''
    Loads the meshes corresponding to given namedtuples 'objects'.
    :param objects: The object information of the meshes to be loaded.
    :param class_index_start: If specified, class index values are assigned starting from this number.
    :return: The loaded meshes as a list, or the loaded mesh object itself if it's only one.
    '''
    paths = [(MESH_BASE_DIR / obj.mesh_fp).resolve() for obj in objects]
    scales = [obj.scale for obj in objects]
    flags = [mesh_flags(obj) for obj in objects]
    meshes = sl.Mesh.load_threaded(filenames=paths, flags=flags)

    # Setup class IDs
    for i, (mesh, scale) in enumerate(zip(meshes, scales)):
        pt = torch.eye(4)
        pt[:3,:3] *= scale
        mesh.pretransform = pt
        mesh.class_index = class_index_start+i+1

    return meshes if len(meshes) != 1 else meshes[0]


def load_table_scenario_meshes():
    '''
    Loads the meshes required for the table scenario: A table and the YCB-Video meshes.
    :return: The loaded meshes as a tuple: (table_mesh, ycb_video_meshes)
    '''
    return load_meshes(TABLE), load_meshes(YCBV_OBJECTS, class_index_start=1)