import stillleben as sl
import pathlib
import random
import torch
from PIL import Image
from collections import namedtuple
from pathlib import Path


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
    #ObjectInfo('062_dice',              'other_models/bowling_ball/ball.obj',            10.000, 0,             0.3, 0.1, 1.000)
]

OBJECT_NAMES = [ obj.name for obj in OBJECT_INFO ]
RESOLUTION = (1920, 1080)
INTRINSICS = (1066.778, 1067.487, 312.9869, 241.3109)


def mesh_flags(info : ObjectInfo):
    if info.flags >= FLAG_CONCAVE:
        return sl.Mesh.Flag.NONE
    else:
        return sl.Mesh.Flag.PHYSICS_FORCE_CONVEX_HULL


def run():
    ibl_path = Path(".") / "external_data" / "ibl" / "Chiricahua_Plaza" / "Chiricahua_Plaza.ibl"
    mesh_path = Path(".") / "external_data"

    print("scene setup...")
    sl.init() # use sl.init_cuda() for CUDA interop
    scene = sl.Scene(RESOLUTION)
    #scene.set_camera_intrinsics(*INTRINSICS)
    #scene.ambient_light = torch.tensor([0.7, 0.7, 0.7])
    #scene.set_camera_look_at(position=torch.Tensor([1., 0., 0.]), look_at=torch.Tensor([0, 0, 0]))
    scene.light_map = sl.LightMap(ibl_path)
    #scene.choose_random_light_position()

    scene.background_plane_size = torch.tensor([3.0, 3.0])
    scene.background_color = torch.tensor([0.1, 0.1, 0.1, 1.0])

    # Load all meshes
    print("loading meshes...")
    ycb_mesh_paths = [ (mesh_path / obj.mesh_fp).resolve() for obj in OBJECT_INFO]
    mesh_scales = [obj.scale for obj in OBJECT_INFO]
    flags = [ mesh_flags(info) for info in OBJECT_INFO[1:] ]
    meshes = sl.Mesh.load_threaded(filenames=ycb_mesh_paths, flags=flags)

    # Setup class IDs
    for i, (mesh, scale) in enumerate(zip(meshes, mesh_scales)):
        pt = torch.eye(4)
        pt[:3,:3] *= scale
        mesh.pretransform = pt
        mesh.class_index = i+1

    for mesh in random.sample(meshes[-10:], 10):
        obj = sl.Object(mesh)
        scene.add_object(obj)
        print(obj.mass)

    # Let them fall in a heap
    scene.simulate_tabletop_scene()

    # Display interactive viewer
    sl.view(scene)

    # Render a frame
    renderer = sl.RenderPass()
    result = renderer.render(scene)

    # Save as JPEG
    # Image.fromarray(result.rgb()[:,:,:3].cpu().numpy()).save('rgb.jpeg')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    run()