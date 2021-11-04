from collections import namedtuple

#########################
# Object Information
#########################
FLAG_CONCAVE = 1 << 0
ObjectInfo = namedtuple(
    "ObjectInfo",
    ["name", "mesh_fp", "weight", "flags", "metallic", "roughness", "restitution", "scale"],
)

"""
YCB object weight sources: http://www.ycbbenchmarks.com/wp-content/uploads/2015/09/object-list-Sheet1.pdf
A few notes on the 'scale' parameter: stillleben is completely metric, so non-metric meshes need to be scaled:
 - YCB-Video (BOP version) in millimeters -> scale = 0.001
 - YCB Objects (the originals) in meters -> scale = 1.0
 - Other objects in centimeters -> scale = 0.01
 - However: you can scale all objects according to your needs (don't forget the weight)!
"""
OBJECT_INFO = [
    ObjectInfo(
        "002_master_chef_can",
        "ycbv_models/models_fine/obj_000001.ply",
        0.414,
        0,
        0.6,
        0.2,
        0.3,
        0.001,
    ),
    ObjectInfo(
        "003_cracker_box",
        "ycbv_models/models_fine/obj_000002.ply",
        0.411,
        0,
        0.1,
        0.5,
        0.1,
        0.001,
    ),
    ObjectInfo(
        "004_sugar_box",
        "ycbv_models/models_fine/obj_000003.ply",
        0.514,
        0,
        0.1,
        0.5,
        0.1,
        0.001,
    ),
    ObjectInfo(
        "005_tomato_soup_can",
        "ycbv_models/models_fine/obj_000004.ply",
        0.349,
        0,
        0.1,
        0.5,
        0.2,
        0.001,
    ),
    ObjectInfo(
        "006_mustard_bottle",
        "ycbv_models/models_fine/obj_000005.ply",
        0.603,
        0,
        0.3,
        0.5,
        0.3,
        0.001,
    ),
    ObjectInfo(
        "007_tuna_fish_can",
        "ycbv_models/models_fine/obj_000006.ply",
        0.171,
        0,
        0.6,
        0.2,
        0.3,
        0.001,
    ),
    ObjectInfo(
        "008_pudding_box",
        "ycbv_models/models_fine/obj_000007.ply",
        0.187,
        0,
        0.1,
        0.5,
        0.1,
        0.001,
    ),
    ObjectInfo(
        "009_gelatin_box",
        "ycbv_models/models_fine/obj_000008.ply",
        0.097,
        0,
        0.1,
        0.5,
        0.1,
        0.001,
    ),
    ObjectInfo(
        "010_potted_meat_can",
        "ycbv_models/models_fine/obj_000009.ply",
        0.370,
        0,
        0.6,
        0.3,
        0.2,
        0.001,
    ),
    ObjectInfo(
        "011_banana",
        "ycbv_models/models_fine/obj_000010.ply",
        0.066,
        0,
        0.3,
        0.3,
        0.3,
        0.001,
    ),
    ObjectInfo(
        "012_strawberry",
        "ycb_models/012_strawberry/google_512k/textured.obj",
        0.066,
        0,
        0.3,
        0.3,
        0.2,
        1.000,
    ),
    ObjectInfo(
        "013_apple",
        "ycb_models/013_apple/google_512k/textured.obj",
        0.180,
        0,
        0.3,
        0.3,
        0.3,
        1.000
    ),
    ObjectInfo(
        "014_lemon",
        "ycb_models/014_lemon/google_512k/textured.obj",
        0.100,
        0,
        0.3,
        0.4,
        0.3,
        1.000
    ),
    ObjectInfo(
        "015_peach",
        "ycb_models/015_peach/google_512k/textured.obj",
        0.150,
        0,
        0.0,
        0.4,
        0.3,
        1.000
    ),
    ObjectInfo(
        "016_pear",
        "ycb_models/016_pear/google_512k/textured.obj",
        0.170,
        0,
        0.1,
        0.3,
        0.3,
        1.000
    ),
    ObjectInfo(
        "017_orange",
        "ycb_models/017_orange/google_512k/textured.obj",
        0.150,
        0,
        0.3,
        0.4,
        0.3,
        1.000
    ),
    ObjectInfo(
        "018_plum",
        "ycb_models/018_plum/google_512k/textured.obj",
        0.100,
        0,
        0.3,
        0.3,
        0.3,
        1.000
    ),
    ObjectInfo(
        "019_pitcher_base",
        "ycbv_models/models_fine/obj_000011.ply",
        0.178 + 0.066,
        0,
        0.1,
        0.5,
        0.3,
        0.001,
    ),
    ObjectInfo(
        "021_bleach_cleanser",
        "ycbv_models/models_fine/obj_000012.ply",
        1.131,
        0,
        0.1,
        0.5,
        0.2,
        0.001,
    ),
    ObjectInfo(
        "024_bowl",
        "ycbv_models/models_fine/obj_000013.ply",
        0.147,
        FLAG_CONCAVE,
        0.6,
        0.3,
        0.3,
        0.001,
    ),
    ObjectInfo(
        "025_mug",
        "ycbv_models/models_fine/obj_000014.ply",
        0.118,
        FLAG_CONCAVE,
        0.6,
        0.3,
        0.3,
        0.001,
    ),
    ObjectInfo(
        "026_sponge",
        "ycb_models/026_sponge/google_512k/textured.obj",
        0.006,
        0,
        0.0,
        1.0,
        0.5,
        1.0,
    ),
    ObjectInfo(
        "029_plate",
        "ycb_models/029_plate/google_512k/textured.obj",
        0.279,
        FLAG_CONCAVE,
        0.6,
        0.3,
        0.3,
        1.0,
    ),  # BAD QUALITY
    ObjectInfo(
        "033_spatula",
        "ycb_models/033_spatula/google_512k/textured.obj",
        0.052,
        FLAG_CONCAVE,
        0.3,
        0.5,
        0.3,
        1.0,
    ),
    ObjectInfo(
        "035_power_drill",
        "ycbv_models/models_fine/obj_000015.ply",
        0.895,
        FLAG_CONCAVE,
        0.1,
        0.6,
        0.0,
        0.001,
    ),
    ObjectInfo(
        "036_wood_block",
        "ycbv_models/models_fine/obj_000016_rotated.ply",  # "ycbv_models/models_fine/obj_000016.ply"
        0.729,
        0,
        0.3,
        0.5,
        0.3,
        0.001,
    ),  # edited
    ObjectInfo(
        "037_scissors",
        "ycbv_models/models_fine/obj_000017.ply",
        0.082,
        0,
        0.1,
        0.5,
        0.1,
        0.001,
    ),
    ObjectInfo(
        "040_large_marker",
        "ycbv_models/models_fine/obj_000018.ply",
        0.016,
        0,
        0.1,
        0.5,
        0.3,
        0.001,
    ),
    ObjectInfo(
        "042_adjustable_wrench",
        "ycb_models/042_adjustable_wrench/google_512k/textured.obj",
        0.252,
        FLAG_CONCAVE,
        0.9,
        0.3,
        0.2,
        1.0,
    ),
    ObjectInfo(
        "043_phillips_screwdriver",
        "ycb_models/043_phillips_screwdriver/google_512k/textured.obj",
        0.097,
        0,  # concave?
        0.1,
        0.5,
        0.2,
        1.0,
    ),
    ObjectInfo(
        "044_flat_screwdriver",
        "ycb_models/044_flat_screwdriver/google_512k/textured.obj",
        0.098,
        0,  # concave?
        0.1,
        0.5,
        0.2,
        1.0,
    ),
    ObjectInfo(
        "048_hammer",
        "ycb_models/048_hammer/google_512k/textured.obj",
        0.665,
        FLAG_CONCAVE,  # concave?
        0.1,
        0.5,
        0.2,
        1.0,
    ),
    ObjectInfo(
        "050_medium_clamp",
        "ycb_models/050_medium_clamp/google_512k/textured.obj",
        0.059,
        FLAG_CONCAVE,  # concave?
        0.1,
        0.5,
        0.3,
        1.0,
    ),
    ObjectInfo(
        "051_large_clamp",
        "ycbv_models/models_fine/obj_000019.ply",
        0.125,
        FLAG_CONCAVE,  # concave?
        0.1,
        0.5,
        0.3,
        0.001,
    ),
    ObjectInfo(
        "052_extra_large_clamp",
        "ycbv_models/models_fine/obj_000020.ply",
        0.202,
        FLAG_CONCAVE,  # concave?
        0.1,
        0.5,
        0.3,
        0.001,
    ),
    ObjectInfo(
        "053_mini_soccer_ball",
        "ycb_models/053_mini_soccer_ball/google_512k/textured.obj",
        0.123,
        0,
        0.1,
        0.3,
        0.7,
        1.0,
    ),
    ObjectInfo(
        "054_softball",
        "ycb_models/054_softball/google_512k/textured.obj",
        0.191,
        0,
        0.0,
        0.7,
        0.7,
        1.0,
    ),
    ObjectInfo(
        "055_baseball",
        "ycb_models/055_baseball/google_512k/textured.obj",
        0.138,
        0,
        0.1,
        0.5,
        0.4,
        1.0,
    ),
    ObjectInfo(
        "056_tennis_ball",
        "ycb_models/056_tennis_ball/google_512k/textured.obj",
        0.058,
        0,
        0.0,
        0.9,
        0.8,
        1.0,
    ),
    ObjectInfo(
        "057_racquetball",
        "ycb_models/057_racquetball/google_512k/textured.obj",
        0.041,
        0,
        0.1,
        0.6,
        1.0,
        1.0,
    ),
    ObjectInfo(
        "058_golf_ball",
        "ycb_models/058_golf_ball/google_512k/textured.obj",
        0.046,
        0,
        0.3,
        0.5,
        0.5,
        1.0,
    ),
    ObjectInfo(
        "061_foam_brick",
        "ycbv_models/models_fine/obj_000021.ply",
        0.028,
        0,
        0.1,
        0.7,
        0.5,
        0.001,
    ),
    ObjectInfo(
        "062_dice",
        "ycb_models/062_dice/google_64k/textured.obj",
        0.006,
        0,
        0.3,
        0.5,
        0.5,
        1.0
    ),
    ObjectInfo(
        "065-a_cups",
        "ycb_models/065-a_cups/google_512k/textured.obj",
        0.013,
        FLAG_CONCAVE,
        0.1,
        0.5,
        0.3,
        1.0,
    ),  # BAD QUALITY (interior)
    ObjectInfo(
        "065-b_cups",
        "ycb_models/065-b_cups/google_512k/textured.obj",
        0.014,
        FLAG_CONCAVE,
        0.1,
        0.5,
        0.3,
        1.0,
    ),  # BAD QUALITY (interior)
    ObjectInfo(
        "065-c_cups",
        "ycb_models/065-c_cups/google_512k/textured.obj",
        0.017,
        FLAG_CONCAVE,
        0.1,
        0.5,
        0.3,
        1.0,
    ),  # BAD QUALITY (interior)
    ObjectInfo(
        "065-d_cups",
        "ycb_models/065-d_cups/google_512k/textured.obj",
        0.019,
        FLAG_CONCAVE,
        0.1,
        0.5,
        0.3,
        1.0,
    ),  # BAD QUALITY (interior)
    ObjectInfo(
        "065-e_cups",
        "ycb_models/065-e_cups/google_512k/textured.obj",
        0.021,
        FLAG_CONCAVE,
        0.1,
        0.5,
        0.3,
        1.0,
    ),  # BAD QUALITY (interior)
    ObjectInfo(
        "065-f_cups",
        "ycb_models/065-f_cups/google_512k/textured.obj",
        0.026,
        FLAG_CONCAVE,
        0.1,
        0.5,
        0.3,
        1.0,
    ),  # BAD QUALITY (interior)
    ObjectInfo(
        "065-g_cups",
        "ycb_models/065-g_cups/google_512k/textured.obj",
        0.028,
        FLAG_CONCAVE,
        0.1,
        0.5,
        0.3,
        1.0,
    ),  # BAD QUALITY (interior)
    ObjectInfo(
        "065-h_cups",
        "ycb_models/065-h_cups/google_512k/textured.obj",
        0.031,
        FLAG_CONCAVE,
        0.1,
        0.5,
        0.3,
        1.0,
    ),  # BAD QUALITY (interior)
    ObjectInfo(
        "065-i_cups",
        "ycb_models/065-i_cups/google_512k/textured.obj",
        0.035,
        FLAG_CONCAVE,
        0.1,
        0.5,
        0.3,
        1.0,
    ),  # BAD QUALITY (interior)
    ObjectInfo(
        "065-j_cups",
        "ycb_models/065-j_cups/google_512k/textured.obj",
        0.038,
        FLAG_CONCAVE,
        0.1,
        0.5,
        0.3,
        1.0,
    ),  # BAD QUALITY (interior)
    ObjectInfo(
        "070-b_colored_wood_blocks",
        "ycb_models/070-b_colored_wood_blocks/google_64k/textured.obj",
        0.011,
        0,
        0.3,
        0.5,
        0.3,
        1.0,
    ),
    ObjectInfo(
        "071_nine_hole_peg_test",
        "ycb_models/071_nine_hole_peg_test/google_512k/textured.obj",
        1.435,
        0,
        0.4,
        0.5,
        0.1,
        1.0,
    ),
    ObjectInfo(
        "072-a_toy_airplane",
        "ycb_models/072-a_toy_airplane/google_512k/textured.obj",
        0.570,
        FLAG_CONCAVE,
        0.3,
        0.5,
        0.2,
        1.0,
    ),
    ObjectInfo(
        "077_rubiks_cube",
        "ycb_models/077_rubiks_cube/google_512k/textured.obj",
        0.094,
        0,
        0.1,
        0.4,
        0.1,
        1.0,
    ),
    # DECORATION
    ObjectInfo(
        "art_deco_table",
        "furniture/art_deco_table/art_deco_table.obj",
        30.0,
        FLAG_CONCAVE,
        0.5,
        0.3,
        0.0,
        0.01
    ),
    ObjectInfo(
        "bowling_ball",
        "misc/bowling_ball/bowling_ball.obj",
        7.000,
        0,
        0.3,
        0.1,
        0.3,
        0.01,
    ),
    ObjectInfo(
        "folding_table",
        "furniture/folding_table/folding_table.obj",
        10.000,
        FLAG_CONCAVE,
        0.7,
        0.5,
        0.1,
        0.01,
    ),
    ObjectInfo(
        "laundry_basket",
        "misc/laundry_basket/laundry_basket.obj",
        1.000,
        FLAG_CONCAVE,
        0.1,
        0.5,
        0.1,
        0.01,
    ),
    ObjectInfo(
        "metal_chair",
        "furniture/metal_chair/metal_chair.obj",
        3.000,
        FLAG_CONCAVE,
        0.7,
        0.5,
        0.1,
        0.01,
    ),
    ObjectInfo(
        "old_cupboard",
        "furniture/old_cupboard/old_cupboard.obj",
        30.000,
        FLAG_CONCAVE,
        0.1,
        0.7,
        0.1,
        0.01,
    ),
    ObjectInfo(
        "plastic_chair",
        "furniture/plastic_chair/plastic_chair.obj",
        3.000,
        FLAG_CONCAVE,
        0.3,
        0.5,
        0.1,
        0.01,
    ),
    ObjectInfo(
        "pool_table",
        "furniture/pool_table/pool_table.obj",
        80.000,
        FLAG_CONCAVE,
        0.3,
        0.7,
        0.0,
        0.01
    ),
    ObjectInfo(
        "red_bowl",
        "misc/red_bowl/red_bowl.obj",
        0.080,
        FLAG_CONCAVE,
        0.3,
        0.1,
        0.3,
        0.01,
    ),
    ObjectInfo(
        "shopping_cart",
        "misc/shopping_cart/shopping_cart.obj",
        10.000,
        FLAG_CONCAVE,
        0.9,
        0.3,
        0.0,
        0.01,
    ),
    ObjectInfo(
        "wicker_basket",
        "misc/wicker_basket/wicker_basket.obj",
        0.200,
        FLAG_CONCAVE,
        0.1,
        0.7,
        0.4,
        0.01,
    ),
    ObjectInfo(
        "wooden_bowl",
        "misc/wooden_bowl/wooden_bowl.obj",
        0.300,
        FLAG_CONCAVE,
        0.3,
        0.3,
        0.3,
        0.01,
    ),
    ObjectInfo(
        "wooden_chair",
        "furniture/wooden_chair/wooden_chair.obj",
        3.000,
        FLAG_CONCAVE,
        0.3,
        0.7,
        0.1,
        0.01,
    ),
    ObjectInfo(
        "wooden_cupboard",
        "furniture/wooden_cupboard/wooden_cupboard.obj",
        20.000,
        FLAG_CONCAVE,
        0.3,
        0.7,
        0.1,
        0.01,
    ),
    ObjectInfo(
        "wooden_table",
        "furniture/wooden_table/wooden_table.obj",
        25.000,
        FLAG_CONCAVE,
        0.3,
        0.7,
        0.0,
        0.01,
    ),
    # FLOORS & WALLS
    ObjectInfo(
        "bamboo_floor",
        "bamboo_wall_2-4K/floor.obj",
        1e3,
        FLAG_CONCAVE,
        0.3,
        0.7,
        0.0,
        0.01,
    ),
    ObjectInfo(
        "bamboo_wall",
        "bamboo_wall_2-4K/wall.obj",
        1e3,
        FLAG_CONCAVE,
        0.3,
        0.7,
        0.0,
        0.01,
    ),
    ObjectInfo(
        "black_tiling_floor",
        "black_tiling_36-4K/floor.obj",
        1e3,
        FLAG_CONCAVE,
        0.3,
        0.7,
        0.0,
        0.01,
    ),
    ObjectInfo(
        "black_tiling_wall",
        "black_tiling_36-4K/wall.obj",
        1e3,
        FLAG_CONCAVE,
        0.3,
        0.7,
        0.0,
        0.01,
    ),
    ObjectInfo(
        "black_marble_floor",
        "blackmarble_2-4K/floor.obj",
        1e3,
        FLAG_CONCAVE,
        0.3,
        0.7,
        0.0,
        0.01,
    ),
    ObjectInfo(
        "black_marble_wall",
        "blackmarble_2-4K/wall.obj",
        1e3,
        FLAG_CONCAVE,
        0.3,
        0.7,
        0.0,
        0.01,
    ),
    ObjectInfo(
        "carpet7_floor",
        "carpet_floor_7-4K/floor.obj",
        1e3,
        FLAG_CONCAVE,
        0.3,
        0.7,
        0.0,
        0.01,
    ),
    ObjectInfo(
        "carpet7_wall",
        "carpet_floor_7-4K/wall.obj",
        1e3,
        FLAG_CONCAVE,
        0.3,
        0.7,
        0.0,
        0.01,
    ),
    ####################
    # PREASSEMBLED ROOMS
    ####################
    ObjectInfo(
        "bedroom",
        "complete_room_s/bedroom/bedroom.obj",
        1e3,
        FLAG_CONCAVE,
        0.3,
        0.7,
        0.0,
        0.01,
    ),
    ObjectInfo(
        "kings_room",
        "complete_rooms/kings_room/kings_room.obj",
        1e3,
        FLAG_CONCAVE,
        0.3,
        0.7,
        0.0,
        0.01,
    ),
]


# ["name", "mesh_fp", "weight", "flags", "metallic", "roughness", "restitution", "scale"],
