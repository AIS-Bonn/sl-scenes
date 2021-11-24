from collections import namedtuple
import json
from typing import List

#########################
# Object Information
#########################
FLAG_CONCAVE = 1 << 0
ObjectInfo = namedtuple(
    "ObjectInfo",
    ["name", "mesh_fp", "weight", "flags", "metallic", "roughness",
     "restitution", "scale", "static_friction", "dynamic_friction", "class_id"],
)

with open('ycb_dynamic/config/objects.json') as json_file:
    obj_info_dict = json.load(json_file)

"""
YCB object weight sources: http://www.ycbbenchmarks.com/wp-content/uploads/2015/09/object-list-Sheet1.pdf
A few notes on the 'scale' parameter: stillleben is completely metric, so non-metric meshes need to be scaled:
 - YCB-Video (BOP version) in millimeters -> scale = 0.001
 - YCB Objects (the originals) in meters -> scale = 1.0
 - Other objects in centimeters -> scale = 0.01
 - However: you can scale all objects according to your needs (don't forget the weight)!
"""
OBJECT_INFO = [ObjectInfo(name=name, **other_properties) for name, other_properties in obj_info_dict.items()]


def get_objects_by_class_id(class_ids : List[int]):
    obj_infos = []
    for class_id in class_ids:
        obj_infos += [obj_info for obj_info in OBJECT_INFO if obj_info.class_id == class_id]
    return obj_infos
