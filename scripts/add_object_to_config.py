import sys, os
sys.path.append(".")
import json

def add_object_to_config():
    '''
    Starts a command line dialog to add an object to the json file of existing objects.
    One after another, object parameters like mesh filepath and weight are taken and
    added as a new dict entry to the json file.
    '''
    with open('ycb_dynamic/assets/objects.json') as json_file:
        obj_info_dict = json.load(json_file)

    obj_name = input("new object name: ")
    while obj_name in obj_info_dict.keys() or len(obj_name) < 1:
        obj_name = input("That object name is invalid or already taken!\n"
                         "path to object's .obj file: ")

    obj_mesh_fp = input("object's mesh filepath: ")
    while not os.path.isfile(os.path.join("external_data/object_models", obj_mesh_fp)):
        obj_mesh_fp = input("That mesh filepath is not valid!\n"
                         "path to object's .obj file: ")

    obj_weight = float(input("object's weight in kg: "))
    obj_concave = int(input("object concave? [y|n]: ").lower() == "y")

    obj_dict = {"mesh_fp": obj_mesh_fp, "weight": obj_weight, "flags": obj_concave}
    for key in ["metallic", "roughness", "restitution", "scale",
                "static_friction", "dynamic_friction"]:
        obj_param = -1.0
        while obj_param < 0.0 or obj_param > 1.0:
            obj_param = input(f"object parameter '{key}' "
                                  f"(from 0 to 1, default: 0.5): ") or 0.5
            obj_param = float(obj_param)
        obj_dict[key] = obj_param

    obj_info_dict[obj_name] = obj_dict
    with open('ycb_dynamic/assets/objects.json', 'w', encoding='utf-8') as f:
        json.dump(obj_info_dict, f, ensure_ascii=False, indent=4)

    print(f"added new object '{obj_name}' to objects.json. Goodbye!")

if __name__ == '__main__':
    add_object_to_config()