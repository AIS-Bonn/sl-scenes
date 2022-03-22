import os, glob, re, sys
sys.path.append(".")
from pathlib import Path

from sl_cutscenes.utils.utils import copy_overwrite

BASE_DIR = Path('external_data') / 'textures'

MTL_STRINGS = {
    "ambientocclusion": "map_Ka {}",
    "opacity": "map_d {}",
    "roughness": "map_Ns {}",
    "specular": "map_Ks {}",
    "diffuse": "map_Kd {}",
    "basecolor": "map_Kd {}",
    "glossiness": "map_Pr {}",
    "displacement": "disp {}",
    "height": "disp {}",
    "metallic": "map_Pm {}",
    "normal": "norm {}"
}

def elaborate(input):
    if input == "d": return "diffuse"
    elif input == "ao": return "ambientocclusion"
    elif input == "n": return "normal"
    elif input == "h": return "height"
    return input


def process_textures():
    '''
    Skims through BASE_DIR, creating several obj files for each found texture information directory
    by creating and filling the corresponding .obj and .mtl files
    '''
    dirs = [dir for dir in os.listdir(str(BASE_DIR)) if os.path.isdir(str(BASE_DIR / dir))]
    for dir in sorted(dirs):

        # erase existing obj files
        existing_objs = [
            str(BASE_DIR / dir / obj_file) for obj_file in os.listdir(str(BASE_DIR / dir))
            if obj_file.endswith(".obj") or obj_file.endswith(".sl_mesh")
        ]
        for existing_obj in existing_objs:
            os.remove(existing_obj)

        # copy floor and wall object files
        obj_files = ["floor_6m.obj", "floor_6m_tiled.obj", "wall_6m.obj", "wall_6m_tiled.obj"]
            # + ["floor_6m_flat.obj", "floor_6m_flat_tiled.obj", "wall_6m_flat.obj", "wall_6m_flat_tiled.obj"]

        for obj_file in obj_files:
            copy_overwrite(str(BASE_DIR / obj_file), str(BASE_DIR / dir / obj_file))

        # prepare file paths and identifiers
        texture_files = glob.glob(f'{str(BASE_DIR)}/{dir}/*.png', recursive=True) \
                        + glob.glob(f'{str(BASE_DIR)}/{dir}/*.jpg', recursive=True) \
                        + glob.glob(f'{str(BASE_DIR)}/{dir}/*.jpeg', recursive=True)
        texture_file_keys = [re.sub("[-_]+", "_", fn.split("/")[-1].replace("4K", ""))
                                 .replace("_.", ".")
                                 .replace(" ", "")
                                 .split("_")[-1].split(".")[0]
                                 .lower()
                                 .replace("level", "")
                             for fn in texture_files]
        texture_file_keys = [elaborate(m) for m in texture_file_keys]

        # set up illumination mode
        illum_mode = 2
        if "metallic" in texture_file_keys or "roughness" in texture_file_keys:
            if "opacity" in texture_file_keys:
                illum_mode = 4
            else:
                illum_mode = 3

        # assemble mtl file lines from identifiers
        mtl_file_lines = ["newmtl texture_mat", f"illum {illum_mode}", ""]
        for t_file, t_file_key in zip(texture_files, texture_file_keys):
            mtl_file_lines.append(MTL_STRINGS[t_file_key].format(t_file.split("/")[-1]))
        mtl_file_lines = [line + "\n" for line in mtl_file_lines]

        # write mtl file
        with open(BASE_DIR / dir / "texture_mat.mtl", "w") as mtl_file:
            mtl_file.writelines(mtl_file_lines)


if __name__ == '__main__':
    process_textures()