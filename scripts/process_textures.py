import os, glob, re
import shutil
from pathlib import Path

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


if __name__ == '__main__':

    dirs = [dir for dir in os.listdir(str(BASE_DIR)) if os.path.isdir(str(BASE_DIR / dir))]
    for dir in sorted(dirs):

        # copy floor and wall object files
        if os.path.exists(str(BASE_DIR / dir / "floor.obj")):
            os.remove(str(BASE_DIR / dir / "floor.obj"))
        shutil.copy(str(BASE_DIR / "floor.obj"), str(BASE_DIR / dir / "floor.obj"))
        if os.path.exists(str(BASE_DIR / dir / "wall.obj")):
            os.remove(str(BASE_DIR / dir / "wall.obj"))
        shutil.copy(str(BASE_DIR / "wall.obj"), str(BASE_DIR / dir / "wall.obj"))

        # prepare file paths and identifiers
        texture_files = glob.glob(f'{str(BASE_DIR)}/{dir}/*.png', recursive=True) \
                        + glob.glob(f'{str(BASE_DIR)}/{dir}/*.jpg', recursive=True) \
                        + glob.glob(f'{str(BASE_DIR)}/{dir}/*.jpeg', recursive=True)
        texture_file_keys = [re.sub("[-_]+",
                          "_",
                                    fn.split("/")[-1].replace("4K", ""))
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


