import shutil
import sys, os, argparse
sys.path.append(".")
import ycb_dynamic.utils.utils as utils
from pathlib import Path

def get_videos_from_folder(cfg):
    '''
    Extracts the videos from the folders of generated data and copies them to a separate folder.
    '''
    data_paths = [Path(dp) for dp in cfg.data_paths]
    for data_path in data_paths:
        out_path = OUT_BASE / f"{str(data_path.stem)}_videos"
        out_path.mkdir(exist_ok=True)
        seq_names = [dir for dir in os.listdir(str(data_path))
                         if os.path.isdir(str(data_path / dir))]
        for seq_name in seq_names:
            video_files = [vid_names for vid_names in os.listdir(str(data_path / seq_name))
                           if vid_names[-4:] == ".mp4"]
            srcs = [str(data_path / seq_name / vid_fn) for vid_fn in video_files]
            dsts = [str(out_path / f"{seq_name}_{vid_fn}") for vid_fn in video_files]
            for src, dst in zip(srcs, dsts):
                utils.copy_overwrite(src, dst)

def merge_folders(cfg):
    '''
    Merges two folders of generated data by adjusting the iteration index.
    '''
    data_paths = [Path(dp) for dp in cfg.data_paths]
    out_path = OUT_BASE / f"merged_{utils.timestamp()}"
    out_path.mkdir(exist_ok=True)
    iter_offset = 0
    for data_path in data_paths:
        seq_names = [dir for dir in os.listdir(data_path)
                         if os.path.isdir(str(data_path / dir))]
        for seq_name in seq_names:
            new_seq_path_it = int(seq_name[:6]) + iter_offset
            seq_path_src = str(data_path / seq_name)
            seq_path_dst = str(out_path / f"{new_seq_path_it:06d}{seq_name[6:]}")
            shutil.copytree(seq_path_src, seq_path_dst)
        iter_offset += max({int(dir[:6]) for dir in seq_names}) + 1  # increase offset by number of its

OUT_BASE = Path("out")
PROGRAMS = {
    "merge_folders": merge_folders,
    "get_videos": get_videos_from_folder
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-paths", type=str, nargs="+", help="paths to data directories"
    )
    parser.add_argument(
        "--program", type=str, choices=PROGRAMS.keys()
    )

    cfg = parser.parse_args()
    run = PROGRAMS[cfg.program]
    run(cfg)
