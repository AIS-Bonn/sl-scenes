"""
Iterating through all sequences in a data directory, computing data stats for
each sequence (#instances, #activities, ...), cleaning stats and saving them
"""

import os
import copy
from tqdm import tqdm
import json
import argparse
import numpy as np
import pandas as pd

import ycb_dynamic.utils.utils as utils


def get_args():
    """ Reading command line arguments """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_path",
        required=True,
        help="Directory with the sequences to extract the stats from. Relative to root directory",
    )
    cfg = parser.parse_args()

    assert os.path.exists(os.path.join(os.getcwd(), cfg.data_path)), \
            f"Data path {cfg.data_path} does not exist..."

    return cfg


class DatasetStats:
    """
    Object for computing and accumulating the dataset stats
    """

    def __init__(self, data_path):
        """ """
        self.data_path = data_path
        self.gt_stats = {
            "num_frames": [],
            "num_instances": [],
            "freq_objects": []
        }
        self.info_stats = {
            "num_pixels": [],
            "bbox_size": [],
            "pixel_visibility": [],
            "bbox_visibility": []
        }

        self.all_stats = {}
        return

    # NOTE: Will the objects always be the same for all frames, or will that eventually change?
    def compute_gt_stats(self, file):
        """
        Computing some stats from the scene_gt.json file
        """
        assert file.split("_")[-1] == "gt.json", f"Wrong gt file {os.path.basename(file)}..."
        with open(file, "r") as f:
            gt_data = json.load(f)

        # fetching object ids and num objects for sequence
        data = gt_data[list(gt_data.keys())[0]]
        obj_ids = [obj["obj_id"] for obj in data]
        unique_ids, counts = np.unique(obj_ids, return_counts=True)
        num_instances = len(data)
        num_frames = len(gt_data)

        # accumulating
        gt_stats = copy.deepcopy(self.gt_stats)
        gt_stats["num_frames"] = num_frames
        gt_stats["num_instances"] = num_instances
        gt_stats["freq_objects"].append({int(id): int(count) for id, count in zip(unique_ids, counts)})

        return gt_stats

    def compute_info_stats(self, file):
        """
        Computing some stats from the scene_gt_info.json file
        """
        assert file.split("_")[-1] == "info.json", f"Wrong gt_info file {os.path.basename(file)}..."
        with open(file, "r") as f:
            info_data = json.load(f)

        # NOTE: What frequency do we want, framewise or sequence wise?. Let"s go sequencewise for now
        # fetching framewise pixel and bbox information
        cur_pixels, cur_bbox, cur_pixel_vis, cur_bbox_vis = [], [], [], []
        for _, data in info_data.items():
            for obj in data:
                cur_pixels.append(obj["px_count_all"])
                cur_bbox.append(obj["bbox_obj"][2] * obj["bbox_obj"][3])
                cur_pixel_vis.append(obj["visib_fract"])
                cur_bbox_vis.append(
                        self._get_bbox_vis(bbox=obj["bbox_obj"], bbox_vis=obj["bbox_visib"])
                    )

        info_stats = copy.deepcopy(self.info_stats)
        info_stats["num_pixels"].append( np.mean(cur_pixels) )
        info_stats["bbox_size"].append( np.mean(cur_bbox) )
        info_stats["pixel_visibility"].append( np.mean(cur_pixel_vis) )
        info_stats["bbox_visibility"].append( np.mean(cur_bbox_vis) )

        return info_stats

    def accumulate_stats(self, seq_path):
        """
        Computing stats for sequence
        """
        # files
        seq_name = os.path.basename(seq_path)
        scene = seq_name.split("_")[1]
        gt_file = os.path.join(seq_path, "scene_gt.json")
        info_file = os.path.join(seq_path, "scene_gt_info.json")

        # computing statistics from each of the files
        gt_stats = self.compute_gt_stats(gt_file)
        info_stats = self.compute_info_stats(info_file)

        # aggregating
        seq_num = len(self.all_stats)
        cur_stats = {}
        cur_stats["scene"] = scene
        cur_stats["seq_name"] = seq_name
        cur_stats["gt_stats"] = gt_stats
        cur_stats["info_stats"] = info_stats
        self.all_stats[str(seq_num)] = cur_stats

        return

    def compute_avg_stats(self, scene=None):
        """ Computing overall average stats considering all sequences """

        stats = self.all_stats if scene is None else {k: v for k, v in self.all_stats.items() if v["scene"] == scene}
        df = pd.DataFrame.from_dict(stats, orient="index")

        avg_stats = {}

        # frequency of each activity
        scene_counts = df["scene"].value_counts().to_dict()
        norm_scene_counts = (df["scene"].value_counts() / df["scene"].count()).to_dict()
        avg_stats["scene_counts"] = scene_counts
        avg_stats["norm_scene_counts"] = norm_scene_counts

        # mean/min/max number of frames and instances
        frames = [stats[n]["gt_stats"]["num_frames"] for n in stats.keys()]
        instances = [stats[n]["gt_stats"]["num_instances"] for n in stats.keys()]
        avg_stats["frames"] = self._add_stats(frames)
        avg_stats["instances"] = self._add_stats(instances)

        # frequency of each object
        freqs = [freqs for n in stats.keys() for freqs in stats[n]["gt_stats"]["freq_objects"]]
        df = pd.DataFrame(freqs)
        obj_freqs = df.sum(0).to_dict()
        norm_obj_freqs = (df.sum(0) / df.sum().sum()).to_dict()
        avg_stats["obj_freqs"] = obj_freqs
        avg_stats["norm_obj_freqs"] = norm_obj_freqs

        # mean/min/max of annotated pixels, bbox and corresponding visibilities
        pix = [p for n in stats.keys() for p in stats[n]["info_stats"]["num_pixels"]]
        bbox = [b for n in stats.keys() for b in stats[n]["info_stats"]["bbox_size"]]
        pix_vis = [p for n in stats.keys() for p in stats[n]["info_stats"]["pixel_visibility"]]
        bbox_vis = [b for n in stats.keys() for b in stats[n]["info_stats"]["bbox_visibility"]]
        avg_stats["pix"] = self._add_stats(pix)
        avg_stats["bbox"] = self._add_stats(bbox)
        avg_stats["pix_visibility"] = self._add_stats(pix_vis)
        avg_stats["bbox visibility"] = self._add_stats(bbox_vis)

        return avg_stats

    def save_stats(self, path=None):
        """
        Saving stats into json files
        """
        path = path if path is not None else self.data_path
        # average stats for each sequence
        all_stats_file = os.path.join(path, "all_stats.json")
        with open(all_stats_file, "w") as f:
            json.dump(self.all_stats, f)

        # stats for each activity independently
        df = pd.DataFrame.from_dict(self.all_stats, orient="index")
        activities = df["scene"].unique()
        for activity in activities:
            activity_stats_file = os.path.join(path, f"stats_{activity}.json")
            activity_stats = self.compute_avg_stats(scene=activity)
            with open(activity_stats_file, "w") as f:
                json.dump(activity_stats, f)

        # overall average stats
        avg_stats_file = os.path.join(path, "avg_stats.json")
        avg_stats = self.compute_avg_stats()
        with open(avg_stats_file, "w") as f:
            json.dump(avg_stats, f)

        return

    def _add_stats(self, data):
        """ Getting mean, std, max, min for a list of measurements """
        stats = {
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data))
        }
        return stats

    def _get_bbox_vis(self, bbox, bbox_vis):
        """ Computing the percentage of the bbox that is visible """
        # NOTE: Im assuming GT bboxes are parameterized as (x0, y0, H, W)
        area_bbox = bbox[2] * bbox[3]
        area_vis = bbox_vis[2] * bbox_vis[3]
        bbox_vis = area_vis / area_bbox if area_bbox > 0 else 0
        return bbox_vis


def main(cfg):
    """ """
    data_path = cfg.data_path
    seq_paths = [os.path.join(data_path, dir) for dir in sorted(os.listdir(data_path))]
    statsCalculator = DatasetStats(data_path)

    for seq_path in tqdm(seq_paths):
        if(not os.path.isdir(seq_path)):
            continue
        statsCalculator.accumulate_stats(seq_path=seq_path)

    statsCalculator.save_stats()

    return


if __name__ == "__main__":
    utils.clear_cmd()
    cfg = get_args()
    main(cfg)
