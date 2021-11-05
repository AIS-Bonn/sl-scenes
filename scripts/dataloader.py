import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate as default_collate

def preprocess_rgb(x):
    '''
    [0, 255, np.uint8] -> [-1, 1, torch.float32]
    '''
    permutation = (2, 0, 1) if x.ndim == 3 else (0, 3, 1, 2)
    x = torch.from_numpy(x.transpose(permutation).astype('float32'))
    return (2 * x / 255) - 1

def preprocess_mask(x):
    '''
    [0, 255, np.uint8] -> [0, 255, torch.float32]
    '''
    permutation = (0, 1) if x.ndim == 2 else (2, 0, 1) if x.ndim == 3 else (0, 3, 1, 2)
    return torch.from_numpy(x.transpose(permutation).astype('float32'))


class YCBDynamicDataset(Dataset):

    def __init__(self, data_dir, mode="full"):
        super(YCBDynamicDataset, self).__init__()
        self.base_path = Path(data_dir)
        self.sequences = sorted([entry for entry in os.listdir(data_dir) if os.path.isdir(str(self.base_path / entry))])
        self.mode = mode

    def __getitem__(self, i):
        sequence_dir = self.base_path / self.sequences[i]
        print(sequence_dir)
        if self.mode == "full":
            return self.get_all(sequence_dir)
        elif self.mode == "rgb":
            return self.get_rgb(sequence_dir)
        elif self.mode == "depth":
            return self.get_mask(sequence_dir, "depth")
        else:
            raise NotImplementedError

    def get_all(self, sequence_dir : Path):
        frames_rgb = self.get_rgb(sequence_dir)
        frames_depth = self.get_mask(sequence_dir, "depth")
        frames_class_idx = self.get_mask(sequence_dir, "class_index_masks")
        frames_instance_idx = self.get_mask(sequence_dir, "instance_index_masks")
        # TODO include visibility masks once the instance_id correspondence is established
        #frames_visib = self.get_mask(sequence_dir, "mask_visib")

        with open(str(sequence_dir / "scene_camera.json"), "r") as cam_info_file:
            frames_cam_info = list(json.load(cam_info_file).values())
            frames_cam_info = {i: v for i, v in enumerate(frames_cam_info)}
        with open(str(sequence_dir / "scene_gt.json"), "r") as scene_gt_file:
            frames_scene_gt = list(json.load(scene_gt_file).values())
            frames_scene_gt = {i: v for i, v in enumerate(frames_scene_gt)}

        frame_information = {
            "rgb": frames_rgb,
            "depth": frames_depth,
            "class_idx": frames_class_idx,
            "instance_idx": frames_instance_idx,
            #"visib": frames_visib,
            "cam_info": frames_cam_info,
            "scene_gt": frames_scene_gt
        }

        return frame_information

    def get_rgb(self, sequence_dir: Path):
        frames_rgb = sorted(os.listdir(str(sequence_dir / 'rgb')))
        frames_rgb = [str(sequence_dir / 'rgb' / rgb) for rgb in frames_rgb]
        frames_rgb = [plt.imread(rgb) for rgb in frames_rgb]
        frames_rgb = [preprocess_rgb(rgb) for rgb in frames_rgb]
        return torch.stack(frames_rgb, dim=0)

    def get_mask(self, sequence_dir: Path, frame_type: str):
        frames_mask = sorted(os.listdir(str(sequence_dir / frame_type)))
        frames_mask = [str(sequence_dir / frame_type / mask) for mask in frames_mask]
        frames_mask = [plt.imread(mask) for mask in frames_mask]
        frames_mask = [preprocess_mask(mask) for mask in frames_mask]
        return torch.stack(frames_mask, dim=0)

    def __len__(self):
        return len(self.sequences)


def collate_full(batch):
    return {
        "rgb": torch.stack([item["rgb"] for item in batch], dim=0),
        "depth": torch.stack([item["depth"] for item in batch], dim=0),
        "class_idx": torch.stack([item["class_idx"] for item in batch], dim=0),
        "instance_idx": torch.stack([item["instance_idx"] for item in batch], dim=0),
        #"visib": torch.stack([item["visib"] for item in batch], dim=0),
        "cam_info": [item["cam_info"] for item in batch],
        "scene_gt": [item["scene_gt"] for item in batch]
    }

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=str, required=True
    )
    parser.add_argument(
        "--dataset-mode", type=str, choices=["rgb", "full"], default="full"
    )
    cfg = parser.parse_args()

    train_data = YCBDynamicDataset(data_dir=cfg.data_dir, mode=cfg.dataset_mode)
    loader_collate_fn = default_collate if cfg.dataset_mode == "rgb" else collate_full
    train_loader = DataLoader(train_data, batch_size=2, shuffle=True, collate_fn=loader_collate_fn)
    for batch in train_loader:
       print(batch.keys(), batch["rgb"].shape, batch["depth"].shape, batch["class_idx"].shape,
             batch["instance_idx"].shape, len(batch["cam_info"]), len(batch["scene_gt"]))
       # print(batch.shape)