import json
import os
from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
from dual_quaternions import DualQuaternion

import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate as default_collate
from torch_geometric.data import Data, HeteroData, Batch
from torch_geometric_temporal.signal import StaticGraphTemporalSignal as SGTS
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal as DGTS
from torch_geometric_temporal.signal import DynamicGraphStaticSignal as DGSS
from torch_geometric_temporal.signal import StaticGraphTemporalSignalBatch as SGTSBatch
from torch_geometric_temporal.signal import DynamicGraphTemporalSignalBatch as DGTSBatch
from torch_geometric_temporal.signal import DynamicGraphStaticSignalBatch as DGSSBatch


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

def preprocess_pose(rotation: List[float], translation: List[float]):
    '''
    rotation matrix in list form + translation vector in list form -> np.array[8, np.float32]
    '''
    rotation = R.from_matrix([rotation[i:i + 3] for i in [0, 3, 6]]).as_quat().tolist()
    pose = DualQuaternion.from_quat_pose_array(rotation + translation)
    return np.array(pose.dq_array()).astype("np.float32")

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

def collate_temporal_signal(batch: List[Union[SGTS, DGTS, DGSS]]):

    # check for inconsistent input signal types
    if len(set([type(signal) for signal in batch])) > 1:
        raise ValueError("Given Signal objects are of differing type!")

    # determine output type
    in_type = type(batch[0])
    signal_batch_type = None
    if in_type == SGTS:
        signal_batch_type = SGTSBatch
    elif in_type == DGTS:
        signal_batch_type = DGTSBatch
    elif in_type == DGSS:
        signal_batch_type = DGSSBatch

    # create list(samples) of list(temporal sequence) of torch_geometric.Data/HeteroData
    all_graphs = [[g for g in iter(signal)] for signal in batch]

    # check for inconsistent sequence lengths
    if len(set([len(sample_list) for sample_list in all_graphs])) > 1:
        raise ValueError("Batch Samples of differing sequence length are currently not supported.")

    # create diagonalized torch_geometric.Batch objects by timestep
    batches_by_timestep = [
        Batch.from_data_list([sample[t] for sample in all_graphs]) for t in range(len(all_graphs[0]))
    ]

    # assemble signal of batched graphs
    return signal_batch_type(
        edge_indices = [batch["edge_index"].numpy() for batch in batches_by_timestep],
        edge_weights = [batch["edge_attr"].numpy() for batch in batches_by_timestep],
        features = [batch["x"].numpy() for batch in batches_by_timestep],
        targets = [batch["y"].numpy() for batch in batches_by_timestep],
        batches = [batch["batch"].numpy() for batch in batches_by_timestep]
    )


class YCBDynamicDataset(Dataset):

    modes_and_collate_fns = {
        "rgb": default_collate,
        "depth": default_collate,
        "rgbd": default_collate,
        "poses": collate_temporal_signal,
        "full": collate_full
    }

    def __init__(self, data_dir, img_size, mode="full"):
        super(YCBDynamicDataset, self).__init__()
        self.base_path = Path(data_dir)
        self.sequences = sorted([entry for entry in os.listdir(data_dir) if os.path.isdir(str(self.base_path / entry))])
        self.mode = mode
        self.collate_fn = self.modes_and_collate_fns[self.mode]
        self.img_size = img_size

    def __getitem__(self, i):
        sequence_dir = self.base_path / self.sequences[i]
        if self.mode == "full":
            return self.get_all(sequence_dir)
        elif self.mode == "rgb":
            return self.get_rgb(sequence_dir)
        elif self.mode == "depth":
            return self.get_mask(sequence_dir, "depth")
        elif self.mode == "rgbd":
            rgb = self.get_rgb(sequence_dir)
            depth = self.get_mask(sequence_dir, "depth").unsqueeze(dim=-3)
            return torch.cat([rgb, depth], dim=-3)
        elif self.mode == "poses":
            return self.get_object_poses(sequence_dir)
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
        frames_rgb = torch.stack(frames_rgb, dim=0)
        if frames_rgb.shape[-2:] != self.img_size:
            frames_rgb = TF.resize(frames_rgb, size=self.img_size)
        return frames_rgb

    def get_mask(self, sequence_dir: Path, frame_type: str):
        frames_mask = sorted(os.listdir(str(sequence_dir / frame_type)))
        frames_mask = [str(sequence_dir / frame_type / mask) for mask in frames_mask]
        frames_mask = [plt.imread(mask) for mask in frames_mask]
        frames_mask = [preprocess_mask(mask) for mask in frames_mask]
        frames_mask = torch.stack(frames_mask, dim=0)
        if frames_mask.shape[-2:] != self.img_size:
            frames_mask = TF.resize(frames_mask, size=self.img_size)
        return frames_mask

    def get_object_poses(self, sequence_dir: Path):
        with open(str(sequence_dir / "scene_gt.json"), "r") as scene_gt_file:
            frames_scene_gt = list(json.load(scene_gt_file).values())

        edge_indices, edge_weights, node_features, targets = [], [], [], []
        for frame_info in frames_scene_gt:
            all_instance_ids = [obj_info["ins_id"] for obj_info in frame_info]
            edge_indices_t, node_features_t, targets_t = [], [], []
            for obj_info in frame_info:

                # assemble node feature vector
                pose_dq = preprocess_pose(obj_info["cam_R_m2c"], obj_info["cam_t_m2c"])
                obj_class = np.array([obj_info["obj_id"]])
                obj_feature = np.concatenate([pose_dq, obj_class])  # object's feature (x) vector
                node_features_t.append(obj_feature)
                targets_t.append(pose_dq)

                # assemble outgoing edges
                instance_idx = obj_info["ins_id"]  # object's instance idx
                node_idx = all_instance_ids.index(instance_idx)
                touches = obj_info.get("touches", [idx for idx in all_instance_ids if idx != instance_idx])  # list of instance idx this object touches
                touched_node_idx = [all_instance_ids.index(t) for t in touches]
                edge_indices_t.extend([np.array([node_idx, touch_idx]) for touch_idx in touched_node_idx])

            node_features.append(np.stack(node_features_t, axis=0))  # shape: [|V|, graph_in_dim]
            edge_indices_t = np.stack(edge_indices_t, axis=1)
            edge_indices.append(edge_indices_t)  # shape: [2, |E|]
            edge_weights.append(np.ones(edge_indices_t.shape[1]))  # shape: [|E|]
            targets.append(np.stack(targets_t, axis=0))  # shape: [|V|, 8]

        data_signal = DGTS(
            edge_indices = edge_indices,
            edge_weights = edge_weights,
            features = node_features,
            targets = targets,
        )  # sequence consists of T graphs

        return data_signal

    def __len__(self):
        return len(self.sequences)

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=str, required=True
    )
    parser.add_argument(
        "--dataset-mode", type=str, choices=YCBDynamicDataset.modes_and_collate_fns.keys(), default="full"
    )
    parser.add_argument(
        "--img-size", nargs='+', type=int, default=(256, 160)
    )
    cfg = parser.parse_args()

    train_data = YCBDynamicDataset(data_dir=cfg.data_dir, img_size=cfg.img_size, mode=cfg.dataset_mode)
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True, collate_fn=train_data.collate_fn)
    for batch in train_loader:
        print(batch)
        # print(batch.keys(), batch["rgb"].shape, batch["depth"].shape, batch["class_idx"].shape,
        #     batch["instance_idx"].shape, len(batch["cam_info"]), len(batch["scene_gt"]))
        # print(batch.shape)