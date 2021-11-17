import json
import sys, os, shutil
sys.path.append(".")
import argparse
from contextlib import ExitStack
from pathlib import Path

import tqdm
import stillleben as sl
import numpy as np
import torch

from ycb_dynamic.output import Writer

def main(cfg):

    # preparation
    pred_path = Path(cfg.pred_path)
    if cfg.no_cuda or cfg.viewer:
        sl.init()
    else:
        sl.init_cuda()
    renderer = sl.RenderPass()
    writer = Writer(pred_path)

    with ExitStack() as stack:

        stack.enter_context(writer)

        # get predicted sequences and render accordingly
        pose_pred_sequences = sorted([fn for fn in os.listdir(pred_path) if fn.startswith("dumped_pose_graph")])
        written_frames = 0
        for pose_pred_seq in tqdm.tqdm(pose_pred_sequences):

            # prepare files for this sequence
            pose_pred_seq_fp = str(pred_path / pose_pred_seq)
            with open(pose_pred_seq_fp, "r") as dumped_info_file:
                dumped_info = json.load(dumped_info_file)
            pose_pred_seq_out_path = pred_path / Path(pose_pred_seq).stem
            pose_pred_seq_out_path.mkdir()
            shutil.move(pose_pred_seq_fp, pose_pred_seq_out_path / pose_pred_seq)
            (pose_pred_seq_out_path / "rgb").mkdir()
            (pose_pred_seq_out_path / "obj_mask").mkdir()

            # get all info needed for this sequence
            frame_info = dumped_info["frame_info"]
            data_origin = pred_path.parent.parent / dumped_info["sequence_dir"]
            rgb_path = data_origin / "rgb"
            with open(str(data_origin / "scene_camera.json"), "r") as scene_camera_file:
                scene_camera_info = json.load(scene_camera_file).items()

            # set up objects for this sequence
            # TODO
            obj_info_all_frames = [frame["objects"] for frame in frame_info]
            involved_objects = NotImplemented  # set([int() for obj_info_frame in obj_info_all_frames for obj_info in obj_info_frame["objects"]])
            obj_poses_all_frames = NotImplemented

            # setup scene for this sequence
            scene_resolution = scene_camera_info[0]["cam_viewport"]
            scene = sl.Scene(scene_resolution)
            for object in involved_objects:
                scene.add_object(object)

            # set, render and save for each frame
            for t, (obj_poses, cam_info) in enumerate(zip(obj_poses_all_frames, scene_camera_info)):
                if t >= (cfg.in_frames + cfg.pred_frames):
                    break
                overlay_color = torch.tensor([[[1.0, 0.0, 0.0]]]) if t >= cfg.in_frames \
                    else torch.tensor([[[0.0, 1.0, 0.0]]])  # red of shape [1, 1, 3]
                cam_P = torch.tensor(cam_info["cam_P"]).reshape(4, 4)
                cam_pose = torch.tensor(cam_info["cam_pose"]).reshape(4, 4)

                for object, pose in zip(involved_objects, obj_poses):
                    object.set_pose(pose)

                scene.set_camera_projection(cam_P)
                scene.set_camera_pose(cam_pose)
                scene.ambient_light = torch.tensor([1.0, 1.0, 1.0])
                result = renderer.render(scene)
                if cfg.viewer:
                    sl.view(scene)
                else:
                    gt_rgb = LOAD_RGB(rgb_path / f"{t:06d}.jpg")
                    rgb_frame_out_fn = str(pose_pred_seq_out_path / "rgb" / f"{t:06d}.png")
                    predicted_rgb =  writer.write_rgb(result, rgb_frame_out_fn)
                    obj_mask_out_fn = str(pose_pred_seq_out_path / "obj_mask" / f"{t:06d}.png")
                    predicted_obj_masks = writer.write_obj_mask(result, obj_mask_out_fn)

                    overlay_img = gt_rgb
                    overlay_img[predicted_obj_masks > 0] = (1 - predicted_obj_masks) * overlay_img \
                                                           + predicted_obj_masks * ((overlay_color.expand_as(predicted_rgb) + predicted_rgb) / 2)
                    SAVE(overlay_img)
                    written_frames += 1
                    # TODO ALPHA-BLENDING





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pred-path", type=str, default="", help=""
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="if specified, opens the viewer to view the scenarios rather than generating data",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="if specified, starts SL in CPU mode."
        "This is useful if viewing a scene using a different graphics device",
    )
    parser.add_argument("--input-frames", type=int)
    parser.add_argument("--pred-frames", type=int)
    parser.add_argument("--resolution", nargs='+', type=int, default=(1280, 800))
    cfg = parser.parse_args()
    main(cfg)