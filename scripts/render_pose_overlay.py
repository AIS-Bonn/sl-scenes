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
from PIL import Image
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from ycb_dynamic.output import OverlayWriter
from ycb_dynamic.OBJECT_INFO import get_objects_by_class_id

def main(cfg):

    # preparation
    pred_path = Path(cfg.pred_path)
    mesh_base_path = Path(cfg.mesh_base_path)
    alpha = cfg.overlay_opacity
    if cfg.no_cuda or cfg.viewer:
        sl.init()
    else:
        sl.init_cuda()
    renderer = sl.RenderPass()
    writer = OverlayWriter(pred_path)

    with ExitStack() as stack:

        stack.enter_context(writer)

        # get predicted sequences and render accordingly
        pose_pred_sequences = sorted(
            [fn for fn in os.listdir(pred_path) if fn.startswith("dumped_pose_graph") and fn.endswith(".json")]
        )
        written_frames = 0
        for pose_pred_seq in tqdm.tqdm(pose_pred_sequences):

            # prepare files for this sequence
            pose_pred_seq_fp = str(pred_path / pose_pred_seq)
            with open(pose_pred_seq_fp, "r") as dumped_info_file:
                dumped_info = json.load(dumped_info_file)
            if not cfg.viewer:
                pose_pred_seq_out_path = pred_path / Path(pose_pred_seq).stem
                pose_pred_seq_out_path.mkdir(exist_ok=True)
                # shutil.move(pose_pred_seq_fp, pose_pred_seq_out_path / pose_pred_seq)
                (pose_pred_seq_out_path / "rgb").mkdir(exist_ok=True)
                (pose_pred_seq_out_path / "obj_mask").mkdir(exist_ok=True)
                (pose_pred_seq_out_path / "blended").mkdir(exist_ok=True)
                blended_fps = []

            # get all info needed for this sequencedol
            frame_info = dumped_info["frame_info"]
            data_origin = pred_path.parent.parent / dumped_info["sequence_dir"]
            rgb_path = data_origin / "rgb"
            with open(str(data_origin / "scene_camera.json"), "r") as scene_camera_file:
                scene_camera_info = list(json.load(scene_camera_file).items())

            # set up objects for this sequence, assuming that the involved objects don't change during the sequence
            obj_info_all_frames = [frame["objects"] for frame in frame_info]
            obj_poses_all_frames = [
                [torch.tensor(obj["pose"]).reshape(4, 4) for obj in obj_info_per_frame]
                for obj_info_per_frame in obj_info_all_frames
            ]  # list(frames) of lists(objects) of 4d homogeneous obj poses
            involved_class_ids = [int(object_dict["obj_id"]) for object_dict in obj_info_all_frames[0]]
            involved_obj_infos = get_objects_by_class_id(involved_class_ids)
            involved_meshes, involved_objects = {}, []
            for obj_info in involved_obj_infos:
                involved_mesh = involved_meshes.get(obj_info.class_id, None)
                if involved_mesh is None:
                    involved_mesh = sl.Mesh(str(mesh_base_path / obj_info.mesh_fp))
                    pt = torch.eye(4)
                    pt[:3, :3] *= obj_info.scale
                    involved_mesh.pretransform = pt
                    involved_meshes[obj_info.class_id] = involved_mesh
                obj = sl.Object(involved_mesh)
                involved_objects.append(obj)

            # setup scene for this sequence
            scene_resolution = scene_camera_info[0][1]["cam_viewport"]
            scene = sl.Scene(scene_resolution)
            for obj in involved_objects:
                scene.add_object(obj)

            # set, render and save for each frame
            for t, (obj_poses_all_objs, (_, cam_info)) in enumerate(zip(obj_poses_all_frames, scene_camera_info)):

                # predicted objects are highlighted in green in context frames and in red in predicted frames
                if t >= (cfg.input_frames + cfg.pred_frames): break
                overlay_color = torch.tensor([[[1.0, 0.0, 0.0]]]) if t > cfg.input_frames \
                    else torch.tensor([[[0.0, 1.0, 0.0]]])  # shape [1, 1, 3]

                # prepare and render frame-specific scene
                cam_P = torch.tensor(cam_info["cam_P"]).reshape(4, 4)
                cam_pose = torch.tensor(cam_info["cam_pose"]).reshape(4, 4)
                for obj, obj_pose in zip(involved_objects, obj_poses_all_objs):
                    obj_pose[:3,  3] /= 1000.0  # pose is saved in millimeters, sl wants meters
                    obj.set_pose(obj_pose)
                scene.set_camera_projection(cam_P)
                scene.set_camera_pose(cam_pose)
                scene.ambient_light = torch.tensor([1.0, 1.0, 1.0])
                result = renderer.render(scene)

                # render or view
                if cfg.viewer:
                    if t >= cfg.input_frames:
                        sl.view(scene)
                else:
                    # get GT rgb image, predicted poses image and its object mask
                    bg_image = np.array(Image.open(str(rgb_path / f"{t:06d}.jpg"))).astype('float32')
                    bg_image = (2 * torch.from_numpy(bg_image) / 255) - 1
                    rgb_frame_out_fn = str(pose_pred_seq_out_path / "rgb" / f"{t:06d}.png")
                    predicted_rgb =  writer.write_rgb(result, rgb_frame_out_fn)
                    predicted_rgb = (2 * predicted_rgb / 255) - 1
                    obj_mask_out_fn = str(pose_pred_seq_out_path / "obj_mask" / f"{t:06d}.png")
                    predicted_obj_masks = writer.write_obj_mask(result, obj_mask_out_fn)
                    use_blended = predicted_obj_masks > 0
                    # TODO value ranges of images?

                    # blend GT rgb image and predicted object poses, emphasizing predictions with a color blend
                    overlay_img = bg_image.clone()
                    fg_image = (overlay_color.expand_as(predicted_rgb) + predicted_rgb) / 2
                    blended = (1 - alpha) * bg_image + alpha * fg_image
                    overlay_img[use_blended] = blended[use_blended]
                    overlay_img = (255 * (overlay_img + 1) / 2).to(torch.uint8)
                    overlay_img_out_fn = str(pose_pred_seq_out_path / "blended" / f"{t:06d}.png")
                    writer.saver.save(overlay_img, overlay_img_out_fn)
                    blended_fps.append(overlay_img_out_fn)
                    written_frames += 1

            # merge all generated blended frames into a single video file
            blended_clip = ImageSequenceClip(blended_fps, fps=10)
            blended_clip.write_videofile(str(pose_pred_seq_out_path / "blended_sequence.mp4"))
            blended_clip.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pred-path", required=True, type=str, default="", help=""
    )
    parser.add_argument(
        "--mesh-base-path", type=str, default="external_data/object_models"
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
    parser.add_argument("--input-frames", type=int, required=True)
    parser.add_argument("--pred-frames", type=int, required=True)
    parser.add_argument("--resolution", nargs='+', type=int, default=(1280, 800))
    parser.add_argument(
        "--overlay-opacity", type=float, default=0.5,
        help="Opacity value for the pose prediction overlay. 0 = no prediction visible, 1 = pose predictions opaque"
    )
    cfg = parser.parse_args()
    main(cfg)