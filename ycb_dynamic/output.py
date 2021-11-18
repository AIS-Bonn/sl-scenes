
"""
Methods for writing an output frame
Taken from SynPick and modified
"""

import stillleben as sl
import torch
import time
from pathlib import Path
from ycb_dynamic.scenarios.scenario import Scenario

class Writer(object):
    def __init__(self, path : Path):
        self.path = path
        self.idx = 0
        self.depth_scale = 10000.0  # depth [m] = pixel / depth_scale
        self.saver = sl.ImageSaver()

        # Create output directory
        path.mkdir(parents=True)

        (path / 'rgb').mkdir()
        (path / 'mask_visib').mkdir()
        (path / 'class_index_masks').mkdir()
        (path / 'instance_index_masks').mkdir()
        (path / 'depth').mkdir()

        self.camera_file = open(path / 'scene_camera.json', 'w')
        self.camera_file.write('{\n')

        self.gt_file = open(path / 'scene_gt.json', 'w')
        self.gt_file.write('{\n')

        self.info_file = open(path / 'scene_gt_info.json', 'w')
        self.info_file.write('{\n')

        self.log_file = open(path / 'log.txt', 'w')

        self.mask_renderer = sl.RenderPass('flat')
        self.mask_renderer.ssao_enabled = False


    def __enter__(self):
        self.saver.__enter__()
        return self


    def __exit__(self, type, value, traceback):
        # Finish camera_file
        self.camera_file.write('\n}')
        self.camera_file.close()

        # Finish gt_file
        self.gt_file.write('\n}')
        self.gt_file.close()

        # Finish info file
        self.info_file.write('\n}')
        self.info_file.close()

        # Finish log file
        self.log_file.close()

        self.saver.__exit__(type, value, traceback)


    @staticmethod
    def intrinsicMatrixFromProjection(proj : torch.tensor, W : int, H : int):
        far = -proj[2,3] / (proj[2,2] - 1.0)
        near = (proj[2,2] - 1.0) / (proj[2,2] + 1.0) * far
        left = -near * (proj[0,2]+1) / proj[0,0]
        right = -near * (proj[0,2]-1) / proj[0,0]
        bottom = -near * (proj[1,2]-1) / proj[1,1]
        top = -near * (proj[1,2]+1) / proj[1,1]

        eps = 2.2204460492503131e-16

        if abs(left-right) < eps:
            cx = W * 0.5
        else:
            cx = (left * W) / (left - right)

        if abs(top-bottom) < eps:
            cy = H * 0.5
        else:
            cy = (top * H) / (top - bottom)

        fx = -near * cx / left
        fy = -near * cy / top

        return torch.tensor([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ])


    @staticmethod
    def bbox_from_mask(mask):
        """Compute bounding boxes from masks.
        mask: [height, width]. Mask pixels are either 1 or 0.
        """

        horizontal_indices = torch.where(torch.any(mask, dim=0))[0]
        vertical_indices = torch.where(torch.any(mask, dim=1))[0]

        if len(horizontal_indices) != 0:
            x1, x2 = horizontal_indices[[0,-1]].tolist()
            y1, y2 = vertical_indices[[0,-1]].tolist()
            x2 += 1
            y2 += 1

            return x1, y1, x2-x1, y2-y1
        else:
            return 0, 0, 0, 0

    def write_log(self, *args, **kwargs):
        self.log_file.write(f'{self.idx:06}: ')
        print(*args, **kwargs, file=self.log_file)

    def write_scene_data(self, scene : sl.Scene):
        with open(self.path / 'scene.sl', 'w') as f:
            f.write(scene.serialize())

    def write_rgb(self, result : sl.RenderPassResult, out_file: str):
        rgb = result.rgb()[:,:,:3].cpu().contiguous()
        self.saver.save(rgb, out_file)
        return rgb

    def write_obj_mask(self, result : sl.RenderPassResult, out_file: str):
        instance_segmentation = result.instance_index()[:,:,0]
        instance_segmentation[instance_segmentation > 0] = 1.0
        instance_segmentation = instance_segmentation.byte().cpu()
        self.saver.save(instance_segmentation, out_file)
        return instance_segmentation

    def write_frame(self, scenario : Scenario, result : sl.RenderPassResult):

        scene = scenario.scene

        # RGB
        #t0 = time.time()
        rgb = result.rgb()[:,:,:3].cpu().contiguous()
        #t1 = time.time()
        #print(f'RGB: {t1-t0}')
        self.saver.save(rgb, str(self.path / 'rgb' / f'{self.idx:06}.jpg'))


        # Depth
        #t0 = time.time()
        depth = (result.depth() * self.depth_scale).short().cpu().contiguous()
        #t1 = time.time()
        #print(f'Depth: {t1-t0}')
        self.saver.save(depth, str(self.path / 'depth' / f'{self.idx:06}.png'))

        if self.idx != 0:
            self.info_file.write(',\n\n')
        self.info_file.write(f'  "{self.idx}": [\n')

        # Masks
        active_objects = scenario.dynamic_objects
        instance_segmentation = result.instance_index()[:,:,0].byte().cpu()
        class_index_masks, instance_index_masks = [], []

        for i, obj in enumerate(active_objects):
            if(not hasattr(obj, "instance_index")):
                continue
            mask = (instance_segmentation == obj.instance_index).byte()
            self.saver.save(mask * 255, str(self.path / 'mask_visib' / f'{self.idx:06}_{i:06}.png'))
            class_index_masks.append(mask * obj.mesh.class_index)
            instance_index_masks.append(mask * obj.instance_index)

            visib_num_pixels = mask.sum()
            visib_bbox = Writer.bbox_from_mask(mask)

            # Render this object alone
            silhouette = self.mask_renderer.render(scene, predicate=lambda o: o == obj)
            sil_mask = (silhouette.class_index()[:,:,0] != 0).byte().cpu()
            sil_num_pixels = sil_mask.sum()
            sil_bbox = Writer.bbox_from_mask(sil_mask)
            visib_fract = float(visib_num_pixels) / float(sil_num_pixels) if sil_num_pixels > 0 else 0

            if i != 0:
                self.info_file.write(',\n')

            self.info_file.write(
                f'    {{"bbox_obj": {list(sil_bbox)}, "bbox_visib": {list(visib_bbox)}, ' +
                f'"px_count_all": {int(sil_num_pixels)}, "px_count_valid": {int(sil_num_pixels)}, ' +
                f'"px_count_visib": {int(visib_num_pixels)}, "visib_fract": {visib_fract}}}'
            )

        self.info_file.write(']')

        class_index_mask = (torch.stack(class_index_masks, dim=0)).sum(dim=0).byte()
        self.saver.save(class_index_mask, str(self.path / 'class_index_masks' / f'{self.idx:06}.png'))
        instance_index_mask = torch.stack(instance_index_masks, dim=0).sum(dim=0).byte()
        self.saver.save(instance_index_mask, str(self.path / 'instance_index_masks' / f'{self.idx:06}.png'))

        # Figure out cam_K
        P = scene.projection_matrix()
        W,H = scene.viewport

        cam_K = Writer.intrinsicMatrixFromProjection(P, W, H)

        world_in_camera = torch.inverse(scene.camera_pose())
        cam_R_w2c = world_in_camera[:3,:3].contiguous()
        cam_t_w2c = world_in_camera[:3,3] * 1000.0 # millimeters, of course.

        # Write scene_camera.json
        if self.idx != 0:
            self.camera_file.write(',\n')
        self.camera_file.write(f'  "{self.idx}": {{"cam_K": {cam_K.view(-1).tolist()}, '
                               f'"cam_P": {P.flatten().tolist()}, "cam_viewport": {[W, H]}, '
                               f'"depth_scale": {1.0 / (self.depth_scale / 1000.0)}, '
                               f'"cam_pose": {scene.camera_pose().flatten().tolist()}, '
                               f'"cam_R_w2c": {cam_R_w2c.view(-1).tolist()}, "cam_t_w2c": {cam_t_w2c.tolist()}}}')

        # Write scene_gt.json
        if self.idx != 0:
            self.gt_file.write(',\n\n')

        def gt(o):
            T = o.pose()
            T_m2c = world_in_camera @ T

            cam_R = T[:3,:3].contiguous()
            cam_t = T[:3,3] * 1000.0 # millimeters, of course.

            cam_R_m2c = T_m2c[:3,:3].contiguous()
            cam_t_m2c = T_m2c[:3,3] * 1000.0 # millimeters, of course.

            return f'{{"cam_R": {cam_R.view(-1).tolist()}, "cam_t": {cam_t.tolist()}' \
                   f', "cam_R_m2c": {cam_R_m2c.view(-1).tolist()}, "cam_t_m2c": {cam_t_m2c.tolist()}' \
                   f', "obj_id": {o.mesh.class_index}, "ins_id": {o.instance_index}}}'

        formatted_gt = ",\n".join([ gt(o) for o in active_objects ])
        self.gt_file.write(f'  "{self.idx}": [\n    {formatted_gt}]')

        self.idx += 1


    def assemble_rgb_video(self, in_fps, out_fps):
        import glob
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
        import moviepy.video.fx.all as vfx

        rgb_frames = sorted(glob.glob(str(self.path / "rgb" / "*.jpg")))
        rgb_clip = ImageSequenceClip(rgb_frames, fps=in_fps)
        rgb_clip = rgb_clip.set_fps(out_fps)
        rgb_clip = rgb_clip.fx(vfx.speedx, out_fps / in_fps)  # both speedup and set_fps needed for re-setting FPS
        rgb_clip.write_videofile(str(self.path / "rgb_video.mp4"))
        rgb_clip.close()
