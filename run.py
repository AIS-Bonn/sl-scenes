import os

import stillleben as sl
from PIL import Image
from pathlib import Path
from ycb_dynamic.scenarios.table import setup_table_scene
from ycb_dynamic.output import Writer

# ==============================================================================

SIM_DT = 1.0 / 500
SIM_STEPS_PER_FRAME = 25  # => FPS = 20
NUM_FRAMES = 100
RESOLUTION = (1920, 1080)
OUT_PATH = Path("output")
SCENARIOS = { "table": setup_table_scene }

# ==============================================================================


def main(cfg):
    '''
    Main data generation loop.
    :param cfg: argparse configuration
    '''

    # preparation
    Path(cfg.out_path).mkdir(exist_ok=True, parents=True)
    if cfg.no_cuda:
        sl.init()
    else:
        sl.init_cuda()
    renderer = sl.RenderPass()

    # set up scenarios and generate data
    scenario_ids = SCENARIOS.keys() if cfg.scenario == "all" else [cfg.scenario]
    for scenario_id in scenario_ids:
        scene = sl.Scene(RESOLUTION)  # (re-)initialize scene
        scene, camera_poses = SCENARIOS[scenario_id](cfg, scene)  # populate scene according to scenario
        run_and_render_scene(cfg, renderer, scene, camera_poses)

    return


def run_and_render_scene(cfg, renderer, scene, camera_poses):
    '''
    TODO Doc
    :param cfg:
    :param scene:
    :return:
    '''

    # TODO 1: sl.view() COULD be called if not using a different graphics device or SL started in CPU mode -> add functionality?

    # TODO 2: Create camera "objects" that can change pos, lookat etc. over time and also hold a name (for out-writing)
    writers = [Writer(out / f'{i:06}') for i in range(len(camera_poses))]
    for t in range(SIM_STEPS_PER_FRAME * NUM_FRAMES):

        # save visualizations every SIM_STEPS_PER_FRAME sim steps
        if t % SIM_STEPS_PER_FRAME == 0:
            for writer, camera_pose in zip(writers, camera_poses):
                cam_pos, cam_lookat = camera_pose
                scene.set_camera_look_at(position=cam_pos, look_at=cam_lookat)
                result = renderer.render(scene)
                writer.write_frame(scene, result)
                # TODO 3: Adjust writers to this Project

        # sim step
        scene.simulate(SIM_DT)

    # TODO 4: generate videos from generated images to quickly check generated data -> extend writer functionality?



# ==============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-path", type=str, default="out",
                        help="relative path to output directory")
    parser.add_argument("--no-cuda", action="store_true",
                        help="if specified, starts SL in CPU mode."
                             "This is useful if viewing a scene using a different graphics device")
    parser.add_argument("--scenario", type=str, choices=list(SCENARIOS.keys()) + ["all"],
                        help="specify which scenario to create and render")

    cfg = parser.parse_args()
    main(cfg)
