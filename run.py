import stillleben as sl
from pathlib import Path
from ycb_dynamic.scenarios.table import setup_table_scene
from ycb_dynamic.output import Writer


# ==============================================================================


SIM_STEPS_PER_SECOND = 500
SIM_STEPS_PER_FRAME = 25
SIM_DT = 1.0 / SIM_STEPS_PER_SECOND
FPS = SIM_STEPS_PER_SECOND / SIM_STEPS_PER_FRAME
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
    for it in range(cfg.iterations):
        for scenario_id in scenario_ids:
            scene = sl.Scene(RESOLUTION)  # (re-)initialize scene
            scenario = SCENARIOS[scenario_id](cfg, scene)  # populate scene according to scenario
            run_and_render_scene(cfg, renderer, scenario, it)

    return


def run_and_render_scene(cfg, renderer, scenario, it):
    '''
    TODO Doc
    :param cfg:
    :param scene:
    :return:
    '''

    # TODO: sl.view() COULD be called if not using a different graphics device or SL started in CPU mode -> add functionality?
    scene = scenario.scene
    cameras = scenario.cameras
    writers = [Writer(cfg.out_path / f"{it:06}_{scenario.name}_{cam.name}") for cam in cameras]
    for t in range(SIM_STEPS_PER_FRAME * NUM_FRAMES):

        # save visualizations every SIM_STEPS_PER_FRAME sim steps
        if t % SIM_STEPS_PER_FRAME == 0:
            for writer, cam in zip(writers, cameras):
                scene.set_camera_look_at(position=cam.pos, look_at=cam.lookat)
                result = renderer.render(scene)
                writer.write_frame(scenario, result)
                cam.step()  # advance camera for next step if it's a moving one

        # sim step
        scene.simulate(SIM_DT)

    if cfg.assemble_rgb:
        for writer in writers:
            writer.assemble_rgb_video(FPS)


# ==============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-path", type=str, default="out",
                        help="relative path to output directory")
    parser.add_argument("--no-cuda", action="store_true",
                        help="if specified, starts SL in CPU mode."
                             "This is useful if viewing a scene using a different graphics device")
    parser.add_argument("--iterations", type=int, default=1,
                        help="number of episodes to generate per scenario")
    parser.add_argument("--scenario", type=str, choices=list(SCENARIOS.keys()) + ["all"],
                        help="specify which scenario to create and render")
    parser.add_argument("--assemble_rgb", action="store_true",
                        help="if specified, creates mp4 video files from the RGB frames of an episode")

    cfg = parser.parse_args()
    if cfg.iterations < 1:
        print("iterations < 1 => exiting...")
        exit(0)
    main(cfg)
