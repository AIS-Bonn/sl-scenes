from pathlib import Path
from contextlib import ExitStack
import tqdm
import stillleben as sl

import ycb_dynamic.utils.utils as utils
from ycb_dynamic.scenarios import (
    StackScenario,
    BowlingScenario,
    BillardsScenario,
    ThrowScenario,
    BowlScenario
)
from ycb_dynamic.output import Writer


SCENARIOS = {
    "stack": StackScenario,
    "bowling": BowlingScenario,
    "billards": BillardsScenario,
    "throw": ThrowScenario,
    "bowl": BowlScenario
}


def main(cfg):
    """
    Main data generation loop.
    :param cfg: argparse configuration
    """

    # preparation
    if cfg.no_cuda or cfg.viewer:
        sl.init()
    else:
        sl.init_cuda()
    renderer = sl.RenderPass()

    if cfg.viewer:  # load scenario and view
        scene = sl.Scene(cfg.resolution)
        scenario = SCENARIOS[cfg.scenario](cfg, scene)
        view_scenario(cfg, renderer, scenario)

    else:  # set up scenarios and generate data
        Path(cfg.out_path).mkdir(exist_ok=True, parents=True)
        print(f"will generate {cfg.iterations} episodes per scenario")
        scenario_ids = SCENARIOS.keys() if cfg.scenario == "all" else [cfg.scenario]
        for it in range(cfg.iterations):
            for scenario_id in scenario_ids:
                scene = sl.Scene(cfg.resolution)  # (re-)initialize scene
                scenario = SCENARIOS[scenario_id](
                    cfg, scene
                )  # populate scene according to scenario
                run_and_render_scenario(cfg, renderer, scenario, it)

    return


def view_scenario(cfg, renderer, scenario):
    scene = scenario.scene
    view_cam = scenario.cameras[0]
    scene.set_camera_look_at(position=view_cam.pos, look_at=view_cam.lookat)
    renderer.render(scene)
    sl.view(scene)


def run_and_render_scenario(cfg, renderer, scenario, it):
    """
    TODO Doc
    """

    cameras = scenario.cameras
    writers = [
        Writer(Path(cfg.out_path) / f"{it:06}_{scenario.name}_{cam.name}")
        for cam in cameras
    ]
    frame_str = "" if cfg.no_gen else f": generating {cfg.frames} frames for {len(cameras)} cameras"
    print(
        f"iteration {it}, scenario '{scenario.name}'{frame_str}"
    )

    with ExitStack() as stack:

        for writer in writers:
            stack.enter_context(writer)

        sim_steps, written_frames = 0, 0
        pbar = tqdm.tqdm(total=cfg.frames)

        while written_frames < cfg.frames:
            # after sim's prep period, save visualizations every SIM_STEPS_PER_FRAME sim steps
            if sim_steps % cfg.sim_steps_per_frame == 0 and scenario.can_render():
                for writer, cam in zip(writers, cameras):
                    scenario.set_camera_look_at(pos=cam.pos, lookat=cam.lookat)
                    result = renderer.render(scenario.scene)
                    if not cfg.no_gen:
                        writer.write_frame(scenario, result)
                    cam.step()  # advance camera for next step if it's a moving one
                    written_frames += 1
                    pbar.update(1)
                    pbar.set_postfix(sim_steps=sim_steps)

            # sim step
            scenario.simulate(1.0 / cfg.sim_steps_per_sec)
            sim_steps += 1
            # time.sleep(10)
        pbar.close()

        if cfg.assemble_rgb and not cfg.no_gen:
            for writer in writers:
                writer.assemble_rgb_video(in_fps=cfg.sim_fps, out_fps=cfg.sim_fps)


# ==============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-path", type=str, default="", help="relative path to output directory"
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="if specified, starts SL in CPU mode."
        "This is useful if viewing a scene using a different graphics device",
    )
    parser.add_argument(
        "--no-gen",
        action="store_true",
        help="if specified, only simulates (no data saving)",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="if specified, opens the viewer to view the scenarios rather than generating data",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="number of episodes to generate per scenario",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        choices=list(SCENARIOS.keys()) + ["all"],
        help="specify which scenario to create and render",
    )
    parser.add_argument(
        "--assemble-rgb",
        action="store_true",
        help="if specified, creates mp4 video files from the RGB frames of an episode",
    )
    parser.add_argument("--resolution", default=(1920, 1080))
    parser.add_argument(
        "--frames", type=int, default=180, help="number of frames generated per episode"
    )
    parser.add_argument(
        "--sim-steps-per-sec",
        type=int,
        default=1500,
        help="each simulation step uses the reciprocal of this value as input value for dt",
    )  # synpick: 500 (might not be enough)
    parser.add_argument(
        "--sim-steps-per-frame",
        type=int,
        default=25,
        help="number of sim steps passing between each frame",
    )

    cfg = parser.parse_args()

    # config preparation
    if cfg.iterations < 1:
        print("parameter 'iterations' < 1, exiting...")
        exit(0)

    if cfg.out_path == "":
        cfg.out_path = f"out/{utils.timestamp()}"

    cfg.sim_fps = (
        cfg.sim_steps_per_sec / cfg.sim_steps_per_frame
    )  # TODO 60 or even 120 for dynamic movement?

    main(cfg)
