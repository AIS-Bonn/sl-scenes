import os
import argparse

import ycb_dynamic.utils.utils as utils
from ycb_dynamic import generate
from ycb_dynamic.scenarios import SCENARIOS
from ycb_dynamic.CONSTANTS import ALL_LIGHTMAPS

if __name__ == "__main__":
    os.system("cls" if os.name == "nt" else "clear")
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="If specified, starts stillleben in CPU mode. "
             "This is useful if viewing a scene using a different graphics device."
    )
    parser.add_argument(
        "--no-gen",
        action="store_true",
        help="If specified, only simulates the scenarios/scenes but does not save any data.",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="If specified, opens the viewer to view the scenarios rather than generating data.",
    )
    parser.add_argument(
        "--iterations",
        type=utils.positive_integer,
        default=1,
        help="Number of episodes to generate per scenario.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        choices=list(SCENARIOS.keys()),
        help="Specify which scenario to create and render.",
    )
    parser.add_argument(
        "--lights",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="Specify the amount of light sources placed around the scene (default: 0). "
             "If 0 lights are specified, a lightmap is used instead."
    )
    parser.add_argument(
        "--lightmap",
        type=str,
        default="random",
        choices=list(ALL_LIGHTMAPS.keys()) + ["random", "default"],
        help="Specify the lightmap to apply in the scene.",
    )
    parser.add_argument(
        "--assemble-rgb",
        action="store_true",
        help="If specified, creates mp4 video files from the RGB frames of an episode.",
    )
    parser.add_argument(
        "--serialize-scene",
        action="store_true",
        help="If specified, serializes each scene to a txt file before simulating."
    )
    parser.add_argument(
        "--resolution",
        nargs='+',
        type=int,
        default=(1920, 1080),
        help="The frame resolution to generate."
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=180,
        help="Number of frames generated per episode."
    )
    parser.add_argument(
        "--sim-steps-per-sec",
        type=int,
        default=600,
        help="Each simulation step uses the reciprocal of this value as input value for dt (time passed).",
    )
    parser.add_argument(
        "--sim-steps-per-frame",
        type=int,
        default=20,
        help="Number of sim steps passing between each frame.",
    )
    parser.add_argument(
        "--cameras",
        type=utils.positive_integer,
        default=1,
        help="Number of cameras to set up per scenario."
    )
    parser.add_argument(
        "--coplanar-stereo",
        action="store_true",
        help="If specified, each specified camera becomes a coplanar stereo pair of cameras."
    )
    parser.add_argument(
        "--coplanar-stereo-dist",
        type=float,
        default=0.06,
        help="Distance between the cameras of a stereo pair (both in camera position and camera lookat)."
    )
    parser.add_argument(
        "--cam-movement-complexity",
        type=int,
        choices=[0, 1, 2, 3],
        default=0,
        help="Specifies degree of complexity of camera movement. 0 = no movement, 1 = slight movement across <=1 dim, "
             "2 = slight or moderate movement across <=2 dim, 3 = slight, moderate or strong movement across <=3 dim. "
             "These levels currently adjust the following camera parameters: "
             " elevation angle, orientation angle (horizontal angle), camera distance to lookat"
    )
    parser.add_argument(
        "--physics-engine",
        type=str,
        choices=["physx", "nimble"],
        default="physx",
        help="Specifies whether to use the default Nvidia PhysX simulator or nimblephysics, "
             "a differentiable DART fork (feature is in beta)."
    )

    # config preparation
    cfg = parser.parse_args()
    cfg.out_path = f"out/{utils.timestamp()}"
    cfg.device = "cpu" if cfg.no_cuda else "cuda"
    cfg.sim_dt = 1.0 / cfg.sim_steps_per_sec
    cfg.cam_dt = cfg.sim_dt * cfg.sim_steps_per_frame
    cfg.sim_fps = cfg.sim_steps_per_sec / cfg.sim_steps_per_frame

    print(f"Generating {cfg.frames} frames at {cfg.sim_fps} fps")
    generate(cfg)
