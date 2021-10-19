import os

import stillleben as sl
from PIL import Image
from pathlib import Path
from ycb_dynamic.scenarios.table import setup_table_scene


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

    # set up scenarios and generate data
    scenario_ids = SCENARIOS.keys() if cfg.scenario == "all" else [cfg.scenario]
    for scenario_id in scenario_ids:
        scene = sl.Scene(RESOLUTION)  # (re-)initialize scene
        scene = SCENARIOS[scenario_id](cfg, scene)  # populate scene according to scenario
        run_and_render_scene(cfg, scene)

    return


def run_and_render_scene(cfg, scene):
    '''
    TODO Doc
    :param cfg:
    :param scene:
    :return:
    '''

    # TODO 1: run simulation and write frames from each camera pose (check SynPick code for this)
    renderer = sl.RenderPass()
    result = renderer.render(scene)

    # TODO 2: sl.view() can be called if not using a different graphics device or SL started in CPU mode

    Image.fromarray(result.rgb()[:,:,:3].cpu().numpy()).save(os.path.join(cfg.out_path, 'scene_output.jpg'))
    # TODO 3: Generate other data in BOP format
    # TODO 4: generate videos from generated images to quickly check generated data



# ==============================================================================

RESOLUTION = (1920, 1080)
OUT_PATH = Path("output")
SCENARIOS = { "table": setup_table_scene }


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
