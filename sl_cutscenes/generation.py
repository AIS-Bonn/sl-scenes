"""
Main logic for running the simulator and generating data
"""
import itertools
import time
from pathlib import Path
from contextlib import ExitStack
import tqdm
import stillleben as sl

from sl_cutscenes.scenarios import SCENARIOS
from sl_cutscenes.output import BOPWriter


def generate(cfg):
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

    if cfg.scenario != "all" and cfg.viewer:  # load scenario and view
        res = init_populate_scene(cfg, scenario_id=cfg.scenario)
        if res["render"]:
            print(f"Scene successfully populated on take #{res['n_errors']}....")
            view_scenario(cfg, renderer, res["scenario"])
        else:
            print("Number of trials exceeded. Scene could not be rendered....")
    else:  # set up scenarios and generate data
        Path(cfg.out_path).mkdir(exist_ok=True, parents=True)
        print(f"will generate {cfg.iterations} episodes per scenario")
        scenario_ids = SCENARIOS.keys() if cfg.scenario == "all" else [cfg.scenario]
        for it in range(cfg.iterations):
            for scenario_id in scenario_ids:
                if scenario_id in ["robopushing"] and cfg.physics_engine != "nimble":
                    assert cfg.scenario == "all", "Robot scenarios require nimblephysics sim"
                    continue
                res = init_populate_scene(cfg, scenario_id=scenario_id)
                if res["render"]:
                    print(f"Scene successfully populated on iteration #{res['n_errors']}....")
                    run_and_render_scenario(cfg, renderer, res["scenario"], it)
                else:
                    print(f"""Iteration {it}, Scene ID {scenario_id} :Number of trials exceeded.
                              Scene could not be rendered....""")
    return


def init_populate_scene(cfg, scenario_id, N_TRIALS=3):
    """
    Initializing a scene, populating it with objects, and making sure there are
    no object collisions
    """
    is_there_collision = True
    n_errors = 0
    scene, scenario = None, None
    while is_there_collision and n_errors < N_TRIALS:
        n_errors += 1
        scene = sl.Scene(cfg.resolution)
        scenario = SCENARIOS[scenario_id](cfg, scene)
        is_there_collision = scenario.is_there_collision()
    else:
        render = True if(n_errors < N_TRIALS) else False

    return {"render": render, "scene": scene, "scenario": scenario, "n_errors": n_errors}


def view_scenario(cfg, renderer, scenario):
    scene = scenario.scene
    view_cam = scenario.cameras[0]
    scene.set_camera_look_at(position=view_cam.get_pos(), look_at=view_cam.get_lookat())
    renderer.render(scene)
    sl.view(scene)


def run_and_render_scenario(cfg, renderer, scenario, it):
    """
    TODO Doc
    """

    # a list of tuples (camera, writers), where each 'writers' itself is a list of tuples (stereo_position, writer)
    writers_per_cam = [(cam, [
        (stereo_pos, BOPWriter(Path(cfg.out_path) / f"{it:06}_{scenario.name}_{cam.get_posed_name(stereo_pos)}"))
        for stereo_pos in cam.stereo_positions
        ]) for cam in scenario.cameras
    ]
    # if cam information is not needed, these are the writers in a plain list
    writers_list = [writer for (_, writer) in list(itertools.chain(*[writers for (cam, writers) in writers_per_cam]))]
    frame_str = "" if cfg.no_gen else f": generating {cfg.frames} frames for {len(writers_list)} individual cameras"
    print(
        f"iteration {it}, scenario '{scenario.name}'{frame_str}"
    )

    with ExitStack() as stack:
        for writer in writers_list:
            stack.enter_context(writer)

        sim_steps, written_frames = 0, 0
        pbar = tqdm.tqdm(total=cfg.frames)

        if cfg.serialize_scene:
            print("Serializing scene...")
            for writer in writers_list:
                writer.serialize_scene(scenario.scene)

        while written_frames < cfg.frames:
            # after sim's prep period, save visualizations every SIM_STEPS_PER_FRAME sim steps
            if sim_steps % cfg.sim_steps_per_frame == 0 and scenario.can_render():
                for cam, cam_writers in writers_per_cam:  # for every cam, there might exist multiple writers (stereo)
                    for cam_stereo_pos, writer in cam_writers:  # set scene camera and render for each posed writer
                        scenario.set_camera_look_at(pos=cam.get_pos(cam_stereo_pos),
                                                    lookat=cam.get_lookat(cam_stereo_pos))
                        result = renderer.render(scenario.scene)
                        if not cfg.no_gen:
                            writer.write_frame(scenario, result)
                    cam.step()  # advance camera for next step if it's a moving one
                written_frames += 1
                pbar.update(1)
                pbar.set_postfix(sim_steps=sim_steps)

            # sim step
            scenario.simulate()
            sim_steps += 1
            # time.sleep(10)
        pbar.close()

        if cfg.assemble_rgb and not cfg.no_gen:
            for writer in writers_list:
                writer.assemble_rgb_video(in_fps=cfg.sim_fps, out_fps=cfg.sim_fps)

        if cfg.physics_engine == "nimble" and cfg.nimble_debug:
            import nimblephysics as nimble
            gui = nimble.NimbleGUI(scenario.nimble_world)
            gui.serve(8080)
            gui.loopStates(scenario.nimble_states)
            vis_secs = 60
            print(f"serving nimblephysics visualization for {vis_secs}s at port 8080")
            time.sleep(vis_secs)
