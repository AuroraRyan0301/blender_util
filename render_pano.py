# Entry point for rendering the RIS dataset (panorama).
#
# By Jet <zxs@ucsd.edu>, 2024.
#
import os
import bpy
import sys
import time
import json
import shutil
import random

sys.path.append(os.path.dirname(__file__))
import render_utils

argv = sys.argv
argv = argv[argv.index("--") + 1:]

SCENE_LIST = argv[0]
OUTDIR = argv[1]

RES_Y = render_utils.getArg(argv, 'resy', int, 512)
NUM_SAMPLES = render_utils.getArg(argv, 'nsamples', int, 32)
NUM_AO_BOUNCES = render_utils.getArg(argv, 'aobounces', int, 3)
BASELINE = render_utils.getArg(argv, 'baseline', float, 0.25)
NO_GPU = render_utils.getArg(argv, 'nogpu', bool)
WORKER_ID = render_utils.getArg(argv, 'wid', int, int(time.time() * 100))
RESOLUTION = (RES_Y * 2, RES_Y)

NUM_SEQ = render_utils.getArg(argv, 'nseq', int, 2)
LEN_SEQ = render_utils.getArg(argv, 'lseq', int, 1)
STEP_SIZE = render_utils.getArg(argv, 'step', int, 0.1)
COLL_THRESH = render_utils.getArg(argv, 'cthresh', float, 0.5)


if __name__ == '__main__':
    blend_files = render_utils.parseSceneList(SCENE_LIST)
    scene_dir = os.path.split(SCENE_LIST)[0]
    os.makedirs(OUTDIR, exist_ok=True)
    log_file = open(os.path.join(OUTDIR, f'log_{WORKER_ID}.txt'), 'w')
    sys.stdout = render_utils.Files([sys.stdout, log_file])

    print('Script arguments:')
    for argname in ['SCENE_LIST', 'OUTDIR', 'RESOLUTION',
                    'NUM_SAMPLES', 'NUM_AO_BOUNCES', 'BASELINE', 'WORKER_ID',
                    'NUM_SEQ', 'LEN_SEQ', 'STEP_SIZE', 'COLL_THRESH']:
        print(f'{argname}:\t{globals()[argname]}')
    sys.stdout.flush()

    for blend_file in blend_files:
        try:
            scene_path = os.path.join(scene_dir, blend_file)
            scene_prefix = os.path.split(blend_file)[1].split('.')[0]
            scene_outdir = os.path.join(OUTDIR, scene_prefix)
            scene_outdir = os.path.abspath(scene_outdir)
            os.makedirs(scene_outdir, exist_ok=True)

            if not (scene_outdir.endswith('\\') or scene_outdir.endswith('/')):
                scene_outdir = (scene_outdir + '/')
            print(f'[Info] Rendering scene: "{scene_prefix}"')

            # Load the blend file
            bpy.ops.wm.read_factory_settings(use_empty=True)
            bpy.ops.wm.open_mainfile(filepath=scene_path)

            # Initialize the renderer
            render_utils.blenderInitPano(
                resolution=RESOLUTION,
                numSamples=NUM_SAMPLES,
                aoBounces=NUM_AO_BOUNCES,
                use_nodes=True,
                no_gpu=NO_GPU
            )

            # We always use the second scene 'Scene.001'
            sceneKey = bpy.data.scenes.keys()[1]
            print(f'[Info | {scene_prefix}] Using Scene "{sceneKey}"')
            scene = bpy.data.scenes[sceneKey]

            # Prepare multi-view camera
            cameras = render_utils.getCameras()
            camera = cameras[0]
            camera.data.type = 'PANO'
            camera.data.cycles.panorama_type = 'EQUIRECTANGULAR'
            render_utils.prepareCameraSet(base_camera=camera, number=1)

            # Scene info
            camera_heuristics = {'loc': camera.location, 'rot': camera.rotation_euler}
            floor = render_utils.getFloor(camera_heuristics=camera_heuristics)
            if floor is None:
                print(f'[Error | {scene_prefix}] Fail to locate the floor, skipping...')
                raise RuntimeError

            # Main Loop
            for i_seq in range(NUM_SEQ):
                seq_dir = os.path.join(scene_outdir, f'{i_seq:02d}')
                seq_meta = {}

                if os.path.exists(seq_dir):
                    if os.path.exists(os.path.join(seq_dir, 'meta.json')):
                        print(f'[Info | {scene_prefix} | {i_seq:02d}] Rendered, skipping...')
                        continue
                    elif render_utils.dirLocked(seq_dir):
                        print(f'[Info | {scene_prefix} | {i_seq:02d}] Work in progress, skipping...')
                        continue
                    else:
                        print(f'[Info | {scene_prefix} | {i_seq:02d}] Deleting incomplete work...')
                        shutil.rmtree(seq_dir, ignore_errors=True)

                os.makedirs(seq_dir, exist_ok=True)
                render_utils.dirLock(seq_dir)

                traj, n_hits = render_utils.findTrajectory(
                    COLL_THRESH, camera_heuristics, floor, n_step=LEN_SEQ, len_step=STEP_SIZE
                )
                sys.stdout.flush()
                seq_meta['is_perfect_traj'] = (n_hits == 0)
                seq_meta['n_hits'] = n_hits

                
                # get all exr files in dir
                folder_path = "/data2/indoorDatasetCalibrated"
                exr_files = [f for f in os.listdir(folder_path) if f.endswith('.exr')]
                if not exr_files:
                    print(f'[Error] No EXR files found in the folder: {folder_path}')

                # random choose
                selected_exr = random.choice(exr_files)
                exr_path = os.path.join(folder_path, selected_exr)

                # record selected exr name to seq_dir
                with open(seq_dir+"/texture_exr.txt", 'w') as file:
                    file.write(f"{selected_exr}\n")
                

                for i_frame, cam_pose in enumerate(traj):
                    frame_dir = os.path.join(seq_dir, f'{i_frame:02d}')
                    os.makedirs(frame_dir, exist_ok=True)

                    camera.location, camera.rotation_euler = cam_pose
                    camera_info_dir = os.path.join(frame_dir, 'camera_info')
                    os.makedirs(camera_info_dir, exist_ok=True)

                    render_utils.placePanoCameraSet(
                        base_camera=camera,
                        baseline=BASELINE,
                        dump_dir=camera_info_dir
                    )

                    render_utils.buildMainNodeTree(frame_dir, save_thumbnails=True, exr_path=exr_path)

                    try:
                        bpy.ops.render.render()
                    except Exception:
                        shutil.rmtree(seq_dir)
                        raise

                render_utils.dirUnlock(seq_dir)
                json.dump(seq_meta, open(os.path.join(seq_dir, 'meta.json'), 'w'))
                sys.stdout.flush()

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(f'[Error | {scene_prefix}] Error:', exc_type, fname, exc_tb.tb_lineno)
            print(repr(e))
            sys.stdout.flush()

            # a = input('Press r to raise\n')
            # if a == 'r':
            #     raise e