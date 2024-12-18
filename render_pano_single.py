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
import gc
sys.path.append(os.path.dirname(__file__))
import render_utils

argv = sys.argv
argv = argv[argv.index("--") + 1:]

SCENE_LIST = argv[0]
OUTDIR = argv[1]

RES_X = render_utils.getArg(argv, 'resx', int, 1024)
RES_Y = render_utils.getArg(argv, 'resy', int, 512)
NUM_SAMPLES = render_utils.getArg(argv, 'nsamples', int, 32)
NUM_AO_BOUNCES = render_utils.getArg(argv, 'aobounces', int, 3)
BASELINE = render_utils.getArg(argv, 'baseline', float, 0.25)
NO_GPU = render_utils.getArg(argv, 'nogpu', bool)
WORKER_ID = render_utils.getArg(argv, 'wid', int, int(time.time() * 100))
RESOLUTION = (RES_X, RES_Y)

NUM_SEQ = render_utils.getArg(argv, 'nseq', int, 1)
LEN_SEQ = render_utils.getArg(argv, 'lseq', int, 25)
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

    with open('ris-vol/full_v1.0/filter_list.txt', 'r') as f:
        filter_list = f.readlines()
    filter_list = [x.strip() for x in filter_list]

    print(f'[Info] Total Scenes: {len(blend_files)}')
    for blend_file in blend_files:
        try:
            scene_path = os.path.join(scene_dir, blend_file)
            scene_prefix = os.path.split(blend_file)[1].split('.')[0]
            scene_outdir = os.path.join(OUTDIR, scene_prefix)
            scene_outdir = os.path.abspath(scene_outdir)

            if os.path.exists(scene_outdir):
                print(f'[Info | {scene_prefix}] Rendered, skipping...')
                continue
            
            if scene_prefix in filter_list:
                print(f'[Info | {scene_prefix}] in filter list, skipping...')
                continue

            os.makedirs(scene_outdir, exist_ok=True)

            if not (scene_outdir.endswith('\\') or scene_outdir.endswith('/')):
                scene_outdir = (scene_outdir + '/')
            print(f'[Info] Rendering scene: "{scene_prefix}"')

            # Load the blend file
            bpy.ops.wm.read_factory_settings(use_empty=True)
            bpy.ops.wm.open_mainfile(filepath=scene_path)

            # Initialize the renderer
            render_utils.blenderInitMultiview(
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
            camera.name = f'Camera_0'
            camera.data.type = 'PANO'
            camera.data.cycles.panorama_type = 'EQUIRECTANGULAR'
            # render_utils.prepareCameraSet(base_camera=camera, number=1)

            camera_heuristics = {'loc': camera.location, 'rot': camera.rotation_euler}
            floor = render_utils.getFloor_2(camera_heuristics=camera_heuristics)
            if floor is None:
                print(f'[Error | {scene_prefix}] Fail to locate the floor, skipping...')
                raise RuntimeError
            else:
                print(f'[Info | {scene_prefix}] Floor: {floor}')
            trajectory, n_hit = render_utils.findTrajectorySpiral(COLL_THRESH, camera_heuristics, floor)
            cam_pose = trajectory[0]
            camera.location, camera.rotation_euler = cam_pose

            camera_info_dir = os.path.join(scene_outdir, 'camera_info')
            os.makedirs(camera_info_dir, exist_ok=True)
            render_utils.dumpPanoCameraInfo(camera, camera_info_dir)

            bpy.context.scene.camera = bpy.data.objects['Camera_0']
            render_utils.buildMainNodeTree(scene_outdir, save_thumbnails=True)
            bpy.ops.render.render()

            sys.stdout.flush()

            bpy.ops.wm.read_factory_settings(use_empty=True)
            gc.collect()


        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(f'[Error | {scene_prefix}] Error:', exc_type, fname, exc_tb.tb_lineno)
            print(repr(e))
            sys.stdout.flush()

            # a = input('Press r to raise\n')
            # if a == 'r':
            #     raise e