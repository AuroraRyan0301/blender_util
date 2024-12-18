# Expand openexr to individual files. Will generate new folders
# under each scene dir. Currently includes `denoised`, `noisy`,
# `depth`, `albedo`, and `normal`.
#
# Usage:
#
#     pip install openEXR
#     python generate_final_outputs.py DATA_DIR OUT_DIR
#
# DATA_DIR should contain several rendered result folders.
# e.g. python generate_final_outputs.py ./phase1_results_0_100 ./phase1_results_0_100_out
#
# TODO:
#     1. Better handling of HDR image data (tone mapping)
#     2. Painless for-loop for more (future) channels
#
# By Jet
import os
import sys
import gzip
import numpy as np
import OpenEXR, Imath
import shutil
from multiprocessing import Pool
import matplotlib.pyplot as plt

INDIR = sys.argv[1]
OUTDIR = sys.argv[2]
SKIPCOMPLETE = True
DEPTH_MAX = 100
DEPTH_MIN = 0.2

def _read_multichannel(data, base_name, channel_suffixes):
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    data_header = data.header()
    data_window = data_header['dataWindow']
    size = (data_window.max.x - data_window.min.x + 1,
                data_window.max.y - data_window.min.y + 1)
    channels = []
    for ch in channel_suffixes:
        buf = data.channel(f'{base_name}.{ch}', pt)
        arr = np.frombuffer(buf, dtype=np.float32)
        arr = arr.reshape(size[1], size[0], 1)
        channels.append(arr)
    img = np.dstack(channels)
    return img


def read_img(data, base_name):
    return _read_multichannel(data, base_name, ('R', 'G', 'B', 'A'))


def read_normal(data, base_name):
    return _read_multichannel(data, base_name, ('X', 'Y', 'Z'))


def read_depth(data, base_name):
    ret = _read_multichannel(data, base_name, ('V'))
    ret = ret.reshape(ret.shape[:-1])
    return ret


# def get_outputs_dir_from_path(outdir_base, path):
#     def get_class_from_path(path: str):
#         return os.path.basename(os.path.abspath(os.path.join(path, '..', '..')))

#     def get_obj_id_from_path(path: str):
#         scene_name = os.path.basename(os.path.abspath(os.path.join(path, '..', '..', '..')))
#         obj_name = os.path.basename(os.path.abspath(os.path.join(path, '..')))
#         return f'{scene_name}_{obj_name}'

#     def get_view_id_from_path(path: str):
#         return os.path.basename(os.path.abspath(os.path.join(path, '.')))

#     cls = get_class_from_path(path)
#     obj_id = get_obj_id_from_path(path)
#     view_id = get_view_id_from_path(path)

#     return os.path.join(outdir_base, f'./{cls}/{obj_id}/{view_id}/')

def get_outputs_dir_from_path(outdir_base, path):
    def get_scene_from_path(path: str):
        return os.path.basename(os.path.abspath(os.path.join(path, '..', '..')))

    def get_view_id_from_path(path: str):
        return os.path.basename(os.path.abspath(os.path.join(path, '.')))

    scene = get_scene_from_path(path)
    view_id = get_view_id_from_path(path)

    return os.path.join(outdir_base, f'./{scene}/{view_id}/')


def get_cam_id(filename):
    cam_id = filename.split('.')[0].split('_')[-1]
    return cam_id


def check_output_complete(d, nviews=5):
#    subdirs = ['albedo', 'cams', 'denoised',
#                'depth', 'masks', 'noisy', 'normal']
#    subdirs = ['cams', 'denoised',
#                'depth', 'masks', 'normal', 'coded_light']
    subdirs = ['cams', 'rgb', 'depth', 'normal']
    for subd in subdirs:
        fullpath = os.path.join(d, subd)
        if not os.path.exists(fullpath):
            return False
        files = os.listdir(fullpath)
        if len(files) != nviews:
            return False

    return True


# Code from GitHub @arseniy-panfilov
# However this is not good enough
def img_to_srgb(img):
    a = 0.055
    return np.where(
        img <= 0.0031308,
        img * 12.92,
        (1 + a) * pow(img, 1 / 2.4) - a
    )


def img_normalize(img):
    ret = img.copy()
    ret -= ret.min()
    ret /= ret.max()
    if ret.shape[-1] == 4:
        ret[..., -1] = img[..., -1]
    return ret


def img_gamma_tone_mapping(img, gamma, A=1):
    img = img.copy()
    img[..., :3] = A * (img[..., :3] ** gamma)
    return img


def img_ACES_tone_mapping(img):
    A = 2.51
    B = 0.03
    C = 2.43
    D = 0.59
    E = 0.14
    def _mapping(color):
        return (color * (A * color + B)) / (color * (C * color + D) + E)
    img = img.copy()
    img = img_gamma_tone_mapping(img, 0.6)
    img[..., :3] = _mapping(img[..., :3])
    img[img > 1] = 1
    return img


if __name__ == '__main__':
    # filepath = "/home/xulab/Nautilus/rendered/0001_1.exr"
    # data = OpenEXR.InputFile(filepath)
    # denoised = read_img(data, 'denoised')
    # # noisy = read_img(data, 'noisy')
    # # albedo = read_img(data, 'albedo')
    # normal = read_normal(data, 'normal')
    # depth = read_depth(data, 'depth')
    # plt.imsave("abc.png", denoised)
    # exit()
    os.makedirs(OUTDIR, exist_ok=True)
    dir_pairs = []
    for root, dirs, files in os.walk(INDIR):
        # if 'rendered' in dirs and 'camera_info' in dirs and 'masks' in dirs and 'thumbnails' in dirs:
        if 'rendered' in dirs and 'camera_info' in dirs and 'thumbnails' in dirs:
            outputs_dir = get_outputs_dir_from_path(OUTDIR, root)
            if SKIPCOMPLETE and os.path.exists(outputs_dir):
                if check_output_complete(outputs_dir):
                    print(f'[Info] "{outputs_dir}" is complete, skipping...')
                else:
                    print(f'[Warning] "{outputs_dir}" is present but incomplete, re-generating...')
                    shutil.rmtree(outputs_dir)
                    dir_pairs.append((root, outputs_dir))
            else:
                dir_pairs.append((root, outputs_dir))

    def process_view_dir(dir_pair):
        scene_dir, outputs_dir = dir_pair
        print(f'[Info] Processing "{scene_dir}"...')
        os.makedirs(outputs_dir)

        rgb_dir = os.path.join(outputs_dir, 'rgb')
        # noisy_dir = os.path.join(outputs_dir, 'noisy')
        # albedo_dir = os.path.join(outputs_dir, 'albedo')
        depth_dir = os.path.join(outputs_dir, 'depth')
        normal_dir = os.path.join(outputs_dir, 'normal')
        cams_dir = os.path.join(outputs_dir, 'cams')
        os.makedirs(rgb_dir, exist_ok=True)
        # os.makedirs(noisy_dir, exist_ok=True)
        # os.makedirs(albedo_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        os.makedirs(normal_dir, exist_ok=True)
        os.makedirs(cams_dir, exist_ok=True)

        # # Process Coded-Light
        # codedlight_src = os.path.join(scene_dir, 'coded_light')
        # if os.path.exists(codedlight_src):
        #     codedlight_dst = os.path.join(outputs_dir, 'coded_light')
        #     os.makedirs(codedlight_dst, exist_ok=True)
        #     for img_name in os.listdir(codedlight_src):
        #         cam_id = get_cam_id(img_name)
        #         ext = img_name.split('.')[-1]
        #         img = plt.imread(os.path.join(codedlight_src, img_name))
        #         img = img[..., 0]    # R channel only!
        #         # Maybe we need to dim the lights instead of turn off all while rendering.
        #         # img = img ** 5   
        #         plt.imsave(
        #             os.path.join(codedlight_dst, f'{cam_id}.{ext}'),
        #             img, cmap='gray'
        #         )

        # # Copy masks
        # mask_dir = os.path.join(scene_dir, 'masks/')
        # mask_dir_new = os.path.join(outputs_dir, 'masks/')
        # shutil.copytree(mask_dir, mask_dir_new)
        # for mask in os.listdir(mask_dir_new):
        #     cam_id = get_cam_id(mask)
        #     ext = mask.split('.')[-1]
        #     shutil.move(
        #         os.path.join(mask_dir_new, mask),
        #         os.path.join(mask_dir_new, f'{cam_id}.{ext}')
        #     )

        # 1. Expand EXR 
        # 2. Dump MVS cam txt
        # 3. Rename files
        rendered_dir = os.path.join(scene_dir, 'rendered/')
        camera_dir = os.path.join(scene_dir, 'camera_info/')
        for filename in os.listdir(rendered_dir):
            assert filename.endswith('.exr')
            filepath = os.path.join(rendered_dir, filename)
            cam_id = get_cam_id(filename)
                        
            try:
                data = OpenEXR.InputFile(filepath)
                # denoised = read_img(data, 'denoised')
                rgb = read_img(data, 'rgb')
                # noisy = read_img(data, 'noisy')
                # albedo = read_img(data, 'albedo')
                normal = read_normal(data, 'normal')
                depth = read_depth(data, 'depth')

                # Assume is hdr data
                # TODO: Very bad way for handling HDR data.
                # Should use some curve in future.
                rgb = img_ACES_tone_mapping(rgb)
                # noisy = img_ACES_tone_mapping(noisy)
                # albedo = img_ACES_tone_mapping(noisy)   # Is this necessary?

                depth[depth > DEPTH_MAX] = 0
                depth[depth < DEPTH_MIN] = 0

                rgb_path = os.path.join(rgb_dir, f'{cam_id}.png')
                # noisy_path = os.path.join(noisy_dir, f'{cam_id}.png')
                # albedo_path = os.path.join(albedo_dir, f'{cam_id}.png')
                normal_path = os.path.join(normal_dir, f'{cam_id}.npy')
                depth_path = os.path.join(depth_dir, f'{cam_id}.npy')

                plt.imsave(rgb_path, rgb)
                # plt.imsave(noisy_path, noisy)
                # plt.imsave(albedo_path, albedo)
                np.save(normal_path, normal)
                np.save(depth_path, depth)


                cam_data = np.load(os.path.join(camera_dir, f'Camera_{cam_id}.npz'))
                K = cam_data['K']
                Rt = cam_data['Rt']
            except Exception as e:
                print(f'[Error] Error processing {filepath}:')
                print(repr(e))
                with open('errlog.txt', 'a') as errf:
                    print(f'[Error] Error processing {filepath}:', file=errf)
                    print(repr(e), file=errf)
                return

            # # m to mm
            # Rt[:, -1] = 1000 * Rt[:, -1] 
            # depth *= 1000

            d_high = depth.max()
            depth[depth==0] = d_high
            d_low = depth.min()

            camtxt_path = os.path.join(cams_dir, f"{cam_id}.txt")
            with open(camtxt_path, 'w') as f:
                print('extrinsic', file=f)
                print(' '.join(map(str, Rt[0])), file=f)
                print(' '.join(map(str, Rt[1])), file=f)
                print(' '.join(map(str, Rt[2])), file=f)
                print('0.0 0.0 0.0 1.0', file=f)
                print(file=f)
                print('intrinsic', file=f)
                print(' '.join(map(str, K[0])), file=f)
                print(' '.join(map(str, K[1])), file=f)
                print(' '.join(map(str, K[2])), file=f)
                print(file=f)
                print(f'{d_low} {d_high}', file=f)
    
    with Pool() as pool:
        pool.map(process_view_dir, dir_pairs)