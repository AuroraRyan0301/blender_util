
import cv2
import torch
import os
import glob
import shutil
from tqdm import tqdm
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
# 获取所有的 .exr 文件路径
exr_paths = glob.glob('/data2/test/*/rgb/0001.exr')
depth_exr_paths = glob.glob('/data2/test/*/depth/0001.exr')
# import ipdb;ipdb.set_trace()

def process_exr_files(exr_paths,depth_exr_paths ,new_output_dir):
    available_data = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for exr_path in tqdm(exr_paths):
        # 使用 cv2 读取 EXR 图像
        image = cv2.imread(exr_path, cv2.IMREAD_UNCHANGED)
        depth_path = exr_path[:-12] + "depth/0001.exr"
        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        # 确保图像是 RGB 通道
        if image is not None and len(image.shape) == 3 and image.shape[2] == 3:
            # 将数据转为 torch Tensor
            image_tensor = torch.tensor(image, dtype=torch.float32).to(device)
            depth_image_tensor = torch.tensor(depth_image, dtype=torch.float32).to(device)
        if image_tensor.mean() > 0.2 and depth_image_tensor.min() >= 0.8:
                            # 获取文件夹路径
            folder_name = exr_path.split('/')[-3]  # 假设文件夹名称在路径的第3个位置
            folder_path = os.path.dirname(os.path.dirname(exr_path))
            
            # 将文件夹中的所有文件（除了 .exr 文件）复制到新目录
            folder_output_dir = os.path.join(new_output_dir, folder_name)
            # os.makedirs(folder_output_dir, exist_ok=True)
            
            # 获取文件夹内的所有文件，过滤掉 .exr 文件
            shutil.copytree(folder_path , folder_output_dir,dirs_exist_ok=True)



new_output_dir = "/data2/risv1/"
with torch.no_grad():
    process_exr_files(exr_paths, depth_exr_paths,new_output_dir)
    # torch.save(pixel_histograms, '/data2/pixel_histograms.pt')