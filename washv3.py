
import cv2
import torch
import os
import glob
import shutil
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
# 获取所有的 .exr 文件路径
exr_paths = glob.glob('/data/risv1/*/rgb/0001.exr')
depth_exr_paths = glob.glob('/data/risv1/*/depth/0001.exr')
# import ipdb;ipdb.set_trace()

def save_as_exr(exr_path, image):
    x_0_hat_img = cv2.cvtColor(image.astype(np.float32),cv2.COLOR_RGB2BGR)
    cv2.imwrite(exr_path, x_0_hat_img)

def down_sample(tensor):
    
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)

    # 使用 interpolate 进行下采样，将其尺寸缩小为 (1, 3, 256, 512)
    resized_tensor = F.interpolate(tensor, size=(256, 512), mode='bilinear', align_corners=False)

    # 调整回原来的 (256, 512, 3)
    resized_tensor = resized_tensor.squeeze(0).permute(1, 2, 0)
    return resized_tensor

def process_exr_files(exr_paths,depth_exr_paths ,new_output_dir):
    available_data = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for exr_path in tqdm(exr_paths):
        # 使用 cv2 读取 EXR 图像
        image = cv2.imread(exr_path, cv2.IMREAD_UNCHANGED)
        depth_path = exr_path[:-12] + "depth/0001.exr"
        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)


        # 将数据转为 torch Tensor
        image_tensor = torch.tensor(image, dtype=torch.float32).to(device)
        depth_image_tensor = torch.tensor(depth_image, dtype=torch.float32).to(device)

        


        image_tensor_mask = image_tensor.sum(dim=-1) == 0.0  # 对最后一个维度（宽度）求和并检查是否为 0

        # 统计 image_tensor_mask 中为 True 的元素个数
        num_true_elements = image_tensor_mask.sum().item()

        depth_image_tensor = down_sample(depth_image_tensor)
        image_tensor = down_sample(image_tensor)


        depth_image_tensor.clamp(0.8, 80)
        depth_image_tensor = 0.72/depth_image_tensor
        image_max = image_tensor.max()
        if image_max > 50.0:
            image_tensor = image_tensor * 50.0 / image_max 

        image_tensor = (image_tensor/10) ** (1/6.0) * 0.7

        

        if num_true_elements < 1000:
                            # 获取文件夹路径
            folder_name = exr_path.split('/')[-3]  # 假设文件夹名称在路径的第3个位置
            folder_path = os.path.dirname(os.path.dirname(exr_path))
            
            # 将文件夹中的所有文件（除了 .exr 文件）复制到新目录
            folder_output_dir = os.path.join(new_output_dir, folder_name)
            # os.makedirs(folder_output_dir, exist_ok=True)
            # 获取文件夹内的所有文件，过滤掉 .exr 文件
            shutil.copytree(folder_path , folder_output_dir,dirs_exist_ok=True)

            save_as_exr(os.path.join(folder_output_dir, "rgb", "0001.exr"), image_tensor.detach().cpu().numpy())
            save_as_exr(os.path.join(folder_output_dir, "depth", "0001.exr"), depth_image_tensor.detach().cpu().numpy())
        
        del image_tensor, image_tensor_mask, num_true_elements,  depth_image_tensor, depth_image, image


new_output_dir = "/data/ris_processed_v3_256/"
with torch.no_grad():
    process_exr_files(exr_paths, depth_exr_paths,new_output_dir)
    # torch.save(pixel_histograms, '/data2/pixel_histograms.pt')