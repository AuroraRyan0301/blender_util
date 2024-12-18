import cv2
import torch
import os
import glob
from tqdm import tqdm
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
# 获取所有的 .exr 文件路径
exr_paths = glob.glob('/data2/test/*/rgb/0001.exr')
# import ipdb;ipdb.set_trace()

def get_pixel_value_histogram(image_tensor):
    # 展平图像 tensor，shape: (H*W, 3)
    pixels = image_tensor.view(-1, 3)

    # 创建一个空的统计数组，大小为 (35,) 用于存储所有 RGB 通道的统计结果
    histogram = torch.zeros(35, dtype=torch.int32)

    # 定义区间边界
    bins = [
        (0, 1, 10),      # 0 to 1, 10 bins
        (1, 10, 10),     # 1 to 10, 10 bins
        (10, 20, 5),     # 10 to 20, 5 bins
        (20, float('inf'), 1)  # 20 to inf, 1 bin
    ]

    # 对于每个 RGB 通道分别进行统计
    for c in range(3):  # 3 RGB channels
        channel_pixels = pixels[:, c]  # 获取当前通道的所有像素值

        for low, high, num_bins in bins:
            # 将像素值映射到区间内
            bin_edges = torch.linspace(low, high, num_bins + 1)
            
            # 手动实现 digitize
            bin_indices = torch.sum(channel_pixels[:, None] >= bin_edges[:-1], dim=1)  # Find the bin index
            bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)  # 保证索引在合法范围内
            
            # 对每个 bin 进行计数
            bin_start_idx = sum([b[2] for b in bins[:bins.index((low, high, num_bins))]])
            histogram[bin_start_idx + bin_indices] += 1

    return histogram
# 定义一个函数来统计所有 EXR 文件的像素值
def process_exr_files(exr_paths):
    all_histograms = []

    for exr_path in tqdm(exr_paths):
        # 使用 cv2 读取 EXR 图像
        image = cv2.imread(exr_path, cv2.IMREAD_UNCHANGED)

        # 确保图像是 RGB 通道
        if image is not None and len(image.shape) == 3 and image.shape[2] == 3:
            # 将数据转为 torch Tensor
            image_tensor = torch.tensor(image, dtype=torch.float32)

            # 不计算梯度
            with torch.no_grad():
                histogram = get_pixel_value_histogram(image_tensor)
                all_histograms.append(histogram)
            del image_tensor, image

    return torch.stack(all_histograms) if all_histograms else None
with torch.no_grad():
    pixel_histograms = process_exr_files(exr_paths[:1000])
    torch.save(pixel_histograms, '/data2/pixel_histograms.pt')

# 打印结果
if pixel_histograms is not None:
    print("Pixel histograms shape:", pixel_histograms.shape)
    # print("Example histogram for the first image:", pixel_histograms[0])
else:
    print("No .exr files found or processed.")