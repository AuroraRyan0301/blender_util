import torch
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取数据
data = torch.load('/data2/pixel_histograms.pt')

# 2. 定义 bin 分布
bins = [
    (0, 1, 10),      # 0 to 1, 10 bins
    (1, 10, 10),     # 1 to 10, 10 bins
    (10, 20, 5),     # 10 to 20, 5 bins
    (20, float('inf'), 1)  # 20 to inf, 1 bin
]

# 计算 bin 边界
bin_edges = []
for low, high, num_bins in bins:
    bin_edges.extend(np.linspace(low, high, num_bins + 1))  # +1 是因为包含边界点
bin_edges = np.array(bin_edges)

# 3. 计算频率分布
hist_data = data.sum(dim=0)  # 将数据展平为一维数组


# 计算每个 bin 的宽度
bin_widths = np.diff(bin_edges)
bin_widths_clean = bin_widths[~np.isnan(bin_widths)]
bin_widths_clean = bin_widths_clean[bin_widths_clean != 0]  # 去掉 0
result = np.ones(35)  # 创建一个包含 35 个 1 的数组
result[:bin_widths_clean.shape[0]] = bin_widths_clean
result_cumsum = np.cumsum(result)

result_new_cum_sum = result_cumsum ** (1.0/6.0) * 0.5
result_new = np.diff(result_cumsum, prepend=0) 

# 计算每个 bin 的 PDF（频率 / bin 宽度）
pdf = hist_data / result_new

# 4. 绘制结果
plt.figure(figsize=(8, 6))

# 绘制 PDF
plt.plot(result_new_cum_sum, pdf, marker='o', linestyle='-', color='b', label='PDF')

# 设置坐标轴
plt.xlim(0, 20)
plt.xlabel('Value')
plt.ylabel('PDF')
# 使用对数坐标
plt.yscale('log')
plt.title('Normalized Distribution (PDF) of Pixel Histograms')

# 保存图像
plt.savefig('pixel_histogram_pdf.png')
