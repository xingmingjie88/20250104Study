import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

# 读取原始 Excel 文件
file_path = '2.xlsx'
df = pd.read_excel(file_path, sheet_name='2.1')

# 提取高光谱波段列（B1 到 B224）
band_columns = [col for col in df.columns if col.startswith('B') and col[1:].isdigit()]
band_data = df[band_columns]

# 定义 SG 平滑参数
window_length = 11  # 窗口长度（必须为奇数）
polyorder = 3       # 多项式阶数

# 对每一行的高光谱数据进行 SG 平滑
smoothed_data = []
for index, row in band_data.iterrows():
    spectrum = row.values
    smoothed_spectrum = savgol_filter(spectrum, window_length, polyorder)
    smoothed_data.append(smoothed_spectrum)

# 将平滑后的数据转换为 DataFrame
smoothed_df = pd.DataFrame(smoothed_data, columns=band_columns)

# 将非光谱列（point, SMC, SDC）与平滑后的光谱数据合并
result_df = pd.concat([df[['point', 'SMC', 'SDC']], smoothed_df], axis=1)

# 保存到新的 Excel 文件
output_file = '2_SG平滑1.xlsx'
result_df.to_excel(output_file, index=False)

print(f"SG平滑完成，结果已保存至: {output_file}")