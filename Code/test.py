# import tensorflow as tf
# from intel_extension_for_tensorflow.python import ipex

# print("TensorFlow version:", tf.__version__)
# print("Intel Extension for TensorFlow version:", ipex.__version__)
# from datetime import datetime

# import pandas as pd
# from datetime import datetime

# # 定义转换函数
# def parse_custom_date(date_str):
#     # 去掉括号和时区信息
#     clean_str = date_str.split(" GMT")[0]
#     # 转换为标准格式
#     return datetime.strptime(clean_str, "%a %b %d %Y %H:%M:%S")

# # 读取 CSV 文件
# csv_file = "data.csv"  # 替换为你的文件名
# df = pd.read_csv(csv_file)

# # 假设时间列的列名为 'saledate'
# # 对 'saledate' 列进行格式化转换
# df['saledate'] = df['saledate'].apply(parse_custom_date)
# df['saledate'] = df['saledate'].dt.strftime("%Y-%m-%d %H:%M:%S")  # 转换为标准格式

# # 保存处理后的数据到新 CSV 文件
# output_file = "processed_file.csv"
# df.to_csv(output_file, index=False)

# print(f"处理完成，保存到 {output_file}")



