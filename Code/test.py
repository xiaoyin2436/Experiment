# import tensorflow as tf
# from intel_extension_for_tensorflow.python import ipex

# print("TensorFlow version:", tf.__version__)
# print("Intel Extension for TensorFlow version:", ipex.__version__)

import pandas as pd
import re

# 定义一个函数来转换原始的时间字符串为标准的 ISO 格式
def clean_timezone(date_str):
    # 将时区部分处理为标准格式: 例如 "GMT-0800" -> "-08:00"
    date_str = re.sub(r' GMT([+-]\d{4})', r' \1', date_str)  # 将 GMT+0800 转换为 +08:00
    return date_str

# 读取数据
data = pd.read_csv('data.csv')

# 清理 saledate 列中的时区部分
data['saledate'] = data['saledate'].apply(clean_timezone)

# 将 saledate 列转换为 datetime 格式，使用 "%a %b %d %Y %H:%M:%S %z"
data['saledate'] = pd.to_datetime(data['saledate'], format='%a %b %d %Y %H:%M:%S %z', errors='coerce')

# 检查转换后的结果
print(data['saledate'].head())

# 计算 saleday（与最小日期的天数差）
data['saleday'] = (data['saledate'] - data['saledate'].min()).dt.days

# 删除原始的 saledate 列
data = data.drop(columns=['saledate'])

# 输出结果
print(data.head())

