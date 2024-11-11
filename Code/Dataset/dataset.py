# 清理数据
import pandas as pd

# 读取CSV文件
data = pd.read_csv('cleaned_car_prices.csv', encoding='gbk')  # 根据需要修改编码方式
print(data.shape)
