import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error

# 1. 读取数据
data = pd.read_csv("data.csv")

# 2. 数据预处理
# 提取年份、月份和日期作为特征
data['saledate'] = pd.to_datetime(data['saledate'], errors='coerce')
data['sale_year'] = data['saledate'].dt.year
data['sale_month'] = data['saledate'].dt.month
data['sale_dayofweek'] = data['saledate'].dt.dayofweek

# 选择有用的特征
features = ['year', 'make', 'model', 'trim', 'body', 'transmission', 'state', 
            'odometer', 'color', 'interior', 'mmr', 'sale_year', 'sale_month', 'sale_dayofweek']
target = 'sellingprice'

# 去除缺失值
data = data.dropna(subset=features + [target])

# 编码分类特征
categorical_features = ['make', 'model', 'trim', 'body', 'transmission', 'state', 'color', 'interior']
numerical_features = ['year', 'odometer', 'mmr', 'sale_year', 'sale_month', 'sale_dayofweek']

# One-hot encode categorical features
data = pd.get_dummies(data, columns=categorical_features, drop_first=True)

# 归一化数值特征
scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# 分离特征和目标
X = data[features]
y = data[target]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 构建神经网络模型
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))  # 输出层，不加激活函数，因为这是回归问题

# 编译模型
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# 训练模型
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# 4. 评估模型
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error: {rmse}")
