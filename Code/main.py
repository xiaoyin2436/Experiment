import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import intel_extension_for_pytorch as ipex

# 1. 读取数据
data = pd.read_csv("data.csv")

# 2. 数据预处理
# 转换日期时间列
data['saledate'] = pd.to_datetime(data['saledate'])
data['sale_year'] = data['saledate'].dt.year
data['sale_month'] = data['saledate'].dt.month
data = data.drop(columns=['saledate'])

# 类别型变量编码
categorical_columns = ['make', 'model', 'trim', 'body', 'transmission', 'state', 'color', 'interior']
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    # 保存对应编码，方便后续逆向解码
    label_encoders[col] = le

# 数值型变量归一化
numerical_columns = ['year', 'odometer', 'mmr']
scaler = MinMaxScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# 目标变量
target = 'sellingprice'
features = [col for col in data.columns if col != target]

# 3. 数据集划分
X = data[features].values
y = data[target].values
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为 PyTorch 数据集
class PriceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = PriceDataset(X_train, y_train)
val_dataset = PriceDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 4. 定义神经网络
class PricePredictor(nn.Module):
    def __init__(self, input_size):
        super(PricePredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.network(x)

# 5. 定义损失函数和优化器
model = PricePredictor(input_size=X.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# model.train()

# 使用 IPEX 优化模型
device = torch.device("xpu")  # Intel Arc GPU
model = model.to(device)
criterion = criterion.to(device)

# 优化模型和优化器（全精度模式）
model, optimizer = ipex.optimize(model, optimizer=optimizer)

# 6. 训练模型
epochs = 50
for epoch in range(epochs):
    model.train()
    train_loss = 0.0

    for X_batch, y_batch in train_loader:
        # 将数据迁移到 XPU
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        # 全精度模式下的预测和损失计算
        predictions = model(X_batch).squeeze()
        loss = criterion(predictions, y_batch)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)

    # 验证模型
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # 全精度模式下的验证
            predictions = model(X_batch).squeeze()
            loss = criterion(predictions, y_batch)
            val_loss += loss.item()
    val_loss /= len(val_loader)


    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# 7. 保存模型
torch.save(model.state_dict(), "price_predictor.pth")
