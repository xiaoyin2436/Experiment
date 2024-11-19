import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# 导入 Intel Extension for PyTorch
import intel_extension_for_pytorch as ipex

# 1. 数据加载
data = pd.read_csv('data.csv', encoding='utf-8', encoding_errors='ignore')

# 1.1 处理类别特征，进行独热编码
categorical_cols = ['make', 'model', 'trim', 'body', 'transmission', 'state', 'color', 'interior']
data = pd.get_dummies(data, columns=categorical_cols)

# 1.2 处理日期特征（saledate）
data['saledate'] = pd.to_datetime(data['saledate'])
data['saleday'] = (data['saledate'] - data['saledate'].min()).dt.days
data = data.drop(columns=['saledate'])  # 移除原始的saledate列

# 1.3 选择特征列和目标列
X = data.drop(columns=['sellingprice', 'mmr'])
y = data['sellingprice']

# 1.4 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. 数据分割
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 3. 将数据转换为 PyTorch 的 tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# 4. 检查是否可以使用 Intel XPU (A770)
device = torch.device("cpu")  # 默认使用 CPU
# 使用 Intel XPU 设备（如果你已经正确安装了 IPEX，并且系统支持）
device = ipex.device("xpu")  # Intel XPU 设备

# 5. 定义神经网络模型
class MLPRegressor(nn.Module):
    def __init__(self, input_dim):
        super(MLPRegressor, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = self.output(x)
        return x

# 6. 初始化模型并迁移到 XPU
model = MLPRegressor(input_dim=X_train.shape[1]).to(device)

# 7. 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 使用 Intel® Extension for PyTorch 优化模型
model = ipex.optimize(model)

# 8. 训练模型
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    # 前向传播
    y_pred = model(X_train_tensor.to(device))
    # 计算损失
    loss = criterion(y_pred, y_train_tensor.to(device))
    # 反向传播
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 9. 在测试集上进行预测
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test_tensor.to(device))

# 10. 计算均方误差（MSE）
test_loss = mean_squared_error(y_test_tensor.numpy(), y_pred_test.cpu().numpy())
print(f"Test MSE: {test_loss:.4f}")

