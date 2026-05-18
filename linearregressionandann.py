import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# =========================
# 1. Load Dataset
# =========================
file_path = 'crabs.txt'
df_crabs = pd.read_csv(file_path, sep=r'\s+')
print("Successfully loaded crabs.txt")

# =========================
# 2. Data Preparation
# =========================
feature_col = 'presz'
target_col = 'postsz'

df_cleaned = df_crabs[[feature_col, target_col]].dropna()

X_np = df_cleaned[feature_col].values.astype(np.float32).reshape(-1, 1)
y_np = df_cleaned[target_col].values.astype(np.float32).reshape(-1, 1)

# Normalize du lieu X
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X_np)

X = torch.tensor(X_normalized, dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.float32)

# =========================
# 3. Split Dataset
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 4. Dinh nghia & Huan luyen Linear Regression
# =========================
model_lr = nn.Linear(1, 1)

criterion = nn.MSELoss()
optimizer_lr = torch.optim.SGD(model_lr.parameters(), lr=0.1)

# Train Linear Regression
epochs = 500
for epoch in range(epochs):
    model_lr.train()
    outputs = model_lr(X_train)
    loss = criterion(outputs, y_train)
    
    optimizer_lr.zero_grad()
    loss.backward()
    optimizer_lr.step()

print("Da train xong mo hinh Linear Regression!")

# =========================
# 5. Dinh nghia & Huan luyen Mo hinh ANN
# =========================
class SimpleANN(nn.Module):
    def __init__(self):
        super(SimpleANN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
    def forward(self, x):
        return self.network(x)

model_ann = SimpleANN()
optimizer_ann = torch.optim.Adam(model_ann.parameters(), lr=0.01)

# Train ANN
for epoch in range(epochs):
    model_ann.train()
    outputs = model_ann(X_train)
    loss = criterion(outputs, y_train)
    
    optimizer_ann.zero_grad()
    loss.backward()
    optimizer_ann.step()

print("Da train xong mo hinh ANN!")

# =========================
# 6. Ve so do so sanh ket qua
# =========================
X_test_np = scaler.inverse_transform(X_test.numpy())
y_test_np = y_test.numpy()

model_lr.eval()
model_ann.eval()
with torch.no_grad():
    y_pred_lr_np = model_lr(X_test).numpy()
    y_pred_ann_np = model_ann(X_test).numpy()

sort_idx = np.argsort(X_test_np.flatten())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Do thi 1: Linear Regression
ax1.scatter(X_test_np, y_test_np, color='blue', alpha=0.6, label='Thuc te (Test)')
ax1.plot(X_test_np[sort_idx], y_pred_lr_np[sort_idx], color='red', linewidth=2, label='Du doan (Linear)')
ax1.set_title('Mo hinh Linear Regression tren tap Test')
ax1.set_xlabel('Kich thuoc truoc khi lot xac (presz)')
ax1.set_ylabel('Kich thuoc sau khi lot xac (postsz)')
ax1.legend()
ax1.grid(True)

# Do thi 2: ANN
ax2.scatter(X_test_np, y_test_np, color='blue', alpha=0.6, label='Thuc te (Test)')
ax2.plot(X_test_np[sort_idx], y_pred_ann_np[sort_idx], color='green', linewidth=2, label='Du doan (ANN)')
ax2.set_title('Mo hinh ANN tren tap Test')
ax2.set_xlabel('Kich thuoc truoc khi lot xac (presz)')
ax2.set_ylabel('Kich thuoc sau khi lot xac (postsz)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()