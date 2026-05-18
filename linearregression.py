import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

# =========================
# 1. Load Dataset
# =========================

# Đổi đường dẫn nếu cần
file_path = 'crabs.txt'

# Load dữ liệu
# File crabs.txt có dữ liệu cách nhau bằng khoảng trắng
# Các cột:
# presz, postsz, inc, year, lf

df_crabs = pd.read_csv(file_path, sep=r'\s+')

print("Successfully loaded crabs.txt")

print("\nFirst 5 rows:")
print(df_crabs.head())

print("\nDataset Information:")
print(df_crabs.info())

print("\nDescriptive Statistics:")
print(df_crabs.describe())

# =========================
# 2. Data Preparation
# =========================

# Dùng presz để dự đoán postsz
feature_col = 'presz'
target_col = 'postsz'

# Xóa dữ liệu null
# year có nhiều NA nên không dùng

df_cleaned = df_crabs[[feature_col, target_col]].dropna()

X_np = df_cleaned[feature_col].values.astype(np.float32).reshape(-1, 1)
y_np = df_cleaned[target_col].values.astype(np.float32).reshape(-1, 1)

# Chuẩn hóa dữ liệu
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X_np)

X = torch.tensor(X_normalized, dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.float32)

print(f"\nShape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

# =========================
# 3. Visualize Data
# =========================

plt.figure(figsize=(8, 6))
plt.scatter(X.numpy(), y.numpy(), s=10, alpha=0.7)
plt.title('Crabs Data: postsz vs presz')
plt.xlabel('Normalized presz')
plt.ylabel('postsz')
plt.grid(True)
plt.show()

# =========================
# 4. Train/Test Split
# =========================
plt.show()