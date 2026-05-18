import numpy as np

# --- BAI 1 [cite: 572, 573] ---
print("--- BAI 1 ---")
# Tao mang 3x3 ngau nhien tu [0, 20]
arr1 = np.random.randint(0, 21, size=(3, 3))
print("Mang:\n", arr1)
# Kiem tra tat ca phan tu khac 0 bang np.all()
print("Tat ca phan tu khac 0?:", np.all(arr1 != 0))

# --- BAI 2 [cite: 573, 574] ---
print("\n--- BAI 2 ---")
arr2 = np.random.randint(0, 21, size=(3, 3))
print("Mang:\n", arr2)
# Kiem tra ton tai phan tu khac 0 bang np.any()
print("Ton tai it nhat mot phan tu khac 0?:", np.any(arr2 != 0))

# --- BAI 3 [cite: 574] ---
print("\n--- BAI 3 ---")
a = np.array([1, 5, 10])
b = np.array([5, 5, 2])
print("So sanh x > y (greater):", np.greater(a, b))
print("So sanh x >= y (greater_equal):", np.greater_equal(a, b))
print("So sanh x < y (less):", np.less(a, b))
print("So sanh x <= y (less_equal):", np.less_equal(a, b))

# --- BAI 4 [cite: 575] ---
print("\n--- BAI 4 ---")
# Tao mang 10 so 1, 10 so 0, 10 so 5
arr4 = np.concatenate([np.ones(10), np.zeros(10), np.ones(10) * 5])
print("Mang ket hop:\n", arr4)

# --- BAI 5 [cite: 576] ---
print("\n--- BAI 5 ---")
# Mang so nguyen chan tu 30 den 70
arr5 = np.arange(30, 71, 2)
print("Mang so chan:\n", arr5)

# --- BAI 6 [cite: 577] ---
print("\n--- BAI 6 ---")
# Ma tran don vi 3x3
arr6 = np.identity(3)
print("Ma tran don vi:\n", arr6)

# --- BAI 7 [cite: 581, 582] ---
print("\n--- BAI 7 ---")
# Tao mang 10 phan tu trong [15, 55], in tru dau va cuoi
arr7 = np.linspace(15, 55, 10)
print("Mang day du:", arr7)
print("Bo phan tu dau va cuoi:", arr7[1:-1])

# --- BAI 8 [cite: 583] ---
print("\n--- BAI 8 ---")
# Tao mang 20 phan tu [0, 20], doi dau so trong khoang [9, 15]
arr8 = np.arange(21)
arr8[(arr8 >= 9) & (arr8 <= 15)] *= -1
print("Mang sau khi doi dau:\n", arr8)

# --- BAI 9 [cite: 584] ---
print("\n--- BAI 9 ---")
# Ma tran 3x4 voi gia tri tu [10, 21]
arr9 = np.random.randint(10, 22, size=(3, 4))
print("Ma tran 3x4:\n", arr9)

# --- BAI 10 [cite: 585] ---
print("\n--- BAI 10 ---")
# Ma tran 10x10, bien bang 1, ben trong bang 0
arr10 = np.ones((10, 10))
arr10[1:-1, 1:-1] = 0
print("Ma tran vien 1:\n", arr10)

# --- BAI 11 [cite: 586] ---
print("\n--- BAI 11 ---")
# Ma tran 5x5 duong cheo chinh 1, 2, 3, 4, 5
arr11 = np.diag([1, 2, 3, 4, 5])
print("Ma tran duong cheo:\n", arr11)

# --- BAI 12 [cite: 587] ---
print("\n--- BAI 12 ---")
# Mang 3x3x3, tinh tong theo dong va cot
arr12 = np.random.randint(1, 10, size=(3, 3, 3))
print("Tong theo cot (axis=0):\n", np.sum(arr12, axis=0))
print("Tong theo dong (axis=1):\n", np.sum(arr12, axis=1))

# --- BAI 13 [cite: 588] ---
print("\n--- BAI 13 ---")
# Inner product 2 vector 10 phan tu
v1 = np.random.rand(10)
v2 = np.random.rand(10)
result13 = np.dot(v1, v2)
print("Ket qua inner product:", result13)

# --- BAI 14 [cite: 589, 590] ---
print("\n--- BAI 14 ---")
# Them vector y (3 phan tu) vao tung dong ma tran A (4x3)
A = np.random.randint(1, 10, size=(4, 3))
y = np.random.randint(1, 10, size=(3,))
print("Ma tran A:\n", A)
print("Vector y:", y)
# Numpy tu dong "broadcasting" khi cong ma tran voi vector
print("Ket qua A + y:\n", A + y)