import numpy as np
import pandas as pd
c= np.array([(1,2,3),(4,5,6),(7,8,9),(10,11,12)], dtype=np.float32)
print(c)
print(c.shape)
print(c.dtype)



import numpy as np
c= np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], dtype=np.float32)
print(c)
print(c.shape)

e = np.zeros((2,2))
f= np.ones((3,3))
print('Zeros:\n',e)
print('Ones:\n',f)

# 3.1. Reshape - Thay đổi kích thước 
e = np.array([[1., 2., 3.], [4., 5., 6.]]) 
f_reshaped = e.reshape((3, 2)) 
print("Sau khi Reshape (3,2):\n", f_reshaped)

# 3.2. Flatten - Chuyển về 1 chiều
g_flatten = e.flatten() 
print("Sau khi Flatten:", g_flatten)

# 3.3. Ghép nối mảng (Stacking) 
v1 = np.array([1., 2., 3.]) 
v2 = np.array([4., 5., 6.]) 
h_stack = np.hstack((v1, v2)) # Nối ngang [cite: 171, 176]
v_stack = np.vstack((v1, v2)) # Nối dọc [cite: 172, 178]
print("H-Stack:", h_stack)
print("V-Stack:\n", v_stack)

# 4.1. Số nguyên ngẫu nhiên 
# Sinh mảng 3x4 có giá trị < 100
rand_int = np.random.randint(100, size=(3, 4)) 
print("So nguyen ngau nhien:\n", rand_int)

# 4.2. Số thực ngẫu nhiên [0, 1] 
rand_float = np.random.rand(3, 4) 
print("So thuc ngau nhien [0,1]:\n", rand_float)

# 4.3. Phân phối chuẩn Gaussian
# loc=trung bình, scale=độ lệch chuẩn, size=số lượng
gaussian = np.random.normal(5, 0.5, 10) 
print("Phan phoi chuan:\n", gaussian)

# 4.4. Chọn ngẫu nhiên từ mảng cho trước 
choices = [2, 4, 6, 8, 10]
rand_choice = np.random.choice(choices, size=(3, 5))
print("Chon ngau nhien:\n", rand_choice)


# 5.1. Sinh dãy số 
a_range = np.arange(1, 20, 2) # Từ 1 đến 19, bước nhảy 2 [cite: 303, 306]
a_linspace = np.linspace(1., 5., num=10) # Chia đều khoảng [1, 5] thành 10 phần [cite: 313]

# 5.2. Index và Slice (Cắt mảng) 
arr = np.array([(1, 2, 3), (4, 5, 6)])
print("Dong 1:", arr[0]) 
print("Cot 2:", arr[:, 1]) 
print("2 gia tri dau cua dong 2:", arr[1, :2]) # [cite: 333, 340]

# 5.3. Thay đổi giá trị ma trận
A = np.matrix(np.ones((5, 5))) 
np.asarray(A)[2] = 2 # Đổi toàn bộ dòng thứ 3 thành số 2 [cite: 275]
print("ma tran sau khi thay doi lan 3 3:\n", A)


# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------

#Pandas
# 2.1. Tao Series tu list [cite: 354, 355]
s1 = pd.Series([1.0, 2.0, 3.0])
print("Series tu list:\n", s1)
print("Gia tri dau tien:", s1[0]) 

# 2.2. Tao Series voi Index tuy chinh [cite: 358]
s2 = pd.Series([1.0, 2.0, 3.0], index=['a', 'b', 'c'])
print("\nSeries co index:\n", s2)
print("Gia tri tai index 'a':", s2['a']) 

# 2.3. Tao Series tu Dictionary [cite: 360, 361]
calories = {"day1": 420, "day2": 380, "day3": 390}
my_var = pd.Series(calories)
print("\nSeries tu Dictionary:\n", my_var) 



# 3.1. Chuyen doi Numpy <-> Pandas [cite: 386]
h = np.array([[1, 2], [3, 4]])
df_h = pd.DataFrame(h) # Numpy sang Pandas [cite: 391]
print("DataFrame tu Numpy:\n", df_h) 

numpy_h = np.array(df_h) # Pandas sang Numpy [cite: 394]
print("\nNumpy array tu DataFrame:\n", numpy_h)

# 3.2. Tao DataFrame co ten cot [cite: 403]
dic = {'Name': ['John', 'Smith'], 'Age': [30, 40]} 
df_dic = pd.DataFrame(dic) [cite: 405]
print("\nDataFrame tu Dictionary:\n", df_dic) 

# 3.3. Tao DataFrame tu so ngau nhien [cite: 411, 412]
r = np.random.randn(20, 4)
df = pd.DataFrame(r, columns=['so1', 'so2', 'so3', 'so4']) 
print("\n5 dong dau tien:\n", df.head()) # In 5 dong dau [cite: 414]



# 4.1. Loc va Cat du lieu (Slicing) [cite: 429]
print("Chon 1 cot 'so1':\n", df['so1'].head()) 
print("Chon dong 0 den 2:\n", df[0:3]) 
print("Chon nhieu cot:\n", df[['so1', 'so3']].head()) 
print("Dung iloc (3 dong dau, cot index 2):\n", df.iloc[:3, 2]) 

# 4.2. Xoa cot [cite: 444]
df_dropped = df.drop(columns=['so1', 'so4']) [cite: 445]

# 4.3. Noi DataFrame (Concatenate) [cite: 447, 448]
df1 = pd.DataFrame({'Name': ['John', 'Smith', 'Paul'], 'Age': [25, 30, 50]})
df2 = pd.DataFrame({'Name': ['Adam', 'Smith'], 'Age': [26, 11]}) 
df_concat = pd.concat([df1, df2]) # Noi 2 bang [cite: 457]
print("\nSau khi noi:\n", df_concat) [cite: 458]

# 4.4. Loai bo trung lap va Sap xep
print("\nXoa trung ten:\n", df_concat.drop_duplicates('Name')) 
print("\nSap xep theo tuoi:\n", df_concat.sort_values('Age')) 