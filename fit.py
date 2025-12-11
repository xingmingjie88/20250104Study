import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# 1⃣ 读取数据
file_path = "GRADM.xlsx"
df = pd.read_excel(file_path, sheet_name=0, engine="openpyxl")

# 假设表格里有两列：SDC 和 SMC
X = df['SDC'].values.reshape(-1, 1)
y = df['SMC'].values

# 2⃣ 构造三次多项式项
X_poly = np.hstack([np.ones_like(X), X, X**2, X**3])

# 3⃣ 拟合线性回归
model = LinearRegression(fit_intercept=False)
model.fit(X_poly, y)

# 4⃣ 输出系数
a, b, c, d = model.coef_
print(f"a = {a}")
print(f"b = {b}")
print(f"c = {c}")
print(f"d = {d}")
