import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# 1⃣ 读取 Excel
file_path = "GRADM.xlsx"
df = pd.read_excel(file_path, sheet_name=0, engine="openpyxl")

# 2⃣ 提取变量
X = np.sqrt(df['SDC'].values).reshape(-1, 1)  # 自变量 √SDC
y = df['SMC'].values                          # 因变量 SMC

# 3⃣ 建立线性回归模型
model = LinearRegression()
model.fit(X, y)

# 4⃣ 提取系数
a = model.intercept_
b = model.coef_[0]

print(f"a = {a}")
print(f"b = {b}")
