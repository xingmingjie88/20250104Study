# -*- coding: utf-8 -*-
"""
读取 1.xlsx，计算三种模型（TOPP / Modified Topp / Herkelrath）对实测 SMC 的
R2、RMSE、RPD，并把结果写入 2.xlsx（工作表名：Metrics）。

要求：1.xlsx 至少包含列：SMC, TOPP, Modified Topp, Herkelrath
"""

import os
import io
import numpy as np
import pandas as pd

# ---------- 优先尝试GPU（CuPy），不可用则回退CPU（NumPy） ----------
use_gpu = False
try:
    import cupy as cp
    _ = cp.arange(3)
    xp = cp
    use_gpu = True
except Exception:
    xp = np
    use_gpu = False

def compute_metrics(y_true, y_pred, backend=np):
    """返回 (R2, RMSE, RPD)"""
    y_true = backend.asarray(y_true, dtype=backend.float64)
    y_pred = backend.asarray(y_pred, dtype=backend.float64)

    ss_res = backend.sum((y_true - y_pred) ** 2)
    y_mean = backend.mean(y_true)
    ss_tot = backend.sum((y_true - y_mean) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else backend.nan

    rmse = backend.sqrt(backend.mean((y_true - y_pred) ** 2))

    # RPD = SD / RMSE，这里 SD 用样本标准差（ddof=1）
    if backend is np:
        sd = backend.std(np.asarray(y_true), ddof=1)
    else:
        sd = backend.std(y_true, ddof=1)

    rpd = (sd / rmse) if rmse != 0 else backend.nan

    # 转为 Python 标量
    to_scalar = (lambda v: float(backend.asnumpy(v))
                 if hasattr(backend, "asnumpy") else float(v))
    return to_scalar(r2), to_scalar(rmse), to_scalar(rpd)

# ---------- 读取 GRADM.xlsx ----------
excel_path = "GRADM.xlsx"
if not os.path.exists(excel_path):
    # 若不存在，示例数据保障可跑；你有真实数据时，请删除此段
    sample = """point,SMC,SDC,容重,孔隙度,TOPP,Modified Topp,Herkelrath
1,0.2847433,14.923674,1.371530584,0.482441289,0.274569545,0.294154878,0.326418118
2,0.289314746,15.32625995,1.475484262,0.443213486,0.280815172,0.299687579,0.329641903
3,0.300708284,16.130674,1.620117315,0.388634976,0.292954299,0.310479333,0.335958992
"""
    df = pd.read_csv(io.StringIO(sample))
    df.to_excel(excel_path, index=False)
else:
    df = pd.read_excel(excel_path)

df.columns = [str(c).strip() for c in df.columns]
required = ["SMC", "TOPP", "Modified Topp", "Herkelrath"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"1.xlsx 缺少必要列：{missing}")

y = df["SMC"].astype(float).values

# ---------- 计算三种模型的指标 ----------
rows = []
for col in ["TOPP", "Modified Topp", "Herkelrath"]:
    y_pred = df[col].astype(float).values
    r2, rmse, rpd = compute_metrics(y, y_pred, backend=cp if use_gpu else np)
    rows.append({"Model": col, "R2": r2, "RMSE": rmse, "RPD": rpd})

metrics_df = pd.DataFrame(rows)

# ---------- 写入 Metrics.xlsx ----------
out_path = "Metrics.xlsx"
with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    metrics_df.to_excel(writer, sheet_name="Metrics", index=False)
print(f"完成。结果已写入 {out_path} 的工作表 Metrics。")
