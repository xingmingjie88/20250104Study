import os
import re
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# 导入 LightGBM
import lightgbm as lgb

# ---------------- 配置区 ----------------
input_path = r"2_SG平滑.xlsx"   # 读取该文件
output_path = os.path.join(os.path.dirname(input_path), "SMC_models_with_gridsearch.xlsx")

bands = ["SDC", "B215", "B119", "B31", "B219", "B214", "B20", "B10"]
smc_col = None           # 若 SMC 列不是 "SMC"，可直接填写列名字符串
cv_folds = 5
random_state = 42
n_jobs = -1              # 并行
# --------------------------------------

def calc_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    rpd = float(np.std(y_true, ddof=1) / rmse) if rmse != 0 else np.nan
    return r2, rmse, rpd

# 读取数据
df = pd.read_excel(input_path)

# 自动识别 SMC 列
if smc_col is None:
    cands = [c for c in df.columns if re.search(r'smc', str(c), re.IGNORECASE)]
    if not cands:
        raise ValueError("未找到 SMC 列，请将 smc_col 设置为实际列名。")
    smc_col = cands[0]

# 检查波段列
missing = [b for b in bands if b not in df.columns]
if missing:
    raise ValueError(f"缺失的波段列：{missing}")

# 准备数据
X = df[bands].apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(df[smc_col], errors="coerce")

mask = (~X.isna().any(axis=1)) & (~y.isna())
X = X.loc[mask].copy()
y = y.loc[mask].copy()

# 交叉验证
outer_cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

# ====== 1) Random Forest ======
rf = RandomForestRegressor(random_state=random_state)
rf_param_grid = {
    "n_estimators": [200, 400, 800],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
}
rf_gs_for_pred = GridSearchCV(
    rf, rf_param_grid, scoring="r2", cv=outer_cv, n_jobs=n_jobs, refit=True, verbose=0
)
# 折外预测（每折内部做网格搜索，预测验证折）
rf_oof_pred = cross_val_predict(rf_gs_for_pred, X, y, cv=outer_cv, n_jobs=n_jobs)
rf_r2, rf_rmse, rf_rpd = calc_metrics(y, rf_oof_pred)

# 为了输出"一个"最优参数，再在全数据上 refit 一次
rf_gs_full = GridSearchCV(
    rf, rf_param_grid, scoring="r2", cv=outer_cv, n_jobs=n_jobs, refit=True, verbose=0
)
rf_gs_full.fit(X, y)
rf_best_params = rf_gs_full.best_params_

# ====== 2) SVR（含标准化管道） ======
svr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", SVR())
])
svr_param_grid = {
    "svr__kernel": ["rbf"],
    "svr__C": [1, 5, 10, 20, 50],
    "svr__epsilon": [0.05, 0.1, 0.2],
    "svr__gamma": ["scale", "auto"]
}
svr_gs_for_pred = GridSearchCV(
    svr_pipe, svr_param_grid, scoring="r2", cv=outer_cv, n_jobs=n_jobs, refit=True, verbose=0
)
svr_oof_pred = cross_val_predict(svr_gs_for_pred, X, y, cv=outer_cv, n_jobs=n_jobs)
svr_r2, svr_rmse, svr_rpd = calc_metrics(y, svr_oof_pred)

svr_gs_full = GridSearchCV(
    svr_pipe, svr_param_grid, scoring="r2", cv=outer_cv, n_jobs=n_jobs, refit=True, verbose=0
)
svr_gs_full.fit(X, y)
svr_best_params = svr_gs_full.best_params_

# ====== 3) LightGBM ======
lgb_reg = lgb.LGBMRegressor(
    random_state=random_state,
    n_estimators=300,
    verbose=-1,  # 不输出训练信息
    force_col_wise=True  # 避免警告
)
lgb_param_grid = {
    "n_estimators": [200, 300, 500],
    "max_depth": [3, 6, 10, -1],  # -1 表示无限制
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "num_leaves": [31, 63, 127]  # LightGBM 特有参数，控制树复杂度
}
lgb_gs_for_pred = GridSearchCV(
    lgb_reg, lgb_param_grid, scoring="r2", cv=outer_cv, n_jobs=n_jobs, refit=True, verbose=0
)
lgb_oof_pred = cross_val_predict(lgb_gs_for_pred, X, y, cv=outer_cv, n_jobs=n_jobs)
lgb_r2, lgb_rmse, lgb_rpd = calc_metrics(y, lgb_oof_pred)

lgb_gs_full = GridSearchCV(
    lgb_reg, lgb_param_grid, scoring="r2", cv=outer_cv, n_jobs=n_jobs, refit=True, verbose=0
)
lgb_gs_full.fit(X, y)
lgb_best_params = lgb_gs_full.best_params_

# ====== 写入 Excel：Sheet1/Sheet2/Sheet3 ======
with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    # Sheet1: RF
    rf_pred_df = pd.DataFrame({
        "SMC_True": y.values,
        "Pred_RF": rf_oof_pred
    })
    rf_pred_df.to_excel(writer, sheet_name="Sheet1", index=False)
    # 指标与最优参数写在后面
    start_row = len(rf_pred_df) + 2
    pd.DataFrame([{"Model": "RF", "R2": rf_r2, "RMSE": rf_rmse, "RPD": rf_rpd}])\
        .to_excel(writer, sheet_name="Sheet1", startrow=start_row, index=False)
    pd.DataFrame([{"BestParams": json.dumps(rf_best_params, ensure_ascii=False)}])\
        .to_excel(writer, sheet_name="Sheet1", startrow=start_row+3, index=False)

    # Sheet2: SVR
    svr_pred_df = pd.DataFrame({
        "SMC_True": y.values,
        "Pred_SVR": svr_oof_pred
    })
    svr_pred_df.to_excel(writer, sheet_name="Sheet2", index=False)
    start_row = len(svr_pred_df) + 2
    pd.DataFrame([{"Model": "SVR", "R2": svr_r2, "RMSE": svr_rmse, "RPD": svr_rpd}])\
        .to_excel(writer, sheet_name="Sheet2", startrow=start_row, index=False)
    pd.DataFrame([{"BestParams": json.dumps(svr_best_params, ensure_ascii=False)}])\
        .to_excel(writer, sheet_name="Sheet2", startrow=start_row+3, index=False)

    # Sheet3: LightGBM
    lgb_pred_df = pd.DataFrame({
        "SMC_True": y.values,
        "Pred_LGB": lgb_oof_pred
    })
    lgb_pred_df.to_excel(writer, sheet_name="Sheet3", index=False)
    start_row = len(lgb_pred_df) + 2
    pd.DataFrame([{"Model": "LightGBM", "R2": lgb_r2, "RMSE": lgb_rmse, "RPD": lgb_rpd}])\
        .to_excel(writer, sheet_name="Sheet3", startrow=start_row, index=False)
    pd.DataFrame([{"BestParams": json.dumps(lgb_best_params, ensure_ascii=False)}])\
        .to_excel(writer, sheet_name="Sheet3", startrow=start_row+3, index=False)

print("✅ 完成：RF/SVR/LightGBM 网格搜索 + 5折CV折外预测 已写入：", output_path)