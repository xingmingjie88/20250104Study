import pandas as pd
import numpy as np
import re
import json
import os

from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error

# LightGBM (确保已安装: pip install lightgbm)
from lightgbm import LGBMRegressor

# ================= 配置区域 =================
input_path = r"2_SG平滑.xlsx"
# 改新文件名，避免覆盖
output_path = os.path.join(os.path.dirname(input_path), "SMC_prediction_with_gridsearch_SMCbounded.xlsx")

# 注意：这里不包含 "SDC" —— SDC 只用于算边界，不作为特征
band_list = ["B215", "B119", "B31", "B219", "B214", "B20", "B10"]

smc_col = 'SMC'          # 若 SMC 列名不是 SMC，可直接填具体列名
sdc_col = 'SDC'            # 若 SDC 列名不是 SDC，可直接填具体列名
random_state = 42
cv_folds = 5
n_jobs = -1                 # 并行线程
# ==========================================

# --------- 读取数据 ----------
df = pd.read_excel(input_path)

# 自动识别 SMC 列
if smc_col is None:
    cand = [c for c in df.columns if re.search(r'\bsmc\b', str(c), re.IGNORECASE)]
    if not cand:
        # 放宽：含 smc 字样
        cand = [c for c in df.columns if re.search(r'smc', str(c), re.IGNORECASE)]
    if not cand:
        raise ValueError("未找到 SMC 列，请设置 smc_col 为实际列名。")
    smc_col = cand[0]

# 自动识别 SDC 列（仅用于边界计算）
if sdc_col is None:
    cand = [c for c in df.columns if re.search(r'\bsdc\b', str(c), re.IGNORECASE)]
    if not cand:
        # 兼容 er/epsilon/eps/ε 等命名
        cand = [c for c in df.columns if re.search(r'(sdc|^er$|epsilon|eps|ε|dielectric|perm)', str(c), re.IGNORECASE)]
    if not cand:
        raise ValueError("未找到 SDC/εr 列，请设置 sdc_col 为实际列名。")
    sdc_col = cand[0]

# 检查波段列是否存在（特征，不含 SDC）
missing = [b for b in band_list if b not in df.columns]
if missing:
    raise ValueError(f"以下波段缺失：{missing}")

# 准备数据（X: 仅 band_list； y: SMC； er: SDC）
X = df[band_list].apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(df[smc_col], errors="coerce")
er = pd.to_numeric(df[sdc_col], errors="coerce")   # 仅用于边界

mask = (~X.isna().any(axis=1)) & (~y.isna()) & (~er.isna())
X, y, er = X[mask], y[mask], er[mask]

# --------- 定义 SMC 的上下边界（由 εr=SDC 计算） ----------
def theta_low(er_vals: np.ndarray) -> np.ndarray:
    # θ_low(εr) = 2.33e-2*εr - 3.6e-4*εr^2 + 1.91e-6*εr^3
    return 2.33e-2*er_vals - 3.6e-4*(er_vals**2) + 1.91e-6*(er_vals**3)

def theta_high(er_vals: np.ndarray) -> np.ndarray:
    # θ_high(εr) = 4e-2 + 2.33e-2*εr - 3.6e-4*εr^2 + 1.91e-6*εr^3
    return 4e-2 + 2.33e-2*er_vals - 3.6e-4*(er_vals**2) + 1.91e-6*(er_vals**3)

def apply_smc_bounds(y_pred: np.ndarray, er_vals: np.ndarray):
    """把模型输出的 SMC 预测裁剪到 [θ_low(εr), θ_high(εr)]。"""
    lo = theta_low(er_vals)
    hi = theta_high(er_vals)
    y_bounded = np.minimum(np.maximum(y_pred, lo), hi)
    clipped = (np.abs(y_bounded - y_pred) > 0)
    return y_bounded, lo, hi, clipped

# --------- 评估 ----------
def calc_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rpd = np.std(y_true, ddof=1) / rmse if rmse != 0 else np.nan
    return r2, rmse, rpd

# --------- CV 设置 ----------
cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

# --------- 模型与参数网格 ----------
# RF
rf = RandomForestRegressor(random_state=random_state)
rf_param_grid = {
    "n_estimators": [200, 400, 800],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
}

# SVR（带标准化）
svr_pipeline = Pipeline(steps=[("scaler", StandardScaler()), ("svr", SVR())])
svr_param_grid = {
    "svr__kernel": ["rbf"],
    "svr__C": [1, 5, 10, 20, 50],
    "svr__epsilon": [0.05, 0.1, 0.2],
    "svr__gamma": ["scale", "auto"]
}

# LightGBM
lgbm = LGBMRegressor(random_state=random_state, n_estimators=500)
lgbm_param_grid = {
    "num_leaves": [15, 31, 63],
    "max_depth": [-1, 8, 12],
    "learning_rate": [0.05, 0.1],
    "min_child_samples": [5, 10, 20],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

# --------- 网格搜索（R²） ----------
scoring = "r2"

rf_gs = GridSearchCV(rf, rf_param_grid, scoring=scoring, cv=cv, n_jobs=n_jobs, verbose=0)
rf_gs.fit(X, y)
rf_best = rf_gs.best_estimator_

svr_gs = GridSearchCV(svr_pipeline, svr_param_grid, scoring=scoring, cv=cv, n_jobs=n_jobs, verbose=0)
svr_gs.fit(X, y)
svr_best = svr_gs.best_estimator_

lgbm_gs = GridSearchCV(lgbm, lgbm_param_grid, scoring=scoring, cv=cv, n_jobs=n_jobs, verbose=0)
lgbm_gs.fit(X, y)
lgbm_best = lgbm_gs.best_estimator_

# --------- 逐模型：CV 预测 -> 对 SMC 施加边界 -> 评估 ----------
pred_df = pd.DataFrame({"SMC_True": y.values})
# 保存每个样本的 SMC 边界，便于审核
pred_df["SMC_Low_Bound"]  = theta_low(er.values)
pred_df["SMC_High_Bound"] = theta_high(er.values)

metrics_rows = []

def eval_one(model_name: str, est, best_params: dict, col_raw: str, col_bnd: str, col_flag: str):
    # 未裁剪的 OOF 预测（SMC）
    y_pred = cross_val_predict(est, X, y, cv=cv, n_jobs=n_jobs)
    r2_raw, rmse_raw, rpd_raw = calc_metrics(y, y_pred)

    # 对 SMC 预测施加边界（按每个样本的 εr 计算θ_low/θ_high）
    y_bnd, _, _, clipped_flag = apply_smc_bounds(y_pred, er.values)
    r2_bnd, rmse_bnd, rpd_bnd = calc_metrics(y, y_bnd)
    clipped_frac = float(np.mean(clipped_flag))

    pred_df[col_raw]  = y_pred
    pred_df[col_bnd]  = y_bnd
    pred_df[col_flag] = clipped_flag.astype(int)

    metrics_rows.extend([
        {"Model": model_name, "Type": "Raw",     "R2": r2_raw, "RMSE": rmse_raw, "RPD": rpd_raw,
         "Clipped_Fraction": 0.0, "BestParams": json.dumps(best_params, ensure_ascii=False)},
        {"Model": model_name, "Type": "Bounded", "R2": r2_bnd, "RMSE": rmse_bnd, "RPD": rpd_bnd,
         "Clipped_Fraction": clipped_frac, "BestParams": json.dumps(best_params, ensure_ascii=False)},
    ])

# RF
eval_one("RF", rf_best, rf_gs.best_params_, "Pred_RF_Raw", "Pred_RF_Bounded", "Clipped_RF")

# SVR
eval_one("SVR", svr_best, svr_gs.best_params_, "Pred_SVR_Raw", "Pred_SVR_Bounded", "Clipped_SVR")

# LightGBM
eval_one("LightGBM", lgbm_best, lgbm_gs.best_params_, "Pred_LightGBM_Raw", "Pred_LightGBM_Bounded", "Clipped_LightGBM")

metrics_df = pd.DataFrame(metrics_rows)

# --------- 写入 Excel ----------
with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    pred_df.to_excel(writer, sheet_name="Predictions", index=False)
    metrics_df.to_excel(writer, sheet_name="Model_Metrics", index=False)

print("✅ 完成：最优参数网格搜索 + CV 预测，并对 SMC 预测施加物理边界（由 SDC→θ_low/θ_high）")
print("输出文件：", output_path)
print("\n=== 模型指标（Raw vs. Bounded） ===")
print(metrics_df)
