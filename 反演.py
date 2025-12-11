import os
import re
import numpy as np
import pandas as pd
import rasterio
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from lightgbm import LGBMRegressor

# ====================== 配置区 ======================
input_table_path = r"F:\资源\AAA项目论文\小论文\Paper1\Python\\4Hybrid_model\\2_SG平滑.xlsx"  # 训练数据表（含 SMC、SDC、Bxxx）
input_raster_path = r"F:\资源\AAA项目论文\小论文\Paper1\Python\第二次合成\\2_cut_quyun_softGPU_strong.tif"         # 要预测的高光谱影像（第 i 个波段即 Band_i）

# 输出文件（Raw：未裁剪；Bounded：用“学到的边界”裁剪）
out_rf_raw  = r"F:\资源\AAA项目论文\小论文\Paper1\Python\\4Hybrid_model\SMC_pred_RF_raw.tif"
out_svr_raw = r"F:\资源\AAA项目论文\小论文\Paper1\Python\\4Hybrid_model\SMC_pred_SVR_raw.tif"
out_lgb_raw = r"F:\资源\AAA项目论文\小论文\Paper1\Python\\4Hybrid_model\SMC_pred_LGB_raw.tif"

out_rf_bnd  = r"F:\资源\AAA项目论文\小论文\Paper1\Python\\4Hybrid_model\SMC_pred_RF_bounded.tif"
out_svr_bnd = r"F:\资源\AAA项目论文\小论文\Paper1\Python\\4Hybrid_model\SMC_pred_SVR_bounded.tif"
out_lgb_bnd = r"F:\资源\AAA项目论文\小论文\Paper1\Python\\4Hybrid_model\SMC_pred_LGB_bounded.tif"

# 特征/标签列
bands_used = ["B215", "B119", "B31", "B219", "B214", "B20", "B10"]
smc_col = "SMC"
sdc_col = "SDC"  # 仅用于训练：计算每个样本的 θ_low/θ_high 并训练边界回归器

# —— 你给出的最优参数（按你上条消息）——
rf_best_params = {
    "max_depth": None, "max_features": "sqrt", "min_samples_leaf": 4,
    "min_samples_split": 10, "n_estimators": 800, "n_jobs": -1, "random_state": 42
}
svr_best_params = {"svr__C": 1, "svr__epsilon": 0.05, "svr__gamma": "auto", "svr__kernel": "rbf"}
lgb_best_params = {
    "colsample_bytree": 0.8, "learning_rate": 0.05, "max_depth": -1, "min_child_samples": 20,
    "num_leaves": 15, "subsample": 0.8, "n_estimators": 500, "random_state": 42, "n_jobs": -1
}

# 边界回归器（用光谱→预测 θ_low 与 θ_high），这俩可以用轻量 LGBM
bound_lgb_params = {
    "n_estimators": 300, "num_leaves": 31, "max_depth": -1, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42, "n_jobs": -1
}

# 输出压缩（需要就打开）
# extra_profile_opts = {"compress": "LZW"}
extra_profile_opts = {}
# ====================================================

# --- 物理边界（由 εr=SDC 计算） ---
def theta_low(er_vals: np.ndarray) -> np.ndarray:
    return 2.33e-2*er_vals - 3.6e-4*(er_vals**2) + 1.91e-6*(er_vals**3)

def theta_high(er_vals: np.ndarray) -> np.ndarray:
    return 4e-2 + 2.33e-2*er_vals - 3.6e-4*(er_vals**2) + 1.91e-6*(er_vals**3)

# --- 读取训练表并准备数据 ---
df = pd.read_excel(input_table_path, engine="openpyxl")
for col in [smc_col, sdc_col] + bands_used:
    if col not in df.columns:
        raise ValueError(f"训练表缺列：{col}")

X = df[bands_used].apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(df[smc_col], errors="coerce")
er = pd.to_numeric(df[sdc_col], errors="coerce")

mask = (~X.isna().any(axis=1)) & (~y.isna()) & (~er.isna())
X, y, er = X.loc[mask].copy(), y.loc[mask].copy(), er.loc[mask].copy()

# --- 生成每个样本的“目标边界”并训练边界回归器（光谱→θ_low_hat, θ_high_hat） ---
theta_lo = theta_low(er.values)
theta_hi = theta_high(er.values)

# ① 训练三个 SMC 模型（全样本拟合）
rf = RandomForestRegressor(**rf_best_params).fit(X, y)
svr_pipe = Pipeline([("scaler", StandardScaler()), ("svr", SVR())])
svr_pipe.set_params(**svr_best_params)
svr_pipe.fit(X, y)
lgb_reg = LGBMRegressor(**lgb_best_params).fit(X, y)

# ② 训练两个边界回归器（用相同特征 X）
lo_reg = LGBMRegressor(**bound_lgb_params).fit(X, theta_lo)
hi_reg = LGBMRegressor(**bound_lgb_params).fit(X, theta_hi)

print("✅ 模型已在全部训练样本上拟合完成（含 SMC 三模型 + 边界双模型）。")

# --- 带边界夹紧 ---
def clamp_with_hat(pred: np.ndarray, lo_hat: np.ndarray, hi_hat: np.ndarray) -> np.ndarray:
    # 确保 hi_hat >= lo_hat
    hi_hat = np.maximum(hi_hat, lo_hat)
    return np.minimum(np.maximum(pred, lo_hat), hi_hat).astype(np.float32)

# --- Bxxx → 影像第 xxx 波段索引 ---
def parse_band_index(bname: str) -> int:
    m = re.match(r"B(\d+)$", bname.strip(), re.IGNORECASE)
    if not m:
        raise ValueError(f"无法解析波段名：{bname}")
    return int(m.group(1))
band_indices = [parse_band_index(b) for b in bands_used]

# --- 分块预测函数 ---
def predict_block(block_stack, rf_model, svr_model, lgb_model, lo_model, hi_model):
    # block_stack: (n_bands, h, w)
    n_b, h, w = block_stack.shape
    Xb = np.moveaxis(block_stack, 0, -1).reshape(-1, n_b)  # (h*w, n_bands)
    valid = np.isfinite(Xb).all(axis=1)

    rf_pred  = np.full((Xb.shape[0],), np.nan, dtype=np.float32)
    svr_pred = np.full((Xb.shape[0],), np.nan, dtype=np.float32)
    lgb_pred = np.full((Xb.shape[0],), np.nan, dtype=np.float32)
    lo_hat   = np.full((Xb.shape[0],), np.nan, dtype=np.float32)
    hi_hat   = np.full((Xb.shape[0],), np.nan, dtype=np.float32)

    if valid.any():
        Xv = Xb[valid, :]
        rf_pred[valid]  = rf_model.predict(Xv).astype(np.float32)
        svr_pred[valid] = svr_model.predict(Xv).astype(np.float32)
        lgb_pred[valid] = lgb_model.predict(Xv).astype(np.float32)
        lo_hat[valid]   = lo_model.predict(Xv).astype(np.float32)
        hi_hat[valid]   = hi_model.predict(Xv).astype(np.float32)

    # 夹紧
    rf_bnd  = clamp_with_hat(rf_pred,  lo_hat, hi_hat)
    svr_bnd = clamp_with_hat(svr_pred, lo_hat, hi_hat)
    lgb_bnd = clamp_with_hat(lgb_pred, lo_hat, hi_hat)

    # 还原为 (h, w)
    to_img = lambda a: a.reshape(h, w)
    return (to_img(rf_pred), to_img(svr_pred), to_img(lgb_pred),
            to_img(rf_bnd), to_img(svr_bnd), to_img(lgb_bnd))

# --- 逐块读取影像并写出 6 个结果 ---
with rasterio.open(input_raster_path) as src:
    # 波段存在性检查
    for idx in band_indices:
        if not (1 <= idx <= src.count):
            raise ValueError(f"影像中不存在第 {idx} 个波段（总波段数 {src.count}）。")

    base_profile = src.profile.copy()
    nodata_val = src.nodata

    out_prof = base_profile.copy()
    out_prof.update(count=1, dtype="float32", nodata=np.nan, **extra_profile_opts)

    dst_rf_raw  = rasterio.open(out_rf_raw,  "w", **out_prof)
    dst_svr_raw = rasterio.open(out_svr_raw, "w", **out_prof)
    dst_lgb_raw = rasterio.open(out_lgb_raw, "w", **out_prof)

    dst_rf_bnd  = rasterio.open(out_rf_bnd,  "w", **out_prof)
    dst_svr_bnd = rasterio.open(out_svr_bnd, "w", **out_prof)
    dst_lgb_bnd = rasterio.open(out_lgb_bnd, "w", **out_prof)

    try:
        for ji, window in src.block_windows(1):
            # 取所需波段并堆叠 (n_bands, h, w)
            blocks = []
            for idx in band_indices:
                arr = src.read(idx, window=window).astype(np.float32)
                if nodata_val is not None:
                    arr = np.where(arr == nodata_val, np.nan, arr)
                blocks.append(arr)
            stack = np.stack(blocks, axis=0)

            (rf_block_raw, svr_block_raw, lgb_block_raw,
             rf_block_bnd, svr_block_bnd, lgb_block_bnd) = predict_block(
                stack, rf, svr_pipe, lgb_reg, lo_reg, hi_reg
            )

            # 写 Raw
            dst_rf_raw.write(rf_block_raw, 1, window=window)
            dst_svr_raw.write(svr_block_raw, 1, window=window)
            dst_lgb_raw.write(lgb_block_raw, 1, window=window)

            # 写 Bounded
            dst_rf_bnd.write(rf_block_bnd, 1, window=window)
            dst_svr_bnd.write(svr_block_bnd, 1, window=window)
            dst_lgb_bnd.write(lgb_block_bnd, 1, window=window)
    finally:
        dst_rf_raw.close(); dst_svr_raw.close(); dst_lgb_raw.close()
        dst_rf_bnd.close(); dst_svr_bnd.close(); dst_lgb_bnd.close()

print("✅ 预测完成：")
print(f" - Raw：{out_rf_raw}, {out_svr_raw}, {out_lgb_raw}")
print(f" - Bounded（由边界回归器夹紧）：{out_rf_bnd}, {out_svr_bnd}, {out_lgb_bnd}")
