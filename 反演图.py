import os
import re
import numpy as np
import pandas as pd
import rasterio
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import lightgbm as lgb

# ====================== 配置区 ======================
input_table_path = r"F:\资源\AAA项目论文\小论文\Paper1\Python\\2SHR_ML\\2_SG平滑.xlsx"     # 训练数据表
input_raster_path = r"F:\资源\AAA项目论文\小论文\Paper1\Python\第二次合成\\2_cut_quyun_softGPU_strong.tif"            # 要预测的高光谱影像（Band_1, Band_2, ...）

out_rf_path  = r"F:\资源\AAA项目论文\小论文\Paper1\Python\\2SHR_ML\SMC_pred_RF.tif"       # RF 单波段输出
out_svr_path = r"F:\资源\AAA项目论文\小论文\Paper1\Python\\2SHR_ML\SMC_pred_SVR.tif"      # SVR 单波段输出
out_lgb_path = r"F:\资源\AAA项目论文\小论文\Paper1\Python\\2SHR_ML\SMC_pred_LGB.tif"      # LGB 单波段输出

bands_used = ["B215", "B119", "B31", "B219", "B214", "B20", "B10"]
smc_col = None                 # 若SMC列名不是"SMC"，可直接填字符串
random_state = 42
n_jobs = -1

# 影像波段命名格式，{idx} 会被替换为波段序号，如 Band_215
band_name_fmt = "Band_{idx}"

# 你给定的最优参数
rf_best_params = {"max_depth": None, "max_features": "sqrt", "min_samples_leaf": 4,
                  "min_samples_split": 10, "n_estimators": 800}
svr_best_params = {"svr__C": 1, "svr__epsilon": 0.05, "svr__gamma": "auto", "svr__kernel": "rbf"}
lgb_best_params = {"colsample_bytree": 0.8, "learning_rate": 0.01, "max_depth": 3,
                   "n_estimators": 300, "num_leaves": 31, "subsample": 0.8}
# ====================================================

# -------- 读取训练表并准备数据 --------
df = pd.read_excel(input_table_path, engine="openpyxl")
if smc_col is None:
    import re as _re
    cands = [c for c in df.columns if _re.search(r'smc', str(c), _re.IGNORECASE)]
    if not cands:
        raise ValueError("未找到 SMC 列，请设置 smc_col 为实际列名。")
    smc_col = cands[0]

missing = [b for b in bands_used if b not in df.columns]
if missing:
    raise ValueError(f"训练表缺失波段列：{missing}")

X = df[bands_used].apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(df[smc_col], errors="coerce")
mask = (~X.isna().any(axis=1)) & (~y.isna())
X = X.loc[mask].copy()
y = y.loc[mask].copy()

# -------- 拟合三个模型（全样本） --------
rf = RandomForestRegressor(random_state=random_state, n_jobs=n_jobs, **rf_best_params)
rf.fit(X, y)

svr_pipe = Pipeline([("scaler", StandardScaler()), ("svr", SVR())])
svr_pipe.set_params(**svr_best_params)
svr_pipe.fit(X, y)

lgb_reg = lgb.LGBMRegressor(random_state=random_state, n_jobs=n_jobs, **lgb_best_params)
lgb_reg.fit(X, y)

print("✅ 模型已在全部训练样本上拟合完成。")

# -------- 建立 Bxxx → Band_xxx 的索引映射并检查 --------
def parse_band_index(bname):
    m = re.match(r"B(\d+)$", bname.strip(), re.IGNORECASE)
    if not m:
        raise ValueError(f"无法解析波段名：{bname}")
    return int(m.group(1))

band_indices = [parse_band_index(b) for b in bands_used]

with rasterio.open(input_raster_path) as src:
    count = src.count
    for i in band_indices:
        if not (1 <= i <= count):
            raise ValueError(f"影像中不存在第 {i} 个波段（总波段数 {count}）。")
    profile = src.profile.copy()
    nodata_val = src.nodata

# 单波段输出配置（float32、nodata=NaN）
out_prof = profile.copy()
out_prof.update(count=1, dtype="float32", nodata=np.nan)

# -------- 逐块预测并分别写出到三个文件 --------
def predict_block(block_stack, rf_model, svr_model, lgb_model):
    # block_stack: (n_bands, h, w)
    n_b, h, w = block_stack.shape
    Xb = np.moveaxis(block_stack, 0, -1).reshape(-1, n_b)   # (h*w, n_bands)
    valid = np.isfinite(Xb).all(axis=1)

    rf_pred  = np.full((Xb.shape[0],), np.nan, dtype=np.float32)
    svr_pred = np.full((Xb.shape[0],), np.nan, dtype=np.float32)
    lgb_pred = np.full((Xb.shape[0],), np.nan, dtype=np.float32)

    if valid.any():
        Xv = Xb[valid, :]
        rf_pred [valid] = rf_model.predict(Xv).astype(np.float32)
        svr_pred[valid] = svr_model.predict(Xv).astype(np.float32)
        lgb_pred[valid] = lgb_model.predict(Xv).astype(np.float32)

    # 还原为 (h, w)
    return (rf_pred.reshape(h, w),
            svr_pred.reshape(h, w),
            lgb_pred.reshape(h, w))

with rasterio.open(input_raster_path) as src, \
     rasterio.open(out_rf_path,  "w", **out_prof) as dst_rf, \
     rasterio.open(out_svr_path, "w", **out_prof) as dst_svr, \
     rasterio.open(out_lgb_path, "w", **out_prof) as dst_lgb:

    for ji, window in src.block_windows(1):
        # 读取所需波段并堆叠 (n_bands, h, w)
        blocks = []
        for idx in band_indices:
            arr = src.read(idx, window=window).astype(np.float32)
            if nodata_val is not None:
                arr = np.where(arr == nodata_val, np.nan, arr)
            blocks.append(arr)
        stack = np.stack(blocks, axis=0)

        rf_block, svr_block, lgb_block = predict_block(stack, rf, svr_pipe, lgb_reg)

        dst_rf.write(rf_block, 1, window=window)
        dst_svr.write(svr_block, 1, window=window)
        dst_lgb.write(lgb_block, 1, window=window)

print("✅ 完成预测：")
print(f" - RF  输出：{out_rf_path}")
print(f" - SVR 输出：{out_svr_path}")
print(f" - LGB 输出：{out_lgb_path}")

