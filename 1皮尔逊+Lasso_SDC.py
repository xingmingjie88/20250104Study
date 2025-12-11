import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def process_spectral_data():
    # 读取Excel文件
    try:
        df = pd.read_excel('2_SG平滑.xlsx', sheet_name='Sheet1')
        print("数据读取成功！")
        print(f"数据形状: {df.shape}")
    except Exception as e:
        print(f"读取文件失败: {e}")
        return
    
    # 设置第一列为索引（point列）
    df.set_index('point', inplace=True)
    
    # 提取SMC列和B6-B219波段列
    smc = df['SDC']
    bands = df.loc[:, 'B6':'B219']
    
    print(f"\nSMC数据形状: {smc.shape}")
    print(f"波段数据形状: {bands.shape}")
    
    # 计算皮尔逊相关系数
    pearson_corr = {}
    for band in bands.columns:
        correlation = smc.corr(bands[band])
        pearson_corr[band] = correlation
    
    # 转换为DataFrame并排序
    corr_df = pd.DataFrame.from_dict(pearson_corr, orient='index', columns=['correlation'])
    corr_df = corr_df.sort_values('correlation', key=abs, ascending=False)
    
    print(f"\n皮尔逊相关系数统计:")
    print(f"最大相关系数: {corr_df['correlation'].abs().max():.4f}")
    print(f"最小相关系数: {corr_df['correlation'].abs().min():.4f}")
    print(f"平均相关系数: {corr_df['correlation'].abs().mean():.4f}")
    
    # 设置阈值筛选特征
    threshold = 0.7
    selected_bands_pearson = corr_df[corr_df['correlation'].abs() >= threshold]
    
    print(f"\n使用阈值 {threshold} 筛选出的波段数量: {len(selected_bands_pearson)}")
    print("\n皮尔逊筛选出的波段:")
    for band in selected_bands_pearson.index:
        print(f"{band}: {selected_bands_pearson.loc[band, 'correlation']:.4f}")
    
    # 如果皮尔逊筛选出的波段为空，使用前10个相关性最高的波段
    if len(selected_bands_pearson) == 0:
        print("\n皮尔逊筛选无结果，使用前10个相关性最高的波段")
        selected_bands_pearson = corr_df.head(10)
        print("前10个相关性最高的波段:")
        for band in selected_bands_pearson.index:
            print(f"{band}: {selected_bands_pearson.loc[band, 'correlation']:.4f}")
    
    # LASSO特征选择
    print(f"\n{'='*50}")
    print("开始LASSO特征选择...")
    
    # 准备数据
    X_pearson = bands[selected_bands_pearson.index]
    y = smc.values
    
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pearson)
    
    # 使用LASSO进行特征选择
    lasso = Lasso(alpha=0.01, max_iter=10000, random_state=42)
    lasso.fit(X_scaled, y)
    
    # 获取非零系数的特征
    lasso_coef = pd.DataFrame({
        'band': selected_bands_pearson.index,
        'lasso_coefficient': lasso.coef_,
        'pearson_correlation': [selected_bands_pearson.loc[band, 'correlation'] for band in selected_bands_pearson.index]
    })
    
    # 筛选非零系数特征
    selected_bands_lasso = lasso_coef[lasso_coef['lasso_coefficient'] != 0]
    
    print(f"\nLASSO筛选后的波段数量: {len(selected_bands_lasso)}")
    print("\n最终筛选出的波段 (皮尔逊 + LASSO):")
    
    if len(selected_bands_lasso) > 0:
        selected_bands_lasso = selected_bands_lasso.sort_values('lasso_coefficient', key=abs, ascending=False)
        final_bands = []
        
        for idx, row in selected_bands_lasso.iterrows():
            band_name = row['band']
            lasso_coef_val = row['lasso_coefficient']
            pearson_corr_val = row['pearson_correlation']
            print(f"{band_name}: LASSO系数={lasso_coef_val:.4f}, 皮尔逊相关={pearson_corr_val:.4f}")
            final_bands.append(band_name)
        
        # 格式化输出最终波段
        print(f"\n最终筛选的波段: {', '.join(final_bands)}")
        
        # 保存结果到文件
        result_df = pd.DataFrame({
            'Band': final_bands,
            'LASSO_Coefficient': [selected_bands_lasso[selected_bands_lasso['band'] == band]['lasso_coefficient'].values[0] for band in final_bands],
            'Pearson_Correlation': [selected_bands_lasso[selected_bands_lasso['band'] == band]['pearson_correlation'].values[0] for band in final_bands]
        })
        
        result_df.to_excel('feature_selection_results.xlsx', index=False)
        print("\n结果已保存到 'feature_selection_results.xlsx'")
        
    else:
        print("LASSO筛选后无特征保留，使用皮尔逊筛选的前5个波段")
        final_bands = selected_bands_pearson.head(5).index.tolist()
        print(f"最终波段: {', '.join(final_bands)}")

if __name__ == "__main__":
    process_spectral_data()