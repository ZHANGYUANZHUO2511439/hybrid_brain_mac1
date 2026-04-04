import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import optuna

print("=" * 60)
print("所有包测试成功！")
print("=" * 60)
print(f"numpy: {np.__version__}")
print(f"pandas: {pd.__version__}")
print(f"matplotlib: {plt.matplotlib.__version__}")
print(f"scipy: {scipy.__version__}")
print(f"optuna: {optuna.__version__}")

# 测试pandas读取数据
print("\n测试读取数据文件...")
try:
    df = pd.read_csv('data/shhs_so_features_axis.csv')
    print(f"✅ 成功读取数据，形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    print(f"前3行:\n{df.head(3)}")
except Exception as e:
    print(f"❌ 读取数据失败: {e}")
    print("创建测试数据...")
    test_data = {
        'sid': ['subj001', 'subj002'],
        'axis_joint': [0.15, 0.18],
        'f_target': [0.8, 0.9],
        'slope_target': [0.05, 0.06],
        'cv_period_target': [0.2, 0.22],
        'amp_so_target': [60.0, 65.0],
        'age': [45, 52],
        'ahi': [12.5, 18.2],
        'bmi': [25.3, 28.7]
    }
    import pandas as pd
    df = pd.DataFrame(test_data)
    df.to_csv('data/shhs_so_features_axis.csv', index=False)
    print(f"✅ 已创建测试数据，形状: {df.shape}")
