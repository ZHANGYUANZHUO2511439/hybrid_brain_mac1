#!/usr/bin/env python3
"""
完整的TVB环境测试脚本
"""
import sys
import os

print("=" * 70)
print("TVB 参数扫描流水线 - 完整环境测试")
print("=" * 70)

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("\n[1/6] Python 环境信息")
print(f"Python 版本: {sys.version}")
print(f"Python 路径: {sys.executable}")
print(f"工作目录: {os.getcwd()}")

print("\n[2/6] 检查核心包")
packages = ['numpy', 'pandas', 'matplotlib', 'scipy', 'optuna']
for pkg in packages:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, '__version__', '未知版本')
        print(f"  ✅ {pkg}: {ver}")
    except ImportError:
        print(f"  ❌ {pkg}: 未安装")

print("\n[3/6] 检查项目模块")
try:
    import tvb_stim
    print("  ✅ tvb_stim: 导入成功")
    print(f"    可用函数: {[x for x in dir(tvb_stim) if not x.startswith('_')][:10]}...")
except ImportError as e:
    print(f"  ❌ tvb_stim: {e}")

try:
    import tvb_rl_pipeline
    print("  ✅ tvb_rl_pipeline: 导入成功")
except ImportError as e:
    print(f"  ❌ tvb_rl_pipeline: {e}")

print("\n[4/6] 检查数据文件")
data_files = [
    'data/shhs_so_features_axis.csv',
    'data/shhs_so_features_axis.csv.gz',
    'data/so_features.csv',
    'data/sh*.csv',
]

found_data = False
for pattern in data_files:
    import glob
    files = glob.glob(pattern)
    for f in files:
        if os.path.exists(f):
            print(f"  ✅ 找到数据文件: {f}")
            # 尝试读取几行
            try:
                import pandas as pd
                df = pd.read_csv(f, nrows=3)
                print(f"     列名: {list(df.columns)}")
                print(f"     形状: {df.shape}")
            except:
                print(f"     无法读取CSV")
            found_data = True

if not found_data:
    print("  ⚠️  未找到数据文件，将创建测试数据")
    # 创建测试数据
    import pandas as pd
    import numpy as np
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
    os.makedirs('data', exist_ok=True)
    df_test = pd.DataFrame(test_data)
    df_test.to_csv('data/shhs_so_features_axis.csv', index=False)
    print(f"  ✅ 已创建测试数据: data/shhs_so_features_axis.csv")

print("\n[5/6] 尝试导入主模块")
try:
    import tvb_bo_pipeline as tbp
    print("  ✅ tvb_bo_pipeline: 导入成功")
    
    # 检查可用的函数
    functions = [x for x in dir(tbp) if not x.startswith('_') and callable(getattr(tbp, x))]
    print(f"    可用函数: {functions[:15]}...")
    
    print("\n[6/6] 运行最小测试")
    print("  创建测试输出目录...")
    os.makedirs('test_output', exist_ok=True)
    
    # 尝试运行最简单的演示
    print("  尝试运行 demo_scan_be_sigma_for_veviw()...")
    try:
        tbp.demo_scan_be_sigma_for_veviw()
        print("  ✅ 演示运行成功！")
    except Exception as e:
        print(f"  ⚠️  演示运行出错: {e}")
        print("  尝试更简单的测试...")
        
        # 尝试创建模拟Subject并运行
        print("  创建模拟Subject测试...")
        from dataclasses import dataclass
        
        @dataclass
        class MockSubject:
            sid: str
            axis_joint: float
            f_target: float
            slope_target: float
            cv_period_target: float
            amp_so_target: float
            age: float = 50.0
            ahi: float = 15.0
            bmi: float = 26.0
        
        mock_subj = MockSubject(
            sid='test001',
            axis_joint=0.2,
            f_target=0.8,
            slope_target=0.05,
            cv_period_target=0.22,
            amp_so_target=60.0
        )
        
        print(f"  创建了模拟Subject: {mock_subj.sid}")
        
except ImportError as e:
    print(f"  ❌ tvb_bo_pipeline 导入失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("测试完成！")
print("输出目录: test_output/")
print("=" * 70)
