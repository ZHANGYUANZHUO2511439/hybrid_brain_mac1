import sys
import os

print("最终修复导入问题...")

# 添加路径
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('tvb_adex_ref'))

print(f"当前目录: {os.getcwd()}")
print(f"Python路径前5个:")
for i, p in enumerate(sys.path[:5]):
    print(f"  {i}: {p}")

# 测试导入
print("\n测试导入tvb_adex_ref...")
try:
    import tvb_adex_ref
    print("✅ tvb_adex_ref 导入成功")
    print(f"  位置: {tvb_adex_ref.__file__}")
except Exception as e:
    print(f"❌ tvb_adex_ref 导入失败: {e}")

print("\n测试导入tvb_model_reference...")
try:
    import tvb_adex_ref.tvb_model_reference
    print("✅ tvb_adex_ref.tvb_model_reference 导入成功")
    
    # 设置别名
    sys.modules['tvb_model_reference'] = tvb_adex_ref.tvb_model_reference
    print("✅ 设置tvb_model_reference别名")
    
except Exception as e:
    print(f"❌ 导入失败: {e}")
    
    # 尝试直接注入
    print("尝试直接注入模块...")
    import types
    mock_module = types.ModuleType('tvb_model_reference')
    sys.modules['tvb_model_reference'] = mock_module
    print("✅ 已注入模拟模块")

print("\n测试导入Zerlaut...")
try:
    import tvb_adex_ref.tvb_model_reference.src.Zerlaut as Zerlaut
    print("✅ Zerlaut 导入成功")
    
    # 设置别名
    sys.modules['tvb_model_reference.src.Zerlaut'] = Zerlaut
    print("✅ 设置Zerlaut别名")
    
except Exception as e:
    print(f"❌ Zerlaut导入失败: {e}")

print("\n现在尝试导入tvb_stim...")
try:
    import tvb_stim_copy_5_19
    print("✅ tvb_stim 导入成功")
    
    # 列出函数
    funcs = [f for f in dir(tvb_stim) if 'tvb_sim' in f]
    print(f"  可用仿真函数: {funcs}")
    
except Exception as e:
    print(f"❌ tvb_stim导入失败: {e}")
    import traceback
    traceback.print_exc()

print("\n修复完成!")
