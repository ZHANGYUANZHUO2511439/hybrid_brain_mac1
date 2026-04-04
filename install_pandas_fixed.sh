#!/bin/bash
echo "===== 修复 pandas 安装 ====="

echo "1. 检查当前Python和pip..."
echo "Python3: $(which python3)"
echo "Pip3: $(which pip3)"
echo "Python3的pip: $(python3 -m pip --version 2>/dev/null || echo '未找到')"

echo -e "\n2. 卸载可能损坏的安装..."
python3 -m pip uninstall -y pandas 2>/dev/null || true

echo -e "\n3. 使用系统方法安装..."
# 尝试多种安装方法
methods=(
    "python3 -m pip install pandas"
    "python3 -m pip install pandas==2.0.3 --no-cache-dir"
    "pip3 install pandas --user"
)

for method in "${methods[@]}"; do
    echo -e "\n尝试: $method"
    if $method 2>&1 | tail -5; then
        echo "安装成功！"
        break
    else
        echo "安装失败，尝试下一种方法..."
    fi
done

echo -e "\n4. 验证安装..."
python3 -c "
import sys
print('Python路径:', sys.executable)

try:
    import pandas as pd
    print('✅ pandas 导入成功！')
    print('   版本:', pd.__version__)
    print('   位置:', pd.__file__)
    
    # 测试基本功能
    df = pd.DataFrame({'test': [1, 2, 3]})
    print('   测试DataFrame创建成功')
    print('   形状:', df.shape)
    
except Exception as e:
    print('❌ pandas 导入失败:', e)
    print('\\nPython路径列表:')
    for p in sys.path:
        print('   ', p)
"

echo -e "\n5. 如果需要，手动下载安装..."
# 如果上述方法都失败，提供手动安装选项
python3 -c "
import subprocess
import sys

print('\\n手动安装选项:')
print('1. 使用conda: conda install pandas')
print('2. 从源代码: git clone https://github.com/pandas-dev/pandas')
print('3. 使用虚拟环境: python3 -m venv venv && source venv/bin/activate')
print('4. 使用系统Python: /usr/bin/python3 -m pip install pandas')
"
