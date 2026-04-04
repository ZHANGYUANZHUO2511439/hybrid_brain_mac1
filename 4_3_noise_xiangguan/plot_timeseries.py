#检查noise
import numpy as np
import matplotlib.pyplot as plt
import os
import re

folder = "/Users/zhangyuanzhuo/PycharmProjects/hybrid_brain_mac1/runs_tvb_be_sigma/scan_seed1/sid_201774_be_60.00_s_0.500_gee_0.400_sd_1/be60.00_s0.50_gee0.40_sd1"

# === 1. 只提取 step_数字.npy ===
files = []
for f in os.listdir(folder):
    match = re.match(r"step_(\d+)\.npy", f)
    if match:
        files.append((int(match.group(1)), f))

# 排序
files.sort(key=lambda x: x[0])
files = [f[1] for f in files]

print("有效step文件:", files)

# === 2. 读取数据 ===
all_data = []

for f in files:
    path = os.path.join(folder, f)
    data = np.load(path, allow_pickle=True)

    print("====", f, "====")
    print("raw type:", type(data))

    # === 解析 object ===
    if isinstance(data, np.ndarray) and data.dtype == object:
        try:
            data = np.array([x[1] for x in data])
        except:
            print("❌ 解析失败:", f)
            continue

    print("after parse shape:", getattr(data, "shape", "no shape"))

    # === 提取 rateE ===
    if isinstance(data, np.ndarray):
        if data.ndim == 3:
            data = data[:, 0, :]
        elif data.ndim == 2:
            pass
        else:
            print("❌ ndim异常:", data.shape)
            continue
    else:
        continue

    # === 去空 ===
    if data.shape[0] == 0:
        print("❌ 空数据:", f)
        continue

    # === 统一shape ===
    if data.shape[0] < data.shape[1]:
        data = data.T

    print("final shape:", data.shape)
    all_data.append(data)

# === 3. 检查是否有数据 ===
if len(all_data) == 0:
    raise RuntimeError("❌ 没有任何有效数据（simulation可能失败）")

# === 4. 拼接 ===
min_regions = min(d.shape[1] for d in all_data)
all_data = [d[:, :min_regions] for d in all_data]

rateE = np.concatenate(all_data, axis=0)

print("最终shape:", rateE.shape)

# === 5. 去 transient ===
rateE = rateE[2000:]

# === 6. 时间轴 ===
dt = 0.001
time = np.arange(rateE.shape[0]) * dt

# === 7. 画图 ===
plt.figure(figsize=(12,4))
plt.plot(time, rateE[:, 0], color='black')
plt.title("FULL Time Series")
plt.xlabel("Time (s)")
plt.ylabel("Rate")
plt.tight_layout()
plt.show()