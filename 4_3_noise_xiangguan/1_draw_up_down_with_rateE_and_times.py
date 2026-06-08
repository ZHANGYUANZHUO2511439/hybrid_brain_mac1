import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# 1. 指向你的数据路径（改这里）
# ============================================================
data_dir = Path(
    "/Users/zhangyuanzhuo/PycharmProjects/hybrid_brain_mac1/runs_tvb_be_sigma/point_simulation_120s_lowest/sid_0_be_60.00_s_0.500_gee_0.400_sd_1/be60.00_s0.50_gee0.40_sd1"

)

# ============================================================
# 2. 加载数据
# ============================================================
rateE = np.load(data_dir / "rateE.npy")
time = np.load(data_dir / "time.npy") / 1000  # ms → s

print("rateE shape:", rateE.shape)

# ============================================================
# 3. 选时间窗口（论文都是局部）
# ============================================================
t_start = 110
t_end = 150

mask = (time >= t_start) & (time <= t_end)
time_plot = time[mask]
rate_plot = rateE[mask]
# ============================================================
# 3.5 降采样（非常重要）
# ============================================================
ds = 10

time_plot = time_plot[::ds]
rate_plot = rate_plot[::ds]


# ============================================================
# 4. 画 up-down（论文风格）
# ============================================================
plt.figure(figsize=(12,4))

# 多脑区（红色透明）
for i in range(68):
    plt.plot(time_plot, rate_plot[:, i],
             color="red", alpha=0.03, linewidth=0.5)

# 平均（蓝色）
mean_rate = rate_plot.mean(axis=1)
plt.plot(time_plot, mean_rate, color="blue", linewidth=2)

plt.xlabel("time [s]")
plt.ylabel("firing rate [Hz]")
plt.title("Up-Down state")
plt.ylim(0, 0.03)


plt.tight_layout()

# ============================================================
# 5. 保存图片
# ============================================================
save_dir = Path("analysis/figures")
save_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(save_dir / "up_down.png", dpi=300)
plt.show()