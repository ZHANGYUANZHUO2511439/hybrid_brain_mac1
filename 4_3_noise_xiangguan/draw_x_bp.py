import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import butter, filtfilt

# ============================================================
# 1. 数据路径
# ============================================================
data_dir = Path(
    "/Users/zhangyuanzhuo/PycharmProjects/hybrid_brain_mac1/runs_tvb_be_sigma/baseline_240s_cut_20/sid_0_be_60.00_s_0.500_gee_0.400_sd_1/be60.00_s0.50_gee0.40_sd1"
)

# ============================================================
# 2. 加载数据
# ============================================================
rateE = np.load(data_dir / "rateE.npy")
time = np.load(data_dir / "time.npy") / 1000.0   # ms → s

print("rateE shape:", rateE.shape)
print("time range:", time[0], "~", time[-1], "s")

# ============================================================
# 3. 选择时间窗口
# ============================================================
t_start = 10
t_end = 70

mask = (time >= t_start) & (time <= t_end)

time_plot = time[mask]
rate_plot = rateE[mask]

# ============================================================
# 4. Downsample（非常重要）
# ============================================================
ds = 20

time_plot = time_plot[::ds]
rate_plot = rate_plot[::ds]

print("after downsample:", rate_plot.shape)

# ============================================================
# 5. sampling rate
# ============================================================
dt = np.median(np.diff(time_plot))
fs = 1.0 / dt

print("fs =", fs)

# ============================================================
# 6. bandpass filter
# ============================================================
def bandpass(x, fs, lo=0.16, hi=4.0, order=2):

    nyq = fs / 2.0

    b, a = butter(
        order,
        [lo / nyq, hi / nyq],
        btype='band'
    )

    return filtfilt(b, a, x)

# ============================================================
# 7. 提取 slow oscillation
# ============================================================
rate_bp = np.zeros_like(rate_plot)

n_regions = rate_plot.shape[1]

for i in range(n_regions):

    x = rate_plot[:, i]

    # 去均值
    x = x - np.mean(x)

    # bandpass
    x_bp = bandpass(
        x,
        fs,
        lo=0.16,
        hi=4.0
    )

    rate_bp[:, i] = x_bp

print("bandpass done")

# ============================================================
# 8. global mean
# ============================================================
mean_trace = rate_bp.mean(axis=1)

# ============================================================
# 9. 画图
# ============================================================
plt.figure(figsize=(18, 5))

# ============================================================
# 只画少量 region（推荐）
# ============================================================
regions_to_plot = [0, 5, 10, 20]

for i in regions_to_plot:

    plt.plot(
        time_plot,
        rate_bp[:, i],
        alpha=0.6,
        linewidth=1,
        label=f"region {i}"
    )

# ============================================================
# Global mean（重点）
# ============================================================
plt.plot(
    time_plot,
    mean_trace,
    color="black",
    linewidth=3,
    label="global mean"
)

# ============================================================
# 美化
# ============================================================
plt.xlabel("time [s]", fontsize=14)
plt.ylabel("SO activity", fontsize=14)

plt.title(
    "Slow Oscillation (0.16–4 Hz bandpass)",
    fontsize=18
)

plt.legend()

plt.tight_layout()

# ============================================================
# 10. 保存
# ============================================================
save_dir = Path("analysis/figures")
save_dir.mkdir(parents=True, exist_ok=True)

save_path = save_dir / "slow_oscillation_trace.png"

plt.savefig(save_path, dpi=300)

print("saved to:", save_path)

plt.show()