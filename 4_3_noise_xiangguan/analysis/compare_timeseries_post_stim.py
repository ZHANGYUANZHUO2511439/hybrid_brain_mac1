import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================
# 数据目录
# ==========================

baseline_dir = Path(
    "/Users/zhangyuanzhuo/PycharmProjects/hybrid_brain_mac1/runs_tvb_be_sigma/baseline_240s_cut_20/sid_0_be_60.00_s_0.500_gee_0.400_sd_1/be60.00_s0.50_gee0.40_sd1"
)

peak_dir = Path(
    "/Users/zhangyuanzhuo/PycharmProjects/hybrid_brain_mac1/runs_tvb_be_sigma/point_simulation_120s_peak/sid_0_be_60.00_s_0.500_gee_0.400_sd_1/be60.00_s0.50_gee0.40_sd1"
)

trough_dir = Path(
    "/Users/zhangyuanzhuo/PycharmProjects/hybrid_brain_mac1/runs_tvb_be_sigma/point_simulation_120s_through/sid_0_be_60.00_s_0.500_gee_0.400_sd_1/be60.00_s0.50_gee0.40_sd1"
)

# ==========================
# ROI
# ==========================

ROI = 4      # 第5脑区

# ==========================
# 读取
# ==========================

time_base = np.load(baseline_dir / "time.npy") / 1000.0
rate_base = np.load(baseline_dir / "rateE.npy")

time_peak = np.load(peak_dir / "time.npy") / 1000.0
rate_peak = np.load(peak_dir / "rateE.npy")

time_trough = np.load(trough_dir / "time.npy") / 1000.0
rate_trough = np.load(trough_dir / "rateE.npy")

print("baseline:", rate_base.shape)
print("peak:", rate_peak.shape)
print("trough:", rate_trough.shape)

# ==========================
# 如果是二维矩阵
# ==========================

if rate_base.ndim == 2:
    rate_base = rate_base[:, ROI]

if rate_peak.ndim == 2:
    rate_peak = rate_peak[:, ROI]

if rate_trough.ndim == 2:
    rate_trough = rate_trough[:, ROI]

# ==========================
# 只看刺激附近
# ==========================

t_start = 119
t_end = 123

mask_base = (time_base >= t_start) & (time_base <= t_end)
mask_peak = (time_peak >= t_start) & (time_peak <= t_end)
mask_trough = (time_trough >= t_start) & (time_trough <= t_end)

# ==========================
# 绘图
# ==========================

plt.figure(figsize=(12, 6))

plt.plot(
    time_base[mask_base],
    rate_base[mask_base],
    linewidth=2,
    label="Baseline"
)

plt.plot(
    time_peak[mask_peak],
    rate_peak[mask_peak],
    linewidth=2,
    label="Peak pulse"
)

plt.plot(
    time_trough[mask_trough],
    rate_trough[mask_trough],
    linewidth=2,
    label="Trough pulse"
)

plt.axvline(
    120,
    linestyle="--",
    linewidth=2,
    label="Stimulus"
)

plt.xlabel("Time [s]")
plt.ylabel("Firing rate")
plt.title("ROI-5 Response Around Single Pulse Stimulation")

plt.legend(loc="upper right")
plt.grid(alpha=0.3)

plt.show()