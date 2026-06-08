import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks

# ============================================================
# 1. 数据路径
# ============================================================

data_dir = Path(
    "/Users/zhangyuanzhuo/PycharmProjects/hybrid_brain_mac1/runs_tvb_be_sigma/weight_0.00039_in_tools_simulation_down_side/sid_0_be_60.00_s_0.500_gee_0.400_sd_1/be60.00_s0.50_gee0.40_sd1"
)

# ============================================================
# 2. load data
# ============================================================

rateE = np.load(data_dir / "rateE.npy")

time = np.load(data_dir / "time.npy") / 1000.0

print("rateE shape:", rateE.shape)

# ============================================================
# 3. remove transient
# ============================================================

cut_start = 10

mask = time >= cut_start

time = time[mask]

rateE = rateE[mask]

# ============================================================
# 4. global mean
# ============================================================

x = rateE.mean(axis=1)

# 去均值（更容易检测）
x = x - np.mean(x)

# ============================================================
# 5. peak detection
# ============================================================

# ------------------------------------------------------------
# 最小峰间隔（秒）
# 防止检测到太密集的小波动
# ------------------------------------------------------------

MIN_DISTANCE_SEC = 0.2

dt = np.median(np.diff(time))

distance_points = int(MIN_DISTANCE_SEC / dt)

# ------------------------------------------------------------
# peak prominence
# 控制峰明显程度
# ------------------------------------------------------------

PROMINENCE = 0.001

# ------------------------------------------------------------
# find peaks
# ------------------------------------------------------------

peaks, properties = find_peaks(
    x,
    distance=distance_points,
    prominence=PROMINENCE
)

# ============================================================
# 6. peak info
# ============================================================

peak_times = time[peaks]

peak_values = x[peaks]

print("number of peaks:", len(peaks))

print("mean peak amplitude:", np.mean(peak_values))

print("std peak amplitude:", np.std(peak_values))

# ============================================================
# 7. representative trace
# ============================================================

plt.figure(figsize=(14,5))

# ------------------------------------------------------------
# 只画前20秒
# ------------------------------------------------------------

plot_mask = time <= 30

plt.plot(
    time[plot_mask],
    x[plot_mask],
    linewidth=2,
    label="Global mean activity"
)

# ------------------------------------------------------------
# peaks
# ------------------------------------------------------------

peak_mask = peak_times <= 30

plt.scatter(
    peak_times[peak_mask],
    peak_values[peak_mask],
    s=60,
    label="Detected peaks"
)

# ------------------------------------------------------------
# beauty
# ------------------------------------------------------------

plt.xlabel("Time [s]", fontsize=14)

plt.ylabel("Activity", fontsize=14)

plt.title(
    "Peak Detection in Time Domain",
    fontsize=18
)

plt.legend()

plt.tight_layout()

# ============================================================
# 8. peak amplitude distribution
# ============================================================

plt.figure(figsize=(8,6))

plt.hist(
    peak_values,
    bins=30
)

plt.xlabel("Peak Amplitude", fontsize=14)

plt.ylabel("Count", fontsize=14)

plt.title(
    "Peak Amplitude Distribution",
    fontsize=18
)

plt.tight_layout()

# ============================================================
# 9. peak interval distribution
# ============================================================

peak_intervals = np.diff(peak_times)

print("mean peak interval =", np.mean(peak_intervals), "s")

print("estimated frequency =",
      1.0 / np.mean(peak_intervals),
      "Hz")

plt.figure(figsize=(8,6))

plt.hist(
    peak_intervals,
    bins=30
)

plt.xlabel("Peak Interval [s]", fontsize=14)

plt.ylabel("Count", fontsize=14)

plt.title(
    "Peak Interval Distribution",
    fontsize=18
)

plt.tight_layout()

# ============================================================
# 10. save
# ============================================================

save_dir = Path("analysis/figures")

save_dir.mkdir(parents=True, exist_ok=True)

plt.figure(1)

plt.savefig(
    save_dir / "peak_detection_trace.png",
    dpi=300,
    bbox_inches="tight"
)

plt.figure(2)

plt.savefig(
    save_dir / "peak_amplitude_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.figure(3)

plt.savefig(
    save_dir / "peak_interval_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

print("\nsaved figures to:")
print(save_dir)

plt.show()