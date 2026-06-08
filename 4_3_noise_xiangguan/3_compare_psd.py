import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import welch

# ============================================================
# 1. 数据路径
# ============================================================

# 上升沿刺激
up_dir = Path(
    "/Users/zhangyuanzhuo/PycharmProjects/hybrid_brain_mac1/runs_tvb_be_sigma/point_simulation_120s_upside/sid_0_be_60.00_s_0.500_gee_0.400_sd_1/be60.00_s0.50_gee0.40_sd1"
)

# 下降沿刺激
down_dir = Path(
    "/Users/zhangyuanzhuo/PycharmProjects/hybrid_brain_mac1/runs_tvb_be_sigma/point_simulation_120s_lowest/sid_0_be_60.00_s_0.500_gee_0.400_sd_1/be60.00_s0.50_gee0.40_sd1"
)

# ============================================================
# 2. PSD函数
# ============================================================

def compute_psd(data_dir):

    # --------------------------------------------------------
    # load
    # --------------------------------------------------------
    rateE = np.load(data_dir / "rateE.npy")
    time = np.load(data_dir / "time.npy") / 1000.0

    # --------------------------------------------------------
    # remove transient
    # --------------------------------------------------------
    mask = time >= 110

    time = time[mask]
    rateE = rateE[mask]

    # --------------------------------------------------------
    # global mean
    # --------------------------------------------------------
    global_mean = rateE.mean(axis=1)

    # --------------------------------------------------------
    # sampling rate
    # --------------------------------------------------------
    dt = np.median(np.diff(time))
    fs = 1.0 / dt

    # --------------------------------------------------------
    # demean
    # --------------------------------------------------------
    x = global_mean - np.mean(global_mean)

    # --------------------------------------------------------
    # Welch PSD
    # --------------------------------------------------------
    freqs, psd = welch(
        x,
        fs=fs,
        nperseg=int(fs * 20),
        noverlap=int(fs * 10)
    )

    return freqs, psd

# ============================================================
# 3. 计算band power
# ============================================================

def band_power(freqs, psd, fmin, fmax):

    mask = (freqs >= fmin) & (freqs <= fmax)

    power = np.trapz(
        psd[mask],
        freqs[mask]
    )

    return power

# ============================================================
# 4. compute PSD
# ============================================================

freqs_up, psd_up = compute_psd(up_dir)
freqs_down, psd_down = compute_psd(down_dir)

# ============================================================
# 5. SO power
# ============================================================

SO_MIN = 0.16
SO_MAX = 1.25

TOTAL_MIN = 0.0
TOTAL_MAX = 5.0

# up
so_power_up = band_power(
    freqs_up,
    psd_up,
    SO_MIN,
    SO_MAX
)

total_power_up = band_power(
    freqs_up,
    psd_up,
    TOTAL_MIN,
    TOTAL_MAX
)

ratio_up = so_power_up / total_power_up

# down
so_power_down = band_power(
    freqs_down,
    psd_down,
    SO_MIN,
    SO_MAX
)

total_power_down = band_power(
    freqs_down,
    psd_down,
    TOTAL_MIN,
    TOTAL_MAX
)

ratio_down = so_power_down / total_power_down

# ============================================================
# 6. print result
# ============================================================

print("\n==============================")
print("UP stimulation")
print("==============================")

print("SO power      =", so_power_up)
print("Total power   =", total_power_up)
print("SO ratio      =", ratio_up)

print("\n==============================")
print("DOWN stimulation")
print("==============================")

print("SO power      =", so_power_down)
print("Total power   =", total_power_down)
print("SO ratio      =", ratio_down)

# ============================================================
# 7. plot PSD
# ============================================================

plt.figure(figsize=(10,6))

# ------------------------------------------------------------
# up
# ------------------------------------------------------------
plt.plot(
    freqs_up,
    psd_up,
    linewidth=3,
    label="Up-phase stimulation"
)

# ------------------------------------------------------------
# down
# ------------------------------------------------------------
plt.plot(
    freqs_down,
    psd_down,
    linewidth=3,
    label="Down-phase stimulation"
)

# ------------------------------------------------------------
# SO band
# ------------------------------------------------------------
plt.axvspan(
    SO_MIN,
    SO_MAX,
    alpha=0.2,
    label="SO band"
)

# ------------------------------------------------------------
# beauty
# ------------------------------------------------------------
plt.xlabel("Frequency [Hz]", fontsize=14)
plt.ylabel("PSD", fontsize=14)

plt.title(
    "PSD Comparison",
    fontsize=18
)

plt.xlim(0, 5)

plt.legend()

plt.tight_layout()

# ============================================================
# 8. save
# ============================================================

save_dir = Path("analysis/figures")
save_dir.mkdir(parents=True, exist_ok=True)

save_path = save_dir / "PSD_comparison.png"

plt.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight"
)

print("\nsaved to:")
print(save_path)

plt.show()