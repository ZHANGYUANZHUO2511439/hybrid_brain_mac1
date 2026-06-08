import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import welch

# ============================================================
# 1. 数据路径
# ============================================================

# ------------------------------------------------------------
# baseline
# ------------------------------------------------------------
baseline_dir = Path(
    "/Users/zhangyuanzhuo/PycharmProjects/hybrid_brain_mac1/runs_tvb_be_sigma/baseline_240s_cut_20/sid_0_be_60.00_s_0.500_gee_0.400_sd_1/be60.00_s0.50_gee0.40_sd1"
)

# ------------------------------------------------------------
# up-phase stimulation
# ------------------------------------------------------------
up_dir = Path(
    "/Users/zhangyuanzhuo/PycharmProjects/hybrid_brain_mac1/runs_tvb_be_sigma/weight_0.00039_in_tools_up_side/sid_0_be_60.00_s_0.500_gee_0.400_sd_1/be60.00_s0.50_gee0.40_sd1"
)

# ------------------------------------------------------------
# down-phase stimulation
# ------------------------------------------------------------
down_dir = Path(
    "/Users/zhangyuanzhuo/PycharmProjects/hybrid_brain_mac1/runs_tvb_be_sigma/weight_0.00039_in_tools_simulation_down_side/sid_0_be_60.00_s_0.500_gee_0.400_sd_1/be60.00_s0.50_gee0.40_sd1"
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

    print("\n===================================")
    print("loading:", data_dir)
    print("rateE shape:", rateE.shape)
    print("time range:", time[0], "~", time[-1])

    # --------------------------------------------------------
    # 去掉transient
    # --------------------------------------------------------
    cut_start = 10

    mask = time >= cut_start

    time = time[mask]
    rateE = rateE[mask]

    # --------------------------------------------------------
    # 全脑平均
    # --------------------------------------------------------
    global_mean = rateE.mean(axis=1)

    # --------------------------------------------------------
    # sampling rate
    # --------------------------------------------------------
    dt = np.median(np.diff(time))

    fs = 1.0 / dt

    print("fs =", fs)

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
# 3. peak检测
# ============================================================

def get_peak(freqs, psd, fmin=0.0, fmax=5.0):

    mask = (freqs >= fmin) & (freqs <= fmax)

    freqs_local = freqs[mask]
    psd_local = psd[mask]

    idx = np.argmax(psd_local)

    peak_freq = freqs_local[idx]
    peak_power = psd_local[idx]

    return peak_freq, peak_power

# ============================================================
# 4. band power
# ============================================================

def band_power(freqs, psd, fmin, fmax):

    mask = (freqs >= fmin) & (freqs <= fmax)

    power = np.trapz(
        psd[mask],
        freqs[mask]
    )

    return power

# ============================================================
# 5. compute PSD
# ============================================================

freqs_base, psd_base = compute_psd(baseline_dir)

freqs_up, psd_up = compute_psd(up_dir)

freqs_down, psd_down = compute_psd(down_dir)

# ============================================================
# 6. peak info
# ============================================================

peak_base, power_base = get_peak(freqs_base, psd_base)

peak_up, power_up = get_peak(freqs_up, psd_up)

peak_down, power_down = get_peak(freqs_down, psd_down)

# ============================================================
# 7. SO ratio
# ============================================================

SO_MIN = 0.16
SO_MAX = 1.25

TOTAL_MIN = 0.0
TOTAL_MAX = 5.0

# ------------------------------------------------------------
# baseline
# ------------------------------------------------------------
so_base = band_power(
    freqs_base,
    psd_base,
    SO_MIN,
    SO_MAX
)

total_base = band_power(
    freqs_base,
    psd_base,
    TOTAL_MIN,
    TOTAL_MAX
)

ratio_base = so_base / total_base

# ------------------------------------------------------------
# up
# ------------------------------------------------------------
so_up = band_power(
    freqs_up,
    psd_up,
    SO_MIN,
    SO_MAX
)

total_up = band_power(
    freqs_up,
    psd_up,
    TOTAL_MIN,
    TOTAL_MAX
)

ratio_up = so_up / total_up

# ------------------------------------------------------------
# down
# ------------------------------------------------------------
so_down = band_power(
    freqs_down,
    psd_down,
    SO_MIN,
    SO_MAX
)

total_down = band_power(
    freqs_down,
    psd_down,
    TOTAL_MIN,
    TOTAL_MAX
)

ratio_down = so_down / total_down

# ============================================================
# 8. print result
# ============================================================

print("\n===================================")
print("BASELINE")
print("===================================")

print("peak frequency =", peak_base)
print("peak power     =", power_base)
print("SO ratio       =", ratio_base)

print("\n===================================")
print("UP STIMULATION")
print("===================================")

print("peak frequency =", peak_up)
print("peak power     =", power_up)
print("SO ratio       =", ratio_up)

print("\n===================================")
print("DOWN STIMULATION")
print("===================================")

print("peak frequency =", peak_down)
print("peak power     =", power_down)
print("SO ratio       =", ratio_down)

# ============================================================
# 9. plot
# ============================================================

plt.figure(figsize=(12,7))

# ------------------------------------------------------------
# baseline
# ------------------------------------------------------------
plt.plot(
    freqs_base,
    psd_base,
    linewidth=3,
    label=f"Baseline ({peak_base:.2f} Hz)"
)

# ------------------------------------------------------------
# up
# ------------------------------------------------------------
plt.plot(
    freqs_up,
    psd_up,
    linewidth=3,
    label=f"Up-phase ({peak_up:.2f} Hz)"
)

# ------------------------------------------------------------
# down
# ------------------------------------------------------------
plt.plot(
    freqs_down,
    psd_down,
    linewidth=3,
    label=f"Down-phase ({peak_down:.2f} Hz)"
)

# ============================================================
# SO band
# ============================================================

plt.axvspan(
    SO_MIN,
    SO_MAX,
    alpha=0.2,
    label="SO band"
)

# ============================================================
# beauty
# ============================================================

plt.xlabel("Frequency [Hz]", fontsize=16)

plt.ylabel("PSD", fontsize=16)

plt.title(
    "PSD Comparison: Baseline vs Phase-dependent Stimulation",
    fontsize=22
)

plt.xlim(0, 5)

plt.legend(fontsize=13)

plt.tight_layout()

# ============================================================
# 10. save
# ============================================================

save_dir = Path("analysis/figures")

save_dir.mkdir(parents=True, exist_ok=True)

save_path = save_dir / "PSD_3group_comparison.png"

plt.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight"
)

print("\nsaved to:")
print(save_path)

plt.show()