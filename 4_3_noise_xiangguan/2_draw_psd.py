import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import welch

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
# 3. 去掉 transient
# ============================================================
cut_start = 10

mask = time >= cut_start

time = time[mask]
rateE = rateE[mask]

# ============================================================
# 4. 全脑平均 firing rate
# ============================================================
global_mean = rateE.mean(axis=1)

print("global_mean shape:", global_mean.shape)

# ============================================================
# 5. sampling rate
# ============================================================
dt = np.median(np.diff(time))

fs = 1.0 / dt

print("dt =", dt)
print("fs =", fs)

# ============================================================
# 6. 去均值
# ============================================================
x = global_mean - np.mean(global_mean)

# ============================================================
# 7. Welch PSD
# ============================================================
freqs, psd = welch(
    x,
    fs=fs,
    nperseg=int(fs * 20),   # 20 s window
    noverlap=int(fs * 10)
)

# ============================================================
# 8. 只保留 slow frequency
# ============================================================
fmin = 0.0
fmax = 5.0

mask_psd = (freqs >= fmin) & (freqs <= fmax)

freqs_plot = freqs[mask_psd]
psd_plot = psd[mask_psd]

# ============================================================
# 9. 找 SO peak
# ============================================================
so_band = (freqs_plot >= 0.16) & (freqs_plot <= 1.25)

if np.any(so_band):

    peak_idx = np.argmax(psd_plot[so_band])

    so_freq = freqs_plot[so_band][peak_idx]

    print("SO peak frequency =", so_freq, "Hz")

else:

    so_freq = np.nan

# ============================================================
# 10. 画图
# ============================================================
plt.figure(figsize=(8,5))

plt.plot(
    freqs_plot,
    psd_plot,
    linewidth=2
)

# SO peak
if np.isfinite(so_freq):

    plt.axvline(
        so_freq,
        linestyle="--",
        linewidth=2,
        label=f"SO peak = {so_freq:.2f} Hz"
    )

# ============================================================
# SO band shading
# ============================================================
plt.axvspan(
    0.16,
    1.25,
    alpha=0.2,
    label="SO band"
)

# ============================================================
# 美化
# ============================================================
plt.xlabel("Frequency [Hz]", fontsize=14)
plt.ylabel("PSD", fontsize=14)

plt.title(
    "Power Spectral Density",
    fontsize=18
)

plt.xlim(0, 5)

plt.legend()

plt.tight_layout()

# ============================================================
# 11. 保存
# ============================================================
save_dir = Path("analysis/figures")
save_dir.mkdir(parents=True, exist_ok=True)

save_path = save_dir / "PSD_global_mean.png"

plt.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight"
)

print("saved to:", save_path)

plt.show()