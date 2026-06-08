# analyze_post_stim.py

#如果：post > baseline，说明：刺激增强了 slow oscillation。
#如果：post > baseline，说明：SO 更连续、更稳定。
#PSD：如果刺激有效：应该看到：0.5~1.25Hz power increase
#dominant frequency可以观察刺激后：是否更接近 1Hz，是否更稳定
import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt, welch


# =========================================================
# bandpass
# =========================================================

def bandpass(x, fs, lo=0.16, hi=4.0, order=2):

    b, a = butter(order, [lo/(fs/2), hi/(fs/2)], btype='band')

    return filtfilt(b, a, x)


# =========================================================
# continuity
# =========================================================

def continuity_index(x):

    env = np.abs(x)

    th = np.percentile(env, 70)

    active = env > th

    return np.mean(active)


# =========================================================
# bandpower ratio
# =========================================================

def bandpower_ratio(
    f,
    pxx,
    so_band=(0.5, 1.25),
    total_band=(0.16, 4.0)
):

    so_mask = (f >= so_band[0]) & (f <= so_band[1])

    total_mask = (
        (f >= total_band[0]) &
        (f <= total_band[1])
    )

    p_so = np.trapz(
        pxx[so_mask],
        f[so_mask]
    )

    p_total = np.trapz(
        pxx[total_mask],
        f[total_mask]
    )

    return p_so / (p_total + 1e-12)


# =========================================================
# settings
# =========================================================

folder = "/Users/zhangyuanzhuo/PycharmProjects/hybrid_brain_mac1/runs_tvb_be_sigma/point_simulation_120s_upside/sid_0_be_60.00_s_0.500_gee_0.400_sd_1/be60.00_s0.50_gee0.40_sd1"
region_idx = 5

stim_time = 120000  # ms


# =========================================================
# load
# =========================================================

t = np.load(
    os.path.join(folder, "time.npy")
)

rateE = np.load(
    os.path.join(folder, "rateE.npy")
)

ve = rateE[:, region_idx]


# =========================================================
# fs
# =========================================================

dt = np.median(np.diff(t))

fs = 1000.0 / dt


# =========================================================
# preprocess
# =========================================================

x = ve - np.mean(ve)

x_bp = bandpass(x, fs)


# =========================================================
# split baseline / post-stim
# =========================================================

mask_base = t < stim_time

mask_post = t >= stim_time

x_base = x_bp[mask_base]

x_post = x_bp[mask_post]


# =========================================================
# amplitude
# =========================================================

amp_base = np.std(x_base)

amp_post = np.std(x_post)

print()
print("===== AMPLITUDE =====")
print("baseline std:", amp_base)
print("post stim std:", amp_post)


# =========================================================
# continuity
# =========================================================

cont_base = continuity_index(x_base)

cont_post = continuity_index(x_post)

print()
print("===== CONTINUITY =====")
print("baseline:", cont_base)
print("post stim:", cont_post)


# =========================================================
# PSD
# =========================================================

f_base, p_base = welch(
    x_base,
    fs=fs,
    nperseg=min(
        len(x_base),
        int(fs * 8)
    )
)

f_post, p_post = welch(
    x_post,
    fs=fs,
    nperseg=min(
        len(x_post),
        int(fs * 8)
    )
)


# =========================================================
# dominant frequency
# =========================================================

band_base = (
    (f_base >= 0.16) &
    (f_base <= 4.0)
)

band_post = (
    (f_post >= 0.16) &
    (f_post <= 4.0)
)

f_dom_base = f_base[band_base][
    np.argmax(p_base[band_base])
]

f_dom_post = f_post[band_post][
    np.argmax(p_post[band_post])
]

print()
print("===== DOMINANT FREQUENCY =====")
print("baseline:", f_dom_base)
print("post stim:", f_dom_post)


# =========================================================
# SO power ratio
# =========================================================

ratio_base = bandpower_ratio(
    f_base,
    p_base
)

ratio_post = bandpower_ratio(
    f_post,
    p_post
)

print()
print("===== SO POWER RATIO =====")
print("baseline:", ratio_base)
print("post stim:", ratio_post)


# =========================================================
# waveform plot
# =========================================================

plt.figure(figsize=(14, 5))

plt.plot(
    t,
    x_bp,
    lw=1
)

plt.axvline(
    stim_time,
    linestyle="--",
    label="stim"
)

plt.xlabel("time (ms)")

plt.ylabel("bandpassed")

plt.title("SO waveform")

plt.legend()

plt.tight_layout()

plt.show()


# =========================================================
# PSD plot
# =========================================================

plt.figure(figsize=(7, 5))

plt.semilogy(
    f_base,
    p_base,
    label="baseline"
)

plt.semilogy(
    f_post,
    p_post,
    label="post-stim"
)

plt.xlim(0, 5)

plt.xlabel("frequency (Hz)")

plt.ylabel("PSD")

plt.title("Power Spectrum")

plt.legend()

plt.tight_layout()

plt.show()