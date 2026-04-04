import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

arr = np.load("./result/DEMO_sync/step_4.npy", allow_pickle=True).squeeze()  # list-like of [t, X]

# 组装成 (T,) 的时间 & (T, 8, 68) 的数据
times  = np.array([np.asarray(rec[0]).squeeze() for rec in arr], dtype=float)
signal = np.stack([np.asarray(rec[1]).squeeze() for rec in arr], axis=0)     # (T, 8, 68)

# —— 1) 按时间排序（防乱序）
order  = np.argsort(times)
times  = times[order]
signal = signal[order]

# —— 2) 取 E、I 并做 EEG 代理
region_idx = 18

rateE = signal[:, 0, :] * 1e3   # KHz->Hz（AdEx参考实现是kHz；若已是Hz可去掉这步）
rateI = signal[:, 1, :] * 1e3
eeg_proxy = rateE[:, region_idx] - rateI[:, region_idx]
eeg_proxy = (eeg_proxy - eeg_proxy.mean()) / (eeg_proxy.std() + 1e-9)

plt.figure(figsize=(9,3))
plt.plot(times, eeg_proxy, lw=1)
plt.xlabel("Time (ms)"); plt.ylabel("EEG proxy (z)")
plt.title(f"AdEx (E-I), region {region_idx}")
plt.tight_layout(); plt.show()
#
# # —— 3) 可选：带通 0.5–1.5 Hz（慢波）+ z-score
# def bandpass(x, fs, lo=0.5, hi=1.5, order=3):
#     nyq = 0.5 * fs
#     b, a = butter(order, [lo/nyq, hi/nyq], btype="band")
#     return filtfilt(b, a, x)
#
# fs = 1000.0 / np.median(np.diff(times))   # 采样率Hz（times单位ms）
# eeg_f = bandpass(eeg, fs, lo=0.1, hi=40, order=3)
# eeg_z = (eeg_f) / (eeg_f.std() + 1e-9)
#
# plt.figure(figsize=(10,4))
# plt.plot(times, eeg_z, lw=1)
# plt.xlabel("Time (ms)")
# plt.ylabel("EEG proxy (E−I, z)")
# plt.title(f"EEG proxy (region {region_idx})")
# plt.tight_layout()
# plt.show()
