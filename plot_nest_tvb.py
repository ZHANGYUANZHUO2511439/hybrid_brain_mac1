import os, glob, json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# ===== 小工具 =====
def bandpass(x, fs, lo, hi, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [lo/nyq, hi/nyq], btype="band")
    return filtfilt(b, a, x)

# ===== 1) 读取并拼接所有 step_*.npy =====
result_dir = "./result/DEMO_sync"
paths = sorted(glob.glob(os.path.join(result_dir, "step_[0-9].npy")))
paths = paths[:-1]
assert paths, f"在 {result_dir} 下没找到 step_*.npy"

times_list = []
signals_list = []

for p in paths:
    arr = np.load(p, allow_pickle=True).squeeze()   # list-like of [t, X]
    # t: (Ti,), X: (Ti, n_vars(≈8), n_regions(≈68)) 或 (Ti, 8, 68, 1)
    t_i  = np.array([np.asarray(rec[0]).squeeze() for rec in arr], dtype=float)
    X_i  = np.stack([np.asarray(rec[1]).squeeze() for rec in arr], axis=0)  # (Ti, 8, 68) or (Ti, 8, 68, 1)
    if X_i.ndim == 4 and X_i.shape[-1] == 1:
        X_i = X_i[..., 0]  # -> (Ti, 8, 68)

    times_list.append(t_i)
    signals_list.append(X_i)

# 拼接
times  = np.concatenate(times_list, axis=0)              # (T_total,)
signal = np.concatenate(signals_list, axis=0)            # (T_total, 8, 68)

# 时间排序（防止乱序）
order  = np.argsort(times)
times  = times[order]
signal = signal[order]

# ===== 2) 选择 ROI & 构造 EEG 代理 =====
region_idx = 18  # A1_L (transversetemporal_L). 右侧可用 19
# AdEx 参考实现里 var0/var1 常对应 E/I 率(单位kHz)；若你模型已是 Hz，可去掉 *1e3
rateE = signal[:, 0, :] * 1e3
rateI = signal[:, 1, :] * 1e3
eeg_proxy = rateE[:, region_idx] - rateI[:, region_idx]
eeg_proxy = (eeg_proxy - eeg_proxy.mean()) / (eeg_proxy.std() + 1e-9)

# ===== 3) 读取刺激时窗（可选）=====
# 优先尝试 stim_log.json: {"onsets_ms":[...], "taus_ms":[...]}
stim_onsets, stim_taus = [], []
for cand in ["stim_log.json", "feats_last.json"]:
    jp = os.path.join(result_dir, cand) if not os.path.exists(cand) else cand
    if os.path.exists(jp):
        try:
            with open(jp, "r") as f:
                js = json.load(f)
            if isinstance(js, dict):
                if "onsets_ms" in js and "taus_ms" in js:
                    stim_onsets = list(map(float, js["onsets_ms"]))
                    stim_taus   = list(map(float, js["taus_ms"]))
                    break
                # feats_last.json 兼容: {"A_TC":..,"latency_ms":..,"sigma_ms":..}
                if all(k in js for k in ["latency_ms", "sigma_ms"]):
                    # 如果你当时在 NEST 里用 aud_on=10000.0，可以按需加上基准
                    aud_on = 0.0  # 如果要与 NEST 对齐可改为 10000.0
                    stim_onsets = [aud_on + float(js["latency_ms"])]
                    stim_taus   = [float(js["sigma_ms"])]
                    break
        except Exception:
            pass
aud_on = 2000
# 若没有外部记录，给个占位示例（可删）
if not stim_onsets:
    stim_onsets = [aud_on]
    stim_taus   = [50.0]

# ===== 4) 作图 =====
plt.figure(figsize=(10, 3.6))
plt.plot(times, eeg_proxy, lw=1, label=f"EEG proxy (E−I), ROI {region_idx}")

# 标注刺激窗
for on, tau in zip(stim_onsets, stim_taus if len(stim_taus)==len(stim_onsets) else [0.0]*len(stim_onsets)):
    plt.axvspan(on, on + tau, alpha=0.2, label="stim window" if on == stim_onsets[0] else None)

plt.xlabel("Time (ms)")
plt.ylabel("z-scored")
plt.title("TVB AdEx — EEG proxy at A1 (transversetemporal)")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()

# ===== 5)（可选）滤波看 SO 或 Sigma =====
# fs = 1000.0 / np.median(np.diff(times))   # Hz（times单位ms）
# eeg_so = bandpass(eeg_proxy, fs, lo=0.5, hi=1.5, order=3)
# eeg_sigma = bandpass(eeg_proxy, fs, lo=10, hi=16, order=3)
# plt.figure(figsize=(10,3))
# plt.plot(times, eeg_sigma, lw=1)
# for on, tau in zip(stim_onsets, stim_taus):
#     plt.axvspan(on, on + tau, alpha=0.2)
# plt.xlabel("Time (ms)"); plt.ylabel("Sigma (z)")
# plt.tight_layout(); plt.show()
