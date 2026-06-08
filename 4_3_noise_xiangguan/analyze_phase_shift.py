#用于对比相位，观察刺激造成的影响
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# 1. 路径（写死）
# ============================================================
base_dir = Path("/Users/zhangyuanzhuo/PycharmProjects/hybrid_brain_mac1/runs_tvb_be_sigma/simulation_weight_0/sid_0_be_60.00_s_0.500_gee_0.400_sd_1/be60.00_s0.50_gee0.40_sd1")
stim_dir = Path("/Users/zhangyuanzhuo/PycharmProjects/hybrid_brain_mac1/runs_tvb_be_sigma/simulation_weight_0.0003/sid_0_be_60.00_s_0.500_gee_0.400_sd_1/be60.00_s0.50_gee0.40_sd1")

# ============================================================
# 2. 加载数据
# ============================================================
def load_data(folder):
    rateE = np.load(folder / "rateE.npy")
    time = np.load(folder / "time.npy") / 1000
    peaks = np.load(folder / "peak_times.npy") / 1000
    return rateE, time, peaks

rate_base, time_base, peaks_base = load_data(base_dir)
rate_stim, time_stim, peaks_stim = load_data(stim_dir)

# ============================================================
# 3. 计算 phase shift
# ============================================================
N = min(len(peaks_base), len(peaks_stim))

pb = peaks_base[:N]
ps = peaks_stim[:N]

delta = ps - pb

print("mean shift:", delta.mean())
print("std shift:", delta.std())

# ============================================================
# 4. 画 histogram
# ============================================================
plt.figure(figsize=(6,4))
plt.hist(delta, bins=30, density=True)

plt.axvline(0, linestyle='--', label="no shift")
plt.axvline(delta.mean(), linestyle='-', label="mean shift")

plt.xlabel("phase shift (s)")
plt.ylabel("density")
plt.title(f"Phase shift (mean={delta.mean():.4f}s)")
plt.legend()

plt.tight_layout()
plt.show()

# ============================================================
# 5. （可选）画 up-down 对比
# ============================================================
t_start = 2
t_end = 10

def crop(time, rate, peaks):
    mask = (time >= t_start) & (time <= t_end)
    return time[mask], rate[mask], peaks[(peaks>=t_start)&(peaks<=t_end)]

tb, rb, pb2 = crop(time_base, rate_base, peaks_base)
ts, rs, ps2 = crop(time_stim, rate_stim, peaks_stim)

plt.figure(figsize=(12,5))

mean_base = rb.mean(axis=1)
mean_stim = rs.mean(axis=1)

plt.plot(tb, mean_base, label="baseline", color="blue")
plt.plot(ts, mean_stim, label="stim", color="red")

plt.scatter(pb2, np.interp(pb2, tb, mean_base), color="blue", s=20)
plt.scatter(ps2, np.interp(ps2, ts, mean_stim), color="red", s=20)

plt.xlabel("time [s]")
plt.ylabel("mean rate")
plt.title("Up-Down with peaks")
plt.legend()

plt.tight_layout()
plt.show()