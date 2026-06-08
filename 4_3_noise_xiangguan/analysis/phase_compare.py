#用于对比折线图看相位是否发生变化
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# 1. 路径（写死版）
# ============================================================
base_dir = Path("/Users/zhangyuanzhuo/PycharmProjects/hybrid_brain_mac1/runs_tvb_be_sigma/simulation_weight_0/sid_0_be_60.00_s_0.500_gee_0.400_sd_1/be60.00_s0.50_gee0.40_sd1")
stim_dir = Path("/Users/zhangyuanzhuo/PycharmProjects/hybrid_brain_mac1/runs_tvb_be_sigma/point_simulation_weight_0.0009/sid_0_be_60.00_s_0.500_gee_0.400_sd_1/be60.00_s0.50_gee0.40_sd1")

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
# 3. 选窗口（关键：同一个时间段）
# ============================================================
t_start = 2
t_end = 10

def crop(time, rate, peaks):
    mask = (time >= t_start) & (time <= t_end)
    return time[mask], rate[mask], peaks[(peaks>=t_start)&(peaks<=t_end)]

tb, rb, pb = crop(time_base, rate_base, peaks_base)
ts, rs, ps = crop(time_stim, rate_stim, peaks_stim)

# ============================================================
# 4. 画图
# ============================================================
plt.figure(figsize=(12,5))

# --- baseline ---
mean_base = rb.mean(axis=1)
plt.plot(tb, mean_base, label="baseline", color="blue")

# --- stim ---
mean_stim = rs.mean(axis=1)
plt.plot(ts, mean_stim, label="stim", color="red")

# --- 标 peak ---
plt.scatter(pb, np.interp(pb, tb, mean_base),
            color="blue", s=20)

plt.scatter(ps, np.interp(ps, ts, mean_stim),
            color="red", s=20)

plt.xlabel("time [s]")
plt.ylabel("mean rate")
plt.title("Up-Down with peaks (baseline vs stim)")
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# 5. 计算 phase shift（简单版）
# ============================================================
N = min(len(pb), len(ps))

delta = ps[:N] - pb[:N]

print("mean phase shift (s):", delta.mean())
print("first 10 shifts:", delta[:10])