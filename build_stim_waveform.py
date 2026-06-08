#此脚本用于通过读取baseline的rateE，time，检测并且记录该把刺激加载哪个时间
import numpy as np


# ===== 读取 baseline 输出目录 =====
folder = "/Users/zhangyuanzhuo/PycharmProjects/hybrid_brain_mac1/runs_tvb_be_sigma/baseline_240s_cut_20/sid_0_be_60.00_s_0.500_gee_0.400_sd_1/be60.00_s0.50_gee0.40_sd1"


# ===== 读取 time 和 stim_times =====
time = np.load(folder + "/time.npy")
stim_times = np.load(folder + "/stim_times.npy")


# ===== 时间分辨率 =====
dt_ms = np.median(np.diff(time))

print("dt_ms =", dt_ms)


# ===== stimulation 参数 =====
amp = 0.0009
tau_ms = 5.0


# ===== pulse 宽度 =====
width = int(tau_ms / dt_ms)

print("pulse width index =", width)


# ===== 创建 stimulus waveform =====
stim = np.zeros(len(time))


# ===== 在 stim_times 加 pulse =====
for t in stim_times:

    # 找最接近的时间点
    idx = np.argmin(np.abs(time - t))

    # 加 pulse
    stim[idx:idx + width] = amp


# ===== 保存 =====
np.save(folder + "/stim_waveform.npy", stim)

print()
print("===== STIM WAVEFORM =====")
print("nonzero points =", np.sum(stim > 0))
print("max amp =", np.max(stim))
print("=========================")
print()

print("✅ saved stim_waveform.npy")