# detect_phase_and_generate_stim.py
# ==========================================
# 基于 baseline 仿真结果：
# 1. 检测 peak / trough
# 2. 计算 SO 周期
# 3. 自动生成 phase-locked stimulation time
# 4. 保存 stimulus onset 文件
#
# 适用于你当前 pipeline:
#   - time.npy
#   - rateE.npy
#
# 作者：ChatGPT
# ==========================================

import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt, find_peaks


# ============================================================
# 1. 带通滤波
# ============================================================

def bandpass(x, fs, lo=0.16, hi=4.0, order=2):
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype='band')
    return filtfilt(b, a, x)


# ============================================================
# 2. 读取 baseline 数据
# ============================================================

def load_data(folder, region_idx=5):

    t = np.load(os.path.join(folder, "time.npy"))
    rateE = np.load(os.path.join(folder, "rateE.npy"))

    ve = rateE[:, region_idx]

    return t, ve


# ============================================================
# 3. detect peaks / troughs
# ============================================================

def detect_so_phase(t, ve):

    dt = np.median(np.diff(t))
    fs = 1000.0 / dt

    x = ve - np.mean(ve)

    x_bp = bandpass(x, fs)

    # 最小间隔 0.5s
    min_dist = int(fs * 0.5)

    peaks, _ = find_peaks(x_bp, distance=min_dist)

    troughs, _ = find_peaks(-x_bp, distance=min_dist)

    peak_times = t[peaks]
    trough_times = t[troughs]

    return {
        "t": t,
        "x_bp": x_bp,
        "peak_times": peak_times,
        "trough_times": trough_times,
        "fs": fs
    }


# ============================================================
# 4. 计算周期
# ============================================================

def compute_cycle_lengths(trough_times):

    if len(trough_times) < 2:
        return np.array([])

    cycles = np.diff(trough_times)

    return cycles


# ============================================================
# 5. 生成 stimulation onset
# ============================================================

def generate_stim_times(
    trough_times,
    phase_ratio=0.5,
    max_stims=None
):
    """
    phase_ratio:
        0.0 = trough
        0.5 = 中间
        1.0 = next trough

    推荐:
        0.3 ~ 0.7
    """

    stim_times = []

    for i in range(len(trough_times) - 1):

        t0 = trough_times[i]
        t1 = trough_times[i + 1]

        cycle = t1 - t0

        stim_t = t0 + phase_ratio * cycle

        stim_times.append(stim_t)

    stim_times = np.array(stim_times)

    if max_stims is not None:
        stim_times = stim_times[:max_stims]

    return stim_times


# ============================================================
# 6. 保存 stimulation 文件
# ============================================================

def save_stim_times(stim_times, out_file):

    np.save(out_file, stim_times)

    print(f"✅ saved stim times -> {out_file}")

    print("stim times:")
    print(stim_times[:10])


# ============================================================
# 7. 可视化
# ============================================================

def plot_debug(
    t,
    x_bp,
    peak_times,
    trough_times,
    stim_times,
    save_path=None
):

    plt.figure(figsize=(14, 5))

    plt.plot(t, x_bp, lw=1)

    # peaks
    peak_y = np.interp(peak_times, t, x_bp)
    plt.scatter(
        peak_times,
        peak_y,
        s=30,
        label="peaks"
    )

    # troughs
    trough_y = np.interp(trough_times, t, x_bp)
    plt.scatter(
        trough_times,
        trough_y,
        s=30,
        label="troughs"
    )

    # stim
    for s in stim_times:
        plt.axvline(s, linestyle="--")

    plt.xlabel("time (ms)")
    plt.ylabel("bandpassed signal")

    plt.title("SO phase + stimulation timing")

    plt.legend()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)
        print(f"✅ saved figure -> {save_path}")

    plt.show()


# ============================================================
# 8. 主程序
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # baseline folder
    # ========================================================

    folder = (
        "/Users/zhangyuanzhuo/PycharmProjects/hybrid_brain_mac1/runs_tvb_be_sigma/simulation_weight_0/sid_0_be_60.00_s_0.500_gee_0.400_sd_1/be60.00_s0.50_gee0.40_sd1"
    )

    region_idx = 5

    # ========================================================
    # load
    # ========================================================

    t, ve = load_data(folder, region_idx)

    # ========================================================
    # detect
    # ========================================================

    res = detect_so_phase(t, ve)

    peak_times = res["peak_times"]
    trough_times = res["trough_times"]

    print("n peaks:", len(peak_times))
    print("n troughs:", len(trough_times))

    # ========================================================
    # cycles
    # ========================================================

    cycles = compute_cycle_lengths(trough_times)

    if len(cycles) > 0:

        print()
        print("===== SO cycle statistics =====")
        print("mean cycle (ms):", np.mean(cycles))
        print("std cycle (ms):", np.std(cycles))

    # ========================================================
    # generate stimulation timing
    # ========================================================

    stim_times = generate_stim_times(
        trough_times,
        phase_ratio=0.5,   # ⭐ phase
        max_stims=20
    )

    # ========================================================
    # save
    # ========================================================

    save_stim_times(
        stim_times,
        os.path.join(folder, "stim_times.npy")
    )

    # ========================================================
    # debug plot
    # ========================================================

    plot_debug(
        t=res["t"],
        x_bp=res["x_bp"],
        peak_times=peak_times,
        trough_times=trough_times,
        stim_times=stim_times,
        save_path=os.path.join(folder, "stim_debug.png")
    )