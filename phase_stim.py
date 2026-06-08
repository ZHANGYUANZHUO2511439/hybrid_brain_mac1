# phase_stim.py
# ============================================================
# 基于 baseline 仿真结果：
#
# 1. 检测 slow oscillation phase
# 2. 自动生成 stimulation timing
# 3. 实现“有无有无”刺激
# 4. 在上升沿(rising phase)进行 pulse stimulation
# 5. 重新调用 TVB pipeline
#
# 推荐：
#     phase_ratio = 0.25
#
# 含义：
#     trough --> rising --> peak
#                  ↑
#               stimulation
#
# 作者：ChatGPT
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter
from scipy.signal import filtfilt
from scipy.signal import find_peaks

# ============================================================
# !!! 非常重要 !!!
#
# 这里导入你自己的 TVB 仿真函数
#
# 根据你当前 pipeline:
#
# from tvb_stim import tvb_sim_single_ve_with_sync
#
# 如果路径不一样自己修改
# ============================================================

from tvb_stim_copy_5_19 import tvb_sim_single_ve_with_sync


# ============================================================
# 带通滤波
#
# slow oscillation:
#     0.16 ~ 4 Hz
# ============================================================

def bandpass(x, fs, lo=0.16, hi=4.0, order=2):

    b, a = butter(
        order,
        [lo / (fs / 2), hi / (fs / 2)],
        btype='band'
    )

    return filtfilt(b, a, x)


# ============================================================
# 从 baseline folder 读取数据
#
# 你当前 pipeline 已经保存：
#
#     time.npy
#     rateE.npy
#
# 所以可以直接读取
# ============================================================

def load_baseline_data(
    folder,
    region_idx=5
):

    t = np.load(
        os.path.join(folder, "time.npy")
    )

    rateE = np.load(
        os.path.join(folder, "rateE.npy")
    )

    ve = rateE[:, region_idx]

    return t, ve


# ============================================================
# 检测 peaks / troughs
#
# peaks:
#     up-state
#
# troughs:
#     down-state
# ============================================================

def detect_phase(
    t,
    ve
):

    # --------------------------------------------------------
    # sampling frequency
    # --------------------------------------------------------

    dt = np.median(np.diff(t))

    fs = 1000.0 / dt

    # --------------------------------------------------------
    # 去均值
    # --------------------------------------------------------

    x = ve - np.mean(ve)

    # --------------------------------------------------------
    # slow oscillation bandpass
    # --------------------------------------------------------

    x_bp = bandpass(x, fs)

    # --------------------------------------------------------
    # peaks
    #
    # distance:
    #     至少间隔 0.5s
    # --------------------------------------------------------

    min_dist = int(fs * 0.5)

    peaks, _ = find_peaks(
        x_bp,
        distance=min_dist
    )

    # --------------------------------------------------------
    # troughs
    # --------------------------------------------------------

    troughs, _ = find_peaks(
        -x_bp,
        distance=min_dist
    )

    # --------------------------------------------------------
    # 转换成时间(ms)
    # --------------------------------------------------------

    peak_times = t[peaks]

    trough_times = t[troughs]

    print()
    print("===== PHASE DETECTION =====")
    print("n peaks:", len(peak_times))
    print("n troughs:", len(trough_times))

    return {
        "t": t,
        "x_bp": x_bp,
        "peak_times": peak_times,
        "trough_times": trough_times,
        "fs": fs
    }


# ============================================================
# 生成 stimulation timing
#
# phase_ratio:
#
# 0.0:
#     trough
#
# 0.25:
#     rising phase（推荐）
#
# 0.5:
#     cycle middle
#
# 1.0:
#     next trough
# ============================================================

def generate_stim_times(
    trough_times,
    phase_ratio=0.25
):

    stim_times = []

    # --------------------------------------------------------
    # 每个 trough --> next trough
    # 形成一个完整周期
    # --------------------------------------------------------

    for i in range(len(trough_times) - 1):

        t0 = trough_times[i]

        t1 = trough_times[i + 1]

        # ----------------------------------------------------
        # 周期长度
        # ----------------------------------------------------

        cycle = t1 - t0

        # ----------------------------------------------------
        # stimulation timing
        # ----------------------------------------------------

        stim_t = t0 + phase_ratio * cycle

        stim_times.append(stim_t)

    stim_times = np.array(stim_times)

    print()
    print("===== GENERATED STIM TIMES =====")
    print(stim_times[:10])

    return stim_times


# ============================================================
# “有无有无”刺激
#
# 例如：
#
# stim
# no stim
# stim
# no stim
#
# 只保留：
#
# 0 2 4 6 ...
# ============================================================

def select_every_other(
    stim_times
):

    selected = stim_times[::2]

    print()
    print("===== INTERMITTENT STIM =====")
    print("original:", len(stim_times))
    print("selected:", len(selected))

    return selected


# ============================================================
# 可视化
#
# 检查：
#
# 1. peaks
# 2. troughs
# 3. stimulation timing
# ============================================================

def plot_debug(
    t,
    x_bp,
    peak_times,
    trough_times,
    stim_times
):

    plt.figure(figsize=(14, 5))

    # --------------------------------------------------------
    # waveform
    # --------------------------------------------------------

    plt.plot(
        t,
        x_bp,
        lw=1
    )

    # --------------------------------------------------------
    # peaks
    # --------------------------------------------------------

    peak_y = np.interp(
        peak_times,
        t,
        x_bp
    )

    plt.scatter(
        peak_times,
        peak_y,
        s=20,
        label="peaks"
    )

    # --------------------------------------------------------
    # troughs
    # --------------------------------------------------------

    trough_y = np.interp(
        trough_times,
        t,
        x_bp
    )

    plt.scatter(
        trough_times,
        trough_y,
        s=20,
        label="troughs"
    )

    # --------------------------------------------------------
    # stimulation timing
    # --------------------------------------------------------

    for s in stim_times:

        plt.axvline(
            s,
            linestyle="--"
        )

    plt.xlabel("time (ms)")

    plt.ylabel("bandpassed signal")

    plt.title(
        "Phase-Locked Intermittent Stimulation"
    )

    plt.legend()

    plt.tight_layout()

    plt.show()


# ============================================================
# 重新运行 stimulation simulation
#
# 新版逻辑：
#
# 1. 不再：
#       每个 stim_time 单独 rerun
#
# 2. 而是：
#       只取第一个 rising-phase stimulation
#
# 3. stimulation 参数：
#       完全由 tvb_stim.py 控制
#
# 4. phase_stim.py:
#       只负责 timing / pattern
#
# ============================================================

def rerun_with_phase_stim(
    stim_times,
    b_e,
    sigma_ou,
    g_ee,
    seed=1,
    region_idx=5,
    out_root="phase_stim_runs"
):

    # ========================================================
    # 安全检查
    # ========================================================

    if len(stim_times) == 0:

        print()
        print("ERROR:")
        print("No stimulation timing found.")
        print()

        return

    # ========================================================
    # 现在：
    #
    # 只取第一个 stimulation timing
    #
    # 原因：
    #
    # TVB 当前 stimulus system
    # 不支持多个 irregular onset
    #
    # ========================================================

    stim_t = stim_times[0]

    print()
    print("================================================")
    print("PHASE-LOCKED STIMULATION")
    print("================================================")
    print(f"selected stim time = {stim_t:.2f} ms")
    print("================================================")

    # ========================================================
    # 真正调用 TVB pipeline
    #
    # 注意：
    #
    # stimulation strength / tau / onset
    # 全部在 tvb_stim.py 中控制
    #
    # ========================================================

    tvb_sim_single_ve_with_sync(

        # ----------------------------------------------------
        # baseline parameters
        # ----------------------------------------------------

        b_e=b_e,

        sigma_ou=sigma_ou,

        g_ee=g_ee,

        # ----------------------------------------------------
        # stimulation timing
        #
        # phase_stim.py 只负责 timing
        # ----------------------------------------------------

        stim_times=stim_times,

        # ----------------------------------------------------
        # fixed seed
        # ----------------------------------------------------

        seed=seed,

        # ----------------------------------------------------
        # stimulation region
        # ----------------------------------------------------

        region_idx=region_idx,

        # ----------------------------------------------------
        # output
        # ----------------------------------------------------

        out_root=out_root
    )

if __name__ == "__main__":

    # ========================================================
    # baseline folder
    #
    # 这里填 baseline 仿真结果目录
    # ========================================================

    baseline_folder = "/Users/zhangyuanzhuo/PycharmProjects/hybrid_brain_mac1/runs_tvb_be_sigma/simulation_weight_0/sid_0_be_60.00_s_0.500_gee_0.400_sd_1/be60.00_s0.50_gee0.40_sd1"

    # ========================================================
    # load baseline
    # ========================================================

    t, ve = load_baseline_data(
        baseline_folder,
        region_idx=5
    )

    # ========================================================
    # detect phase
    # ========================================================

    res = detect_phase(
        t,
        ve
    )

    # ========================================================
    # generate stimulation timing
    #
    # phase_ratio = 0.25
    #     rising phase
    # ========================================================

    stim_times = generate_stim_times(
        res["trough_times"],
        phase_ratio=0.25
    )

    # ========================================================
    # 有无有无
    # ========================================================

    stim_times = select_every_other(
        stim_times
    )

    print()
    print("===== FINAL STIM TIMES =====")
    print(stim_times[:10])

    print()
    print("stim onset =", stim_times[0])

    # ========================================================
    # visualization
    # ========================================================

    plot_debug(
        t=res["t"],
        x_bp=res["x_bp"],
        peak_times=res["peak_times"],
        trough_times=res["trough_times"],
        stim_times=stim_times
    )

    # ========================================================
    # rerun simulation
    #
    # 这里填 baseline 参数
    # ========================================================

    rerun_with_phase_stim(

        stim_times=stim_times,

        b_e=60.0,

        sigma_ou=0.5,

        g_ee=0.4,

        seed=1,

        region_idx=5,

        out_root="phase_stim_results"
    )