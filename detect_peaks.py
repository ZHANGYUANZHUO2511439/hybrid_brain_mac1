import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


# ===== 带通滤波 =====
def bandpass(x, fs, lo=0.16, hi=4.0, order=2):
    b, a = butter(order, [lo/(fs/2), hi/(fs/2)], btype='band')
    return filtfilt(b, a, x)


# ===== 从文件读取并检测 =====
def detect_from_file(folder, region_idx=5):
    """
    从仿真输出文件中读取数据并检测波峰/波谷

    参数：
    folder: str
        包含 time.npy 和 rateE.npy 的目录
    region_idx: int
        脑区编号（默认第5个）

    返回：
    peak_times, trough_times
    """

    # ===== 读取数据 =====
    try:
        t = np.load(folder + "/time.npy")
        rateE = np.load(folder + "/rateE.npy")
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ 找不到文件，请检查路径: {folder}")

    # ===== 取单脑区 =====
    ve = rateE[:, region_idx]

    # ===== 采样率 =====
    dt = np.median(np.diff(t))
    fs = 1000.0 / dt

    # ===== 去均值 + 带通 =====
    x = ve - np.mean(ve)
    x_bp = bandpass(x, fs)

    # ===== 找波峰 =====
    peaks, _ = find_peaks(x_bp, distance=fs * 0.5)

    # ===== 找波谷 =====
    troughs, _ = find_peaks(-x_bp, distance=fs * 0.5)

    # ===== 转时间 =====
    peak_times = t[peaks]
    trough_times = t[troughs]

    # ===== 输出信息 =====
    print(f"[FILE] peaks: {len(peak_times)}")
    print(f"[FILE] troughs: {len(trough_times)}")

    return peak_times, trough_times
def compute_dt(peak_times, trough_times):
    dt_list = []

    for t_trough in trough_times:
        # 找第一个在它之后的 peak
        future_peaks = peak_times[peak_times > t_trough]
        if len(future_peaks) == 0:
            continue

        dt = future_peaks[0] - t_trough
        dt_list.append(dt)

    return np.array(dt_list)
if __name__ == "__main__":

    folder = "/Users/zhangyuanzhuo/PycharmProjects/hybrid_brain_mac1/runs_tvb_be_sigma/baseline_240s_cut_20/sid_0_be_60.00_s_0.500_gee_0.400_sd_1/be60.00_s0.50_gee0.40_sd1"

    peak_times, trough_times = detect_from_file(folder)

    dt = compute_dt(peak_times, trough_times)

#--------------上升沿前百分之30，对应波峰，上升或下降仅使用一个---------------
    #stim_times = trough_times[:len(dt)] + dt * 0.3
#------------------------------------------------------
#-------------下降沿前百分之30，对应波谷-------------------
    stim_times = peak_times[:-1] + 100
#------------------------------------------------------------------


    # 有无有无
    stim_times = stim_times[::2]

    print()
    print("===== STIM TIMES =====")
    print(stim_times)
    print("======================")
    print()

    np.save(folder + "/stim_times.npy", stim_times)