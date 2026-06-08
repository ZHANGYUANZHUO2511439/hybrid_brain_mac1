import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
#在origin基础上升级，直接生成peak times和trough times，跑一次生成两组times，为了加点刺激已经删除了有无有无的刺激模式

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
#---------------------------用于120s后的第一个波峰波谷，后续可删除--------------
def get_stim_times(
        peak_times,
        trough_times,
        start_time=120000.0):
    """
    返回120s后的第一个Peak和第一个Trough
    时间单位与time.npy保持一致(ms)
    """

    peak_stim = peak_times[
        peak_times > start_time
    ][:1]

    trough_stim = trough_times[
        trough_times > start_time
    ][:1]

    return peak_stim, trough_stim
#---------------------------------------------------------


if __name__ == "__main__":

    folder = "/Users/zhangyuanzhuo/PycharmProjects/hybrid_brain_mac1/runs_tvb_be_sigma/baseline_240s_cut_20/sid_0_be_60.00_s_0.500_gee_0.400_sd_1/be60.00_s0.50_gee0.40_sd1"

    peak_times, trough_times = detect_from_file(folder)

    peak_stim, trough_stim = get_stim_times(
        peak_times,
        trough_times,
        start_time=120000.0
    )

    print()
    print("Peak stim:")
    print(peak_stim)

    print()
    print("Trough stim:")
    print(trough_stim)

    np.save(
        folder + "/peak_stim_times.npy",
        peak_stim
    )

    np.save(
        folder + "/trough_stim_times.npy",
        trough_stim
    )



