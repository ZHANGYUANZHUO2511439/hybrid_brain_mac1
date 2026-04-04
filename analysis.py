import re
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt, hilbert, find_peaks

# ---------------- 基础工具 ----------------

def _bandpass_1d(x, fs, lo, hi, order=3):
    b, a = butter(order, [lo/(fs/2), hi/(fs/2)], btype="band")
    return filtfilt(b, a, x)

def _zscore_1d(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / (x.std() + 1e-9)

def _window_mask(times_ms, t0, t1, closed="right"):
    """
    时间窗掩码。默认右闭左开 (t0, t1]，避免与前一窗重叠。
    可选: 'both'->[t0,t1], 'left'->[t0,t1), 'open'->(t0,t1)
    """
    if closed == "both":  return (times_ms >= t0) & (times_ms <= t1)
    if closed == "left":  return (times_ms >= t0) & (times_ms <  t1)
    if closed == "right": return (times_ms >  t0) & (times_ms <= t1)
    return (times_ms >  t0) & (times_ms <  t1)

# ---------------- 读取与统一接口 ----------------

def load_tvb_steps(folder):
    """
    只读取形如 step_<数字>.npy 的文件，按数字排序并拼接。
    返回:
      times:  (T,)       # ms
      signal: (T, 8, 68) # 每时刻 8 个变量 × 68 脑区
    """
    folder = Path(folder)

    # 1) 过滤出严格匹配 step_<数字>.npy 的文件
    pairs = []
    for p in folder.glob("step_*.npy"):
        m = re.match(r"^step_(\d+)\.npy$", p.name)   # 只接受纯数字序号
        if m:
            pairs.append((int(m.group(1)), p))

    if not pairs:
        return np.array([]), np.array([[]])

    # 2) 按序号排序
    pairs.sort(key=lambda t: t[0])
    files = [p for _, p in pairs]

    # 3) 逐块读取并展开
    times_blocks, signal_blocks = [], []
    for fp in files:
        r = np.load(fp, allow_pickle=True).squeeze()
        if not isinstance(r, np.ndarray) or r.size == 0:
            continue
        # 第1列: 时间 (10000,)
        t_block = np.asarray(r[:, 0], dtype=float)
        # 第2列: 每行一个 (8,68,1)
        col = r[:, 1]
        sig_block = np.stack([np.asarray(x) for x in col], axis=0).astype(float)  # (10000,8,68,1)
        sig_block = np.squeeze(sig_block, axis=-1)                                # (10000,8,68)

        times_blocks.append(t_block)
        signal_blocks.append(sig_block)

    if not times_blocks:
        return np.array([]), np.array([[]])

    # 4) 沿时间拼接并按时间排序（防止顺序异常）
    times = np.concatenate(times_blocks, axis=0).astype(float)        # (T,)
    signal = np.concatenate(signal_blocks, axis=0).astype(float)      # (T,8,68)
    order = np.argsort(times)
    times = times[order]
    signal = signal[order, :, :]

    return times, signal

def eeg_proxy_from_signal(signal_T_8_68: np.ndarray, roi_idx: int,
                          mode: str = "EminusI", chE: int = 0, chI: int = 1, zscore: bool = True):
    """
    从 (T,8,68) 信号提取单通道 EEG 代理（ROI = args.stim_region）
    - 默认 'EminusI'：chE - chI （一般 0=E, 1=I）
    - 也可 'channel'：单通道直接取 chE
    """
    X = signal_T_8_68
    if mode == "EminusI":
        eeg = X[:, chE, roi_idx] - X[:, chI, roi_idx]
    elif mode == "channel":
        eeg = X[:, chE, roi_idx]
    else:
        raise ValueError("Unknown mode for eeg proxy")
    return _zscore_1d(eeg) if zscore else np.asarray(eeg, float)

# ---------------- 三项量化（口径与你之前一致） ----------------

def delta_power_change(times_ms, eeg, stim_ms,
                       pre_win=(-500, 0), post_win=(300, 1200),
                       band=(0.5, 2.0), closed="right"):
    """
    Δδ功率：0.5–2 Hz，刺激前 0.5s vs 刺激后 0.3–1.2s
    返回: dP (=post-pre), 细节字典
    """
    times_ms = np.asarray(times_ms, float); eeg = np.asarray(eeg, float)
    if times_ms.size == 0 or eeg.size == 0:
        return np.nan, dict(P_pre=np.nan, P_post=np.nan, note="empty input")
    dt = float(np.median(np.diff(times_ms))); fs = 1000.0/dt

    x = _bandpass_1d(eeg, fs, band[0], band[1])
    pre_mask  = _window_mask(times_ms, stim_ms + pre_win[0],  stim_ms + pre_win[1],  closed=closed)
    post_mask = _window_mask(times_ms, stim_ms + post_win[0], stim_ms + post_win[1], closed=closed)
    if not np.any(pre_mask) or not np.any(post_mask):
        return np.nan, dict(P_pre=np.nan, P_post=np.nan, note="empty window")

    P_pre  = float(np.mean(x[pre_mask]**2))
    P_post = float(np.mean(x[post_mask]**2))
    return (P_post - P_pre), dict(P_pre=P_pre, P_post=P_post)

def itpc_timecourse(times_ms, eeg_trials, stim_ms,
                    post_win=(0, 1500), band=(0.5, 2.0), closed="right"):
    """
    ITPC 时间程：对齐刺激后窗口 (默认 0–1500 ms)，逐点计算 ITPC
    需要：eeg_trials = [eeg_trial1, eeg_trial2, ...]，每条 trial 与同一 times_ms 对齐
    返回：t_rel_ms, itpc
    """
    times_ms = np.asarray(times_ms, float)
    if len(eeg_trials) == 0:
        return np.array([]), np.array([])
    dt = float(np.median(np.diff(times_ms))); fs = 1000.0/dt

    mask = _window_mask(times_ms, stim_ms + post_win[0], stim_ms + post_win[1], closed=closed)
    if not np.any(mask):
        return np.array([]), np.array([])

    phases = []
    for eeg in eeg_trials:
        x = _bandpass_1d(np.asarray(eeg, float), fs, band[0], band[1])
        ph = np.angle(hilbert(x))
        phases.append(ph[mask])
    phases = np.stack(phases, axis=0)              # (n_trials, T_win)
    itpc = np.abs(np.mean(np.exp(1j * phases), axis=0))
    return times_ms[mask] - stim_ms, itpc

def first_so_cycle_features(times_ms, eeg, stim_ms,
                            band=(0.5, 2.0), search_win=(200, 1500),
                            min_peak_dist_ms=300, closed="right"):
    """
    刺激后首个 SO 周期（峰-谷幅与周期）
    返回: amp (峰-谷), period_ms, t_peak1, t_peak2
    """
    times_ms = np.asarray(times_ms, float); eeg = np.asarray(eeg, float)
    if times_ms.size == 0 or eeg.size == 0:
        return np.nan, np.nan, np.nan, np.nan
    dt = float(np.median(np.diff(times_ms))); fs = 1000.0/dt

    x = _bandpass_1d(eeg, fs, band[0], band[1])
    mask = _window_mask(times_ms, stim_ms + search_win[0], stim_ms + search_win[1], closed=closed)
    if not np.any(mask):
        return np.nan, np.nan, np.nan, np.nan

    t_win = times_ms[mask]; x_win = x[mask]
    pk_idx, _ = find_peaks(x_win, distance=max(1, int(min_peak_dist_ms / dt)))
    if pk_idx.size < 2:
        return np.nan, np.nan, np.nan, np.nan

    i1, i2 = pk_idx[0], pk_idx[1]
    t1, t2 = float(t_win[i1]), float(t_win[i2])
    valley = float(np.min(x_win[i1:i2+1]))  # 峰间谷值
    amp = float(x_win[i2] - valley)
    period = t2 - t1
    return amp, period, t1, t2

# ---------------- 一键分析封装（单 run） ----------------

def analyze_single_run(folder: str | Path, stim_ms: float, roi_idx: int,
                       pre_win=(-500, 0), post_win=(300, 1200),
                       itpc_post_win=(0, 1500), band=(0.5, 2.0)):
    """
    读取单个 run（一个 seed 的输出）并在 ROI=roi_idx 上计算三项指标。
    返回: dict 结果 + 基本向量（便于你做图）
    """
    times, signal = load_tvb_steps(folder)
    if times.size == 0:
        return dict(ok=False, note="empty read", folder=str(folder))

    eeg = eeg_proxy_from_signal(signal, roi_idx=roi_idx, mode="EminusI", chE=0, chI=1, zscore=True)

    dP, dP_detail = delta_power_change(times, eeg, stim_ms,
                                       pre_win=pre_win, post_win=post_win, band=band)

    # ITPC 需要多 trial：单 run 先不算（返回空），或自己给一个列表
    t_rel, itpc_tc = np.array([]), np.array([])

    amp, period, t1, t2 = first_so_cycle_features(times, eeg, stim_ms,
                                                  band=band, search_win=(200, 1500), min_peak_dist_ms=300)

    return dict(
        ok=True,
        folder=str(folder),
        roi_idx=int(roi_idx),
        stim_ms=float(stim_ms),
        delta_power=dict(dP=float(dP) if np.isfinite(dP) else np.nan, **dP_detail),
        itpc_timecourse=dict(t_rel_ms=t_rel, itpc=itpc_tc),
        first_cycle=dict(amp=amp, period_ms=period, t_peak1_ms=t1, t_peak2_ms=t2),
        times_ms=times,           # 可用于画图
        eeg=eeg                   # 可用于画图
    )
def itpc_across_runs(folders: list[str | Path], stim_ms: float, roi_idx: int,
                     post_win=(0, 1500), band=(0.5, 2.0)):
    """
    聚合多个 run（多个 seed）的 ROI EEG，计算 ITPC 时间程
    前提：各 run 的采样与时间轴对齐（同 dt，刺激对齐）
    """
    times0, _ = load_tvb_steps(folders[0])
    if times0.size == 0:
        return dict(ok=False, note="empty first run")

    eeg_trials = []
    for f in folders:
        t, sig = load_tvb_steps(f)
        if t.size == 0:  # 跳过空
            continue
        # 简单对齐检查（也可加严谨 resample）
        if t.shape != times0.shape or not np.allclose(np.diff(t), np.diff(times0), rtol=0, atol=1e-6):
            return dict(ok=False, note=f"time axis mismatch in {f}")
        eeg_trials.append(eeg_proxy_from_signal(sig, roi_idx, mode="EminusI", chE=0, chI=1, zscore=True))

    if len(eeg_trials) < 2:
        return dict(ok=False, note="need >=2 trials for ITPC")

    t_rel, itpc_tc = itpc_timecourse(times0, eeg_trials, stim_ms, post_win=post_win, band=band)
    return dict(ok=True, t_rel_ms=t_rel, itpc=itpc_tc)


if __name__ == "__main__":

    # 1) 单次 run 上的三项量化
    folder = "./results/stim_1e-05_b_40_seed_51"
    stim_ms = 10000.0              # 你的刺激绝对时刻（与 times 对齐）
    roi = 5         # 统一用主 ROI

    res1 = analyze_single_run(folder, stim_ms=stim_ms, roi_idx=roi)
    print(res1["delta_power"], res1["first_cycle"])

    # 2) 多 run（多 seed）ITPC
    folders = [
        "./results/stim_1e-05_b_40_seed_51",
        "./results/stim_1e-05_b_40_seed_52",
        "./results/stim_1e-05_b_40_seed_53",
    ]
    res_itpc = itpc_across_runs(folders, stim_ms=stim_ms, roi_idx=roi)
    # res_itpc["t_rel_ms"], res_itpc["itpc"] 可直接作图
