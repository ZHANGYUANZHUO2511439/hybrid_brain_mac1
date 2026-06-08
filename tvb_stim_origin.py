#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TVB-AdEx: state (b_e) × stimulus amplitude 扫描 + 刺激对齐分析
在 PyCharm 中直接运行。需要你已有:
  - tools.init(...), tools.run_simulation(...)
  - Parameter() 参数容器
输出:
  - 每个组合的原始 step_*.npy 到结果目录
  - 以刺激为中心对齐的均值±标准差曲线（PDF）
  - 可选: 相位门控与传播达时图
"""

import os
import sys
import glob
import json
import time
import re
import math
import shutil
import logging
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Tuple
from scipy.signal import find_peaks

import numpy as np
import matplotlib
# matplotlib.use("Agg")  # 后端设为非交互，便于服务器/批量跑
matplotlib.use("Agg")  # 或 macOS 上 'MacOSX'

import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert, welch
from scipy.stats import ttest_rel
#-------替换路径 1.10----------------------------------
import os
import sys

from tvb_adex_ref.config import DATA_DIR

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
TVB_ADEX_REF = os.path.join(PROJECT_DIR, "tvb_adex_ref")

if TVB_ADEX_REF not in sys.path:
    sys.path.insert(0, TVB_ADEX_REF)
#-----------------------------------------------------
from tvb_adex_ref.tvb_model_reference.simulation_file.parameter.parameter_M_Berlin import Parameter
import tvb_adex_ref.tvb_model_reference.src.tools_simulation as tools
from tvb_adex_ref.tvb_model_reference.simulation_file.parameter.parameter_M_Berlin import Parameter

_BASE_PARAM = None  # 模块级缓存

# ------------------------- 实用函数 -------------------------
def _mad(x: np.ndarray, eps: float = 1e-12) -> float:
    """Median Absolute Deviation (robust scale)."""
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(max(mad, eps))

def _psd_ratio(x, fs, band=(0.16, 1.25), total=(0.1, 30.0)):
    # 简洁版 Welch；你也可用已有实现
    from scipy.signal import welch
    f, pxx = welch(x, fs=fs, nperseg=min(len(x), int(fs*8)))
    def band_power(fr, pr, lo, hi):
        m = (fr >= lo) & (fr <= hi)
        return float(np.trapz(pr[m], fr[m])) if np.any(m) else 0.0
    p_band = band_power(f, pxx, band[0], band[1])
    p_tot  = band_power(f, pxx, total[0], total[1])
    return p_band / (p_tot + 1e-12)

def so_bandpower_ratio(
    x: np.ndarray,
    sf: float,
    so_band=(0.5, 1.25),
    total_band=(0.16, 4.0),
    nperseg: int | None = None,
):
    """
    Compute P_SO / P_total using Welch PSD.
    Returns np.nan if scipy.signal.welch is unavailable.
    """
    if welch is None:
        return np.nan

    x = np.asarray(x, dtype=float)
    if nperseg is None:
        # default: ~10 s windows, capped
        nperseg = int(min(len(x), max(256, sf * 10)))

    f, pxx = welch(x, fs=sf, nperseg=nperseg, noverlap=nperseg // 2)
    f = np.asarray(f)
    pxx = np.asarray(pxx)

    def _bandpower(band):
        lo, hi = band
        m = (f >= lo) & (f <= hi)
        if not np.any(m):
            return 0.0
        return float(np.trapz(pxx[m], f[m]))

    p_so = _bandpower(so_band)
    p_tot = _bandpower(total_band)
    if p_tot <= 0:
        return 0.0
    return float(p_so / p_tot)



def get_base_parameters():
    """
    只在第一次调用时真正构造 Parameter() 并设置那些
    永远不变的东西（connectivity 路径、stimulus 关闭等）。
    后续 tvb_sim_single 每次只拷贝一份，不再重复读 zip / 配置。
    """
    global _BASE_PARAM
    if _BASE_PARAM is None:
        p = Parameter()
        p.parameter_connection_between_region['path'] = \
            DATA_DIR

        # 这里可以顺带把“永远不变”的设置好：
        # 例如 stimulus 关闭、monitor 配置、integrator 配置等，
        # 这样后面每次只是改 model/coupling 里的少数几个参数。
        nregions = 68
        p.parameter_stimulus["tau"] = 1e9
        p.parameter_stimulus["T"]   = 1e9
        p.parameter_stimulus["weights"]   = [0.0] * nregions
        p.parameter_stimulus["variables"] = [0]
        p.parameter_stimulus["onset"]     = 1e9
        p.parameter_integrator["dt"]    = 0.1 # ms, 500Hz

        _BASE_PARAM = p
    return _BASE_PARAM

def setup_logger(log_path: Path):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )


def bandpass(x, fs, lo=0.5, hi=4.0, order=2):
    b, a = butter(order, [lo/(fs/2), hi/(fs/2)], btype='band')
    return filtfilt(b, a, x, axis=0)

import numpy as np
from scipy import signal

import numpy as np
from scipy import signal

def _estimate_f_so(x_bp: np.ndarray,
                   fs: float,
                   fmin: float = 0.16,
                   fmax: float = 2.0) -> float:
    x_bp = np.asarray(x_bp, dtype=float)
    if x_bp.size < int(2 * fs):
        return float("nan")

    nperseg = min(len(x_bp), int(8 * fs))
    freqs, psd = signal.welch(
        x_bp,
        fs=fs,
        nperseg=nperseg,
        detrend="constant",
        noverlap=nperseg // 2,
        scaling="density",
    )

    band = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(band):
        return float("nan")

    psd_band = psd[band]
    freqs_band = freqs[band]

    if psd_band.max() <= 0:
        return float("nan")

    idx_peak = np.argmax(psd_band)
    print("freqs_band:", freqs_band[:5])
    print("psd_band:", psd_band[:5])

    return float(freqs_band[idx_peak])

def _estimate_slope_up_simple(x_bp: np.ndarray,
                              fs: float,
                              q: float = 95.0) -> float:
    """
    维持你现在的“正导数 q 分位数”思路，但在 SO 带通后再算。

    slope_up ≈ 上升沿导数的高分位数，单位大致是 “振幅单位/秒”。
    """
    if x_bp.size < 3:
        return float("nan")

    dx = np.diff(x_bp) * fs  # Δx/Δt，Δt=1/fs → 乘 fs 得每秒斜率
    pos_dx = dx[dx > 0]
    if pos_dx.size == 0:
        return float("nan")

    return float(np.percentile(pos_dx, q))


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def list_region_labels(simulator) -> list:
    # TVB Connectivity labels，便于你确认某个解剖区的索引
    try:
        labels = list(simulator.connectivity.region_labels)
    except Exception:
        labels = []
    return labels


def load_concat_steps(folder_path: Path, nstep_guess: int):
    """
    读取该目录下所有 step_*.npy 并按时间拼接。
    如果 nstep_guess 不可靠，就用 glob 动态寻找。
    返回:
        times_l: (nt,)
        rateE_m: (nt, nregions) excitatory firing rate
    """
    # 仅保留 step_<数字>.npy，忽略 step_init.npy 等
    step_pairs = []
    for p in folder_path.glob("step_*.npy"):
        m = re.match(r"^step_(\d+)\.npy$", p.name)
        if m:
            step_pairs.append((int(m.group(1)), p))

    # 如果一个都没匹配到，回退按猜测的 nstep 读取（兼容老输出）
    if not step_pairs:
        for i in range(nstep_guess):
            q = folder_path / f"step_{i}.npy"
            if q.exists():
                step_pairs.append((i, q))

    step_pairs.sort(key=lambda t: t[0])
    files = [p for _, p in step_pairs]
    if not files:
        # 兼容旧逻辑: 用 nstep_guess 顺序加载
        files = [folder_path / f"step_{i}.npy" for i in range(nstep_guess) if (folder_path / f"step_{i}.npy").exists()]

    times_l, rateE_m = [], []
    for fp in files:
        raw = np.load(fp, allow_pickle=True, encoding="latin1")
        # 约定: raw[0][i_time] = [time_ms, [ [E_rates (nregions,)], [I_rates], ... ]]
        for i_time in range(len(raw[0])):
            t_ms = raw[0][i_time][0]
            e_vec = np.array(raw[0][i_time][1][0]).squeeze()  # excitatory across regions
            zz = np.array(raw[0][i_time][1])  # excitatory across regions
            times_l.append(t_ms)
            rateE_m.append(e_vec)

    if len(times_l) == 0:
        return np.array([]), np.array([[]])
    return np.array(times_l), np.vstack(rateE_m)

def load_concat_steps(folder_path: Path, nstep_guess: int):
    """
    读取该目录下所有 step_*.npy 并按时间拼接。
    如果 nstep_guess 不可靠，就用 glob 动态寻找。
    返回:
        times_l: (nt,)
        rateE_m: (nt, nregions) excitatory firing rate
    """
    # 仅保留 step_<数字>.npy，忽略 step_init.npy 等
    step_pairs = []
    for p in folder_path.glob("step_*.npy"):
        m = re.match(r"^step_(\d+)\.npy$", p.name)
        if m:
            step_pairs.append((int(m.group(1)), p))

    # 如果一个都没匹配到，回退按猜测的 nstep 读取（兼容老输出）
    if not step_pairs:
        for i in range(nstep_guess):
            q = folder_path / f"step_{i}.npy"
            if q.exists():
                step_pairs.append((i, q))

    step_pairs.sort(key=lambda t: t[0])
    files = [p for _, p in step_pairs]
    if not files:
        # 兼容旧逻辑: 用 nstep_guess 顺序加载
        files = [folder_path / f"step_{i}.npy" for i in range(nstep_guess) if (folder_path / f"step_{i}.npy").exists()]

    times_l, rateE_m = [], []
    for fp in files:
        raw = np.load(fp, allow_pickle=True, encoding="latin1")
        # 约定: raw[0][i_time] = [time_ms, [ [E_rates (nregions,)], [I_rates], ... ]]
        for i_time in range(len(raw[0])):
            t_ms = raw[0][i_time][0]
            e_vec = np.array(raw[0][i_time][1][0]).squeeze()  # excitatory across regions
            times_l.append(t_ms)
            rateE_m.append(e_vec)

    if len(times_l) == 0:
        return np.array([]), np.array([[]])
    return np.array(times_l), np.vstack(rateE_m)


def load_concat_steps_re_ri_w(folder_path: Path, nstep_guess: int):
    """
    读取该目录下所有 step_*.npy 并按时间拼接。
    如果 nstep_guess 不可靠，就用 glob 动态寻找。
    返回:
        times_l: (nt,)
        rateE_m: (nt, nregions) excitatory firing rate
    """
    # 仅保留 step_<数字>.npy，忽略 step_init.npy 等
    step_pairs = []
    for p in folder_path.glob("step_*.npy"):
        m = re.match(r"^step_(\d+)\.npy$", p.name)
        if m:
            step_pairs.append((int(m.group(1)), p))

    # 如果一个都没匹配到，回退按猜测的 nstep 读取（兼容老输出）
    if not step_pairs:
        for i in range(nstep_guess):
            q = folder_path / f"step_{i}.npy"
            if q.exists():
                step_pairs.append((i, q))

    step_pairs.sort(key=lambda t: t[0])
    files = [p for _, p in step_pairs]
    if not files:
        # 兼容旧逻辑: 用 nstep_guess 顺序加载
        files = [folder_path / f"step_{i}.npy" for i in range(nstep_guess) if (folder_path / f"step_{i}.npy").exists()]

    times_l, rateE_m, rateI_m, W = [], [], [], []
    for fp in files:
        raw = np.load(fp, allow_pickle=True, encoding="latin1")
        # 约定: raw[0][i_time] = [time_ms, [ [E_rates (nregions,)], [I_rates], ... ]]
        for i_time in range(len(raw[0])):
            t_ms = raw[0][i_time][0]
            pops = raw[0][i_time][1]  # list of 8 states

            e_vec = np.array(pops[0]).squeeze()  # excitatory across regions
            i_vec = np.array(pops[1]).squeeze()  # inhibitory firing rate
            w_evec = np.array(pops[5]).squeeze()  # adaptation W_e            times_l.append(t_ms)
            times_l.append(t_ms)
            rateE_m.append(e_vec)
            rateI_m.append(i_vec)
            W.append(w_evec)

    if len(times_l) == 0:
        return np.array([]), np.array([[]]), np.array([[]]),np.array([[]])
    return np.array(times_l), np.vstack(rateE_m), np.vstack(rateI_m), np.vstack(W)

def run_sim_in_memory(parameters, run_sim_ms, region_idx):
    sim = tools.init(
        parameters.parameter_simulation,
        parameters.parameter_model,
        parameters.parameter_connection_between_region,
        parameters.parameter_coupling,
        parameters.parameter_integrator,
        parameters.parameter_monitor,
        parameter_stimulation=parameters.parameter_stimulus,
        my_seed=42,
    )
    time_list = []
    rate_list = []

    for t, data in sim(simulation_length=run_sim_ms):
        if data is None:
            continue
        # 假设 monitor[0] 是你要的 firing rate monitor
        # data[0].shape = (n_time, n_regions, n_states)
        m0 = data[0]
        time_list.append(t.copy())
        rate_list.append(m0[:, region_idx, 0])   # E population

    time_ms = np.concatenate(time_list, axis=0)
    rateE   = np.concatenate(rate_list, axis=0)

    return time_ms, rateE

def align_trials(times_ms, data_mat, onset_ms, t_plot_ms, dt_ms):
    """
    以刺激时刻为中心，裁剪一个 [onset-TP/2, onset+TP/2] 的窗口。
    data_mat: (nt, nregions)
    返回:
        aligned (nw, nregions), idx_start, idx_stop
    """
    mask = (times_ms > onset_ms - t_plot_ms/2.) & (times_ms < onset_ms + t_plot_ms/2.)
    trial_aligned = data_mat[mask, :]
    idx_start = int(max(0.0, t_plot_ms/2. - onset_ms) / dt_ms)
    idx_stop = int((t_plot_ms - max(0.0, onset_ms + t_plot_ms/2. - times_ms[-1])) / dt_ms) + 1
    return trial_aligned, idx_start, idx_stop


def earliest_significant_time(times_ms, trace, baseline_mask, alpha=0.01):
    """
    简单/保守: 用基线段的均值±2SD 定阈，然后查找刺激后首次超过阈值的时间。
    也可以换成 t-test 滑窗判定。trace: (nw,)
    返回: 时间(ms) 或 np.nan
    """
    base = trace[baseline_mask]
    if base.size < 10:
        return np.nan
    mu, sd = np.nanmean(base), np.nanstd(base)
    hi, lo = mu + 2*sd, mu - 2*sd
    # 刺激后区域:
    post_idx0 = np.where(~baseline_mask)[0][0] if np.any(~baseline_mask) else 0
    for k in range(post_idx0, trace.size):
        if trace[k] > hi or trace[k] < lo:
            return times_ms[k]
    return np.nan

def save_waveform_debug(
    t_s,
    muV,
    x,
    x_eeg,
    fs,
    out_dir,
    tag,
):
    """
    保存三联波形图，不 show，只 save
    """
    import os
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(t_s, muV, lw=1, color="black")
    axes[0].set_ylabel("muV (model)")
    axes[0].set_title("Raw membrane potential")

    axes[1].plot(t_s, x, lw=1, color="blue")
    axes[1].set_ylabel("demeaned")

    axes[2].plot(t_s, x_eeg, lw=1, color="red")
    axes[2].set_ylabel("x_eeg (μV)")
    axes[2].set_xlabel("Time (s)")

    fig.suptitle(tag)
    fig.tight_layout()

    png = os.path.join(out_dir, f"wave_{tag}.png")
    fig.savefig(png, dpi=200)
    plt.close(fig)


# ================== 单次仿真 + SO 特征提取 ==================

def tvb_sim_single(
    b_e: float,
    sigma_ou: float,
    g_ee: float,
    sim_dur_s: float = 20.0,
    cut_transient_s: float = 5.0,
    region_idx: int = 5,
    seed: int = 0,
    out_root: str = "tmp_rl_tvb",
    keep = False
) -> tuple[float, float]:

    """
    输入：
        b_e       → excitatory 适应参数 (parameter_model['b_e'])
        sigma_ou  → OU 背景噪声强度 (parameter_model['weight_noise'])
        g_ee      → 区间耦合强度，影响SO同步 (parameter_coupling['a'])

    输出：
        f_SO, slope_up
    """

    # 1) 初始化参数
    # parameters = Parameter()
    # parameters.parameter_connection_between_region['path'] = \
    #     DATA_DIR
    parameters = get_base_parameters()
    pm = parameters.parameter_model
    pc = parameters.parameter_coupling

    # --- baseline 值（用于缩放） ---
    BASE_WEIGHT_NOISE = 1.0  # 写死 or 从 config 读一次
    BASE_COUPLING_A = 1.0  # 同理

    # 2) 设置模型参数（真正作用于 AdEx）
    pm['b_e'] = float(b_e)               # → 适应性（Age/AHI 会影响）
    # pm['weight_noise'] = base_noise * max(float(sigma_ou), 0.0)
    # pc['parameter']['a'] = base_coupling * float(g_ee)
    pm['weight_noise'] = BASE_WEIGHT_NOISE * float(sigma_ou)
    pc['parameter']['a'] = BASE_COUPLING_A * float(g_ee)

    # 3) 仿真输出目录
    run_sim_ms = sim_dur_s * 1000.0
    ### comment this part out if you dont want to record in npy ###
    out_root = Path(out_root); ensure_dir(out_root)
    folder = out_root / f"be{b_e:.2f}_s{sigma_ou:.2f}_gee{g_ee:.2f}_sd{seed}"
    ensure_dir(folder)
    parameters.parameter_simulation['path_result'] = str(folder)
    ### comment this part out if you dont want to record in npy ###

    # 4) 关闭外界刺激
    nregions = 68
    parameters.parameter_stimulus["tau"] = 1e9
    parameters.parameter_stimulus["T"]   = 1e9
    parameters.parameter_stimulus["weights"]   = [0.0] * nregions
    parameters.parameter_stimulus["variables"] = [0]
    parameters.parameter_stimulus["onset"]     = 1e9

    # 5) 初始化并运行仿真
    sim = tools.init(
        parameters.parameter_simulation,
        parameters.parameter_model,
        parameters.parameter_connection_between_region,
        parameters.parameter_coupling,
        parameters.parameter_integrator,
        parameters.parameter_monitor,
        parameter_stimulation=parameters.parameter_stimulus,
        my_seed=seed,
    )
    pm['weight_noise'] = 2e-4
    print("==== DEBUG NOISE ====")
    print("b_e =", pm['b_e'])
    print("weight_noise =", pm['weight_noise'])
    print("sigma_ou =", sigma_ou)
    print("effective noise ≈", pm['weight_noise'] * sigma_ou)
    tools.run_simulation(sim, run_sim_ms,
        parameters.parameter_simulation,
        parameters.parameter_monitor,
    )

    # 6) 读取拼接 firing rate
    nstep_guess = int(sim_dur_s)
    times_ms, rateE = load_concat_steps(folder, nstep_guess=nstep_guess)
    # times_ms, rateE = run_sim_in_memory(parameters, run_sim_ms, region_idx)

    if times_ms.size == 0 or rateE.size == 0:
        return (np.nan, np.nan)

    # 7) 丢掉 transient
    cut_ms = cut_transient_s * 1000.0
    mask = times_ms >= cut_ms
    if not np.any(mask): return (np.nan, np.nan)

    t_use = times_ms[mask]
    sig = rateE[mask, region_idx] * 1e3  # kHz → Hz

    # 8) 采样率
    dt_ms = np.median(np.diff(t_use))
    fs = 1.0 / (dt_ms / 1000.0)

    # 9) f_SO（0.3–2 Hz）
    x = sig - sig.mean()
    freqs = np.fft.rfftfreq(x.size, d=1.0/fs)
    psd   = np.abs(np.fft.rfft(x))**2

    band = (freqs >= 0.3) & (freqs <= 2.0)
    f_so = freqs[band][np.argmax(psd[band])] if np.any(band) else np.nan

    # 10) slope_up = 正导数的95%分位
    dx = np.diff(sig) * fs
    pos_dx = dx[dx > 0]
    slope_up = np.percentile(pos_dx, 95) if pos_dx.size else np.nan

    if not keep:
        shutil.rmtree(folder, ignore_errors=True)

    return float(f_so), float(slope_up)

def tvb_sim_single_eeg(
    b_e: float,
    sigma_ou: float,
    g_ee: float,
    sim_dur_s: float = 25.0,
    cut_transient_s: float = 5.0,
    region_idx: int = 5,
    seed: int = 0,
    out_root: str = "tmp_rl_tvb",
    keep = False,
    save_wave: bool = True,
    wave_dir: str = "wave_debug",
) -> tuple[float, float]:

    """
    输入：
        b_e       → excitatory 适应参数 (parameter_model['b_e'])
        sigma_ou  → OU 背景噪声强度 (parameter_model['weight_noise'])
        g_ee      → 区间耦合强度，影响SO同步 (parameter_coupling['a'])

    输出：
        f_SO, slope_up
    """

    # 1) 初始化参数
    # parameters = Parameter()
    # parameters.parameter_connection_between_region['path'] = \
    #     DATA_DIR
    parameters = get_base_parameters()
    pm = parameters.parameter_model
    pc = parameters.parameter_coupling


    # --- baseline 值（用于缩放） ---
    # base_noise     = pm.get('weight_noise', 0.1) 这种写法会出错，一直沿用上面的pm的数值
    # base_coupling  = 1.0

    BASE_WEIGHT_NOISE = 1e-02  # 写死 or 从 config 读一次
    BASE_COUPLING_A = 1.0  # 同理


    # 2) 设置模型参数（真正作用于 AdEx）
    pm['b_e'] = float(b_e)               # → 适应性（Age/AHI 会影响）
    # pm['weight_noise'] = base_noise * max(float(sigma_ou), 0.0)
    # pc['parameter']['a'] = base_coupling * float(g_ee)
    pm['weight_noise'] = BASE_WEIGHT_NOISE * float(sigma_ou)
    pc['parameter']['a'] = BASE_COUPLING_A * float(g_ee)
    pm["E_L_e"] = -64
    pm["tau_w_i"] = 100  # 抑制群体适应性
    pm["tau_w_e"] = 500  # 兴奋群体适应性
    # 3) 仿真输出目录
    run_sim_ms = sim_dur_s * 1000.0
    ### comment this part out if you dont want to record in npy ###
    out_root = Path(out_root); ensure_dir(out_root)
    folder = out_root / f"be{b_e:.2f}_s{sigma_ou:.2f}_gee{g_ee:.2f}_sd{seed}"
    ensure_dir(folder)
    parameters.parameter_simulation['path_result'] = str(folder)
    ### comment this part out if you dont want to record in npy ###
    # print("pm is: ",pm)
    # 4) 关闭外界刺激
    nregions = 68
    parameters.parameter_stimulus["tau"] = 1e9
    parameters.parameter_stimulus["T"]   = 1e9
    parameters.parameter_stimulus["weights"]   = [0.0] * nregions
    parameters.parameter_stimulus["variables"] = [0]
    parameters.parameter_stimulus["onset"]     = 1e9

    # 5) 初始化并运行仿真
    sim = tools.init(
        parameters.parameter_simulation,
        parameters.parameter_model,
        parameters.parameter_connection_between_region,
        parameters.parameter_coupling,
        parameters.parameter_integrator,
        parameters.parameter_monitor,
        parameter_stimulation=parameters.parameter_stimulus,
        my_seed=seed,
    )
    tools.run_simulation(sim, run_sim_ms,
        parameters.parameter_simulation,
        parameters.parameter_monitor,
    )

    # 6) 读取拼接 firing rate
    nstep_guess = int(sim_dur_s)
    # 6) 读取 All time series
    times_ms, rateE, rateI, W = load_concat_steps_re_ri_w(folder, nstep_guess=nstep_guess)
    # W still in the shape of (4000, 68) which is not right
    if times_ms.size == 0 or rateE.size == 0:
        return (np.nan, np.nan)

    # 7) 丢掉 transient
    cut_ms = cut_transient_s * 1000.0
    mask = times_ms >= cut_ms
    if not np.any(mask):
        return (np.nan, np.nan)

    t_use = times_ms[mask]
    ve = rateE[mask, region_idx]   # excitatory rate
    vi = rateI[mask, region_idx]   # inhibitory rate
    Wt = W[mask, region_idx]       # adaptation

    # 8) fs
    dt_ms = np.median(np.diff(t_use))
    fs = 1000.0 / dt_ms

    # 9) 用 Parameter 里的常数算 μGe, μGi, μV
    pm = parameters.parameter_model
    tau_e = pm["tau_e"]
    tau_i = pm["tau_i"]
    Qe = pm["Q_e"]
    Qi = pm["Q_i"]

    gL = pm["g_L"]
    Ee = pm["E_e"]
    Ei = pm["E_i"]
    EL = pm["E_L_e"]  # excitatory leak potential

    # # ve, vi, Wt 分别来自 pops[0], pops[1], pops[5]
    # e_part = tau_e * Qe * ve * (Ee - EL) / gL #tau in ms, so ve should be in khz
    # i_part =  tau_i * Qi * vi * (Ei - EL) / gL
    # v_part = Wt / gL
    # muV = (
    #         EL
    #         + e_part
    #         + i_part
    #         - v_part
    # )
    # 使用 raw rate (kHz = 1/ms)，不要 *1000
    ve_khz = ve
    vi_khz = vi

    muGe = tau_e * Qe * ve_khz  # ~ effective excitatory conductance
    muGi = tau_i * Qi * vi_khz  # ~ effective inhibitory conductance

    denom = gL + muGe + muGi
    denom = np.maximum(denom, 1e-9)

    muV = (gL * EL + muGe * Ee + muGi * Ei - Wt) / denom

    # 10) 去均值 + SO 带通
    x = muV - muV.mean()
    # 2) 归一化为 EEG proxy
    # x_eeg = (x / np.std(x)) * 30.0  # 30 μV 级 SO
    x_bp = bandpass(x, fs, lo=0.16, hi=4.0)*200

    if save_wave:
        tag = f"be{b_e:.1f}_sig{sigma_ou:.2f}_gee{g_ee:.2f}"
        save_waveform_debug(
            t_s=t_use / 1000.0,
            muV=muV,
            x=x,
            x_eeg=x_bp,
            fs=fs,
            out_dir=wave_dir,
            tag=tag,
        )

    # 11) f_SO & slope_up 都在 x_bp 上算
    # f_so = _estimate_f_so(x_bp, fs, fmin=0.16, fmax=2.0)
    # events = detect_so(x_bp,fs,10,0.8,2.0,x_in_uV=True)
    events, meta = detect_so_(
        x_bp, fs,
        Tmin=0.8, Tmax=2.0,
        x_in_uV=True,  # 如果 x_bp 已是 µV；否则 False
        min_pp_rel=2.0,  # 相对幅度门槛（先从 8 起）
        compute_psd_ratio=True  # 返回窗口级 P_SO/P_total 指标
    )

    # periods = np.array([e["period"] for e in events], float)
    #
    # if len(periods) >= 3:
    #     CV_period = np.std(periods, ddof=1) / np.mean(periods)
    # else:
    #     CV_period = np.nan
    #
    # amps = np.array([e["amp_uv"] for e in events], float)
    # amp=np.median(amps)
    #
    if not keep:
        shutil.rmtree(folder, ignore_errors=True)

    # === 窗口级 gating ===
    psd_ratio = meta.get("psd_ratio", np.nan)
    so_flag = (len(events) >= 3) and (np.isfinite(psd_ratio) and psd_ratio > 0.30)

    if not so_flag:
        CV_period = np.nan
        amp = np.nan
        n_events = 0  # 推荐：直接视为没有 SO
    else:
        periods = np.array([e["period"] for e in events], float)
        CV_period = np.std(periods, ddof=1) / np.mean(periods)

        amps = np.array([e["amp_uv"] for e in events], float)
        amp = float(np.median(amps)) if amps.size else np.nan

        n_events = len(events)

    return float(CV_period), float(amp), int(n_events)

def tvb_sim_single_ve(
    b_e: float,
    sigma_ou: float,
    g_ee: float,
    sim_dur_s: float = 25.0,
    cut_transient_s: float = 5.0,
    region_idx: int = 5,
    seed: int = 0,
    out_root: str = "tmp_rl_tvb",
    keep = False,
    save_wave: bool = True,
    wave_dir: str = "wave_debug",
) -> tuple[float, float]:

    """
    输入：
        b_e       → excitatory 适应参数 (parameter_model['b_e'])
        sigma_ou  → OU 背景噪声强度 (parameter_model['weight_noise'])
        g_ee      → 区间耦合强度，影响SO同步 (parameter_coupling['a'])

    输出：
        f_SO, slope_up
    """

    # 1) 初始化参数
    # parameters = Parameter()
    # parameters.parameter_connection_between_region['path'] = \
    #     DATA_DIR
    parameters = get_base_parameters()
    pm = parameters.parameter_model
    pc = parameters.parameter_coupling


    # --- baseline 值（用于缩放） ---
    # base_noise     = pm.get('weight_noise', 0.1) 这种写法会出错，一直沿用上面的pm的数值
    # base_coupling  = 1.0

    BASE_WEIGHT_NOISE = 3e-04  # 写死 or 从 config 读一次
    BASE_COUPLING_A = 1.0  # 同理


    # 2) 设置模型参数（真正作用于 AdEx）
    pm['b_e'] = float(b_e)               # → 适应性（Age/AHI 会影响）
    # pm['weight_noise'] = base_noise * max(float(sigma_ou), 0.0)
    # pc['parameter']['a'] = base_coupling * float(g_ee)
    pm['weight_noise'] = BASE_WEIGHT_NOISE * float(sigma_ou)
    pc['parameter']['a'] = BASE_COUPLING_A * float(g_ee)
    pm["E_L_e"] = -64
    # pm["tau_w_i"] = 100  # 抑制群体适应性
    pm["tau_w_e"] = 500  # 兴奋群体适应性
    pm["E_L_i"] = -64
    pm["T"] = 20
    # 3) 仿真输出目录
    run_sim_ms = sim_dur_s * 1000.0
    ### comment this part out if you dont want to record in npy ###
    out_root = Path(out_root); ensure_dir(out_root)
    folder = out_root / f"be{b_e:.2f}_s{sigma_ou:.2f}_gee{g_ee:.2f}_sd{seed}"
    ensure_dir(folder)
    parameters.parameter_simulation['path_result'] = str(folder)
    ### comment this part out if you dont want to record in npy ###
    # print("pm is: ",pm)
    # 4) 关闭外界刺激
    nregions = 68
    parameters.parameter_stimulus["tau"] = 1e9
    parameters.parameter_stimulus["T"]   = 1e9
    parameters.parameter_stimulus["weights"]   = [0.0] * nregions
    parameters.parameter_stimulus["variables"] = [0]
    parameters.parameter_stimulus["onset"]     = 1e9

    # 5) 初始化并运行仿真
    sim = tools.init(
        parameters.parameter_simulation,
        parameters.parameter_model,
        parameters.parameter_connection_between_region,
        parameters.parameter_coupling,
        parameters.parameter_integrator,
        parameters.parameter_monitor,
        parameter_stimulation=parameters.parameter_stimulus,
        my_seed=seed,
    )
    tools.run_simulation(sim, run_sim_ms,
        parameters.parameter_simulation,
        parameters.parameter_monitor,
    )
    print("RUN PARAMS:", pm["b_e"], pm["T"], pm["E_L_e"], pm["E_L_i"], pc["parameter"]["a"], pm["weight_noise"])

    # 6) 读取拼接 firing rate
    nstep_guess = int(sim_dur_s)
    # 6) 读取 All time series
    times_ms, rateE, rateI, W = load_concat_steps_re_ri_w(folder, nstep_guess=nstep_guess)
    # W still in the shape of (4000, 68) which is not right
    if times_ms.size == 0 or rateE.size == 0:
        return (np.nan, np.nan)

#zyz修改
    np.save(folder / "rateE.npy", rateE)
    np.save(folder / "time.npy", times_ms)

    print("✅ saved rateE to:", folder)
    # 7) 丢掉 transient
    cut_ms = cut_transient_s * 1000.0
    mask = times_ms >= cut_ms
    if not np.any(mask):
        return (np.nan, np.nan)

    t_use = times_ms[mask]
    ve = rateE[mask, region_idx]   # excitatory rate
    vi = rateI[mask, region_idx]   # inhibitory rate
    Wt = W[mask, region_idx]       # adaptation
    print("b_e=", b_e,
          "ve mean/std=", float(ve.mean()), float(ve.std()),
          "vi mean/std=", float(vi.mean()), float(vi.std()),
          "W  mean/std=", float(Wt.mean()), float(Wt.std()))
    # 8) fs
    dt_ms = np.median(np.diff(t_use))
    fs = 1000.0 / dt_ms

    # 9) 用 Parameter 里的常数算 μGe, μGi, μV
    pm = parameters.parameter_model
    tau_e = pm["tau_e"]
    tau_i = pm["tau_i"]
    Qe = pm["Q_e"]
    Qi = pm["Q_i"]

    gL = pm["g_L"]
    Ee = pm["E_e"]
    Ei = pm["E_i"]
    EL = pm["E_L_e"]  # excitatory leak potential

    # # ve, vi, Wt 分别来自 pops[0], pops[1], pops[5]
    # e_part = tau_e * Qe * ve * (Ee - EL) / gL #tau in ms, so ve should be in khz
    # i_part =  tau_i * Qi * vi * (Ei - EL) / gL
    # v_part = Wt / gL
    # muV = (
    #         EL
    #         + e_part
    #         + i_part
    #         - v_part
    # )
    # 使用 raw rate (kHz = 1/ms)，不要 *1000
    ve_khz = ve
    vi_khz = vi

    muGe = tau_e * Qe * ve_khz  # ~ effective excitatory conductance
    muGi = tau_i * Qi * vi_khz  # ~ effective inhibitory conductance

    denom = gL + muGe + muGi
    denom = np.maximum(denom, 1e-9)

    muV = (gL * EL + muGe * Ee + muGi * Ei - Wt) / denom
    # 1) 去均值（零交叉用）
    x = ve - np.mean(ve)

    # 2) SO 带通（建议 0.16–4 Hz 保留形状；PSD ratio 用 0.16–1.25 Hz）
    x_bp = bandpass(x, fs, lo=0.16, hi=4.0)

    if save_wave:
        tag = f"be{b_e:.1f}_sig{sigma_ou:.2f}_gee{g_ee:.2f}"
        save_waveform_debug(
            t_s=t_use / 1000.0,
            muV=ve,
            x=x,
            x_eeg=x_bp,
            fs=fs,
            out_dir=wave_dir,
            tag=tag,
        )

    # 11) f_SO & slope_up 都在 x_bp 上算
    # f_so = _estimate_f_so(x_bp, fs, fmin=0.16, fmax=2.0)
    # events = detect_so(x_bp,fs,10,0.8,2.0,x_in_uV=True)
    events, meta = detect_so_ve(x_bp, fs, Tmin=0.8, Tmax=2.0, min_pp_rel=6.0)

    if not keep:
        shutil.rmtree(folder, ignore_errors=True)

    periods = np.array([e["period"] for e in events], float)
    CV_period = (np.std(periods, ddof=1) / np.mean(periods)) if len(periods) >= 3 else np.nan

    amps = np.array([e["amp_hz"] for e in events], float)
    amp_hz = float(np.median(amps)) if len(amps) > 0 else np.nan

    # 外层窗口级 gating（建议写死在返回里，避免你到处重复逻辑）
    so_flag = (len(events) >= 3) and (meta.get("psd_ratio", 0.0) > 0.30)

    return {"CV_period": float(CV_period),"amp_hz": float(amp_hz),
            "n_events": int(len(events)),"psd_ratio": float(meta.get("psd_ratio", np.nan)),
            "event_rate": float(meta.get("event_rate", np.nan)),"so_flag": bool(so_flag)
            }

def _zero_crossings(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """返回 downcross(正->负), upcross(负->正) 的索引"""
    x1, x2 = x[:-1], x[1:]
    down = np.where((x1 >= 0) & (x2 < 0))[0]
    up   = np.where((x1 <  0) & (x2 >= 0))[0]
    return down, up

def detect_so(
    x: np.ndarray,
    sf: float,
    min_pp_uv: float,
    Tmin: float,
    Tmax: float,
    *,
    x_in_uV: bool = False,
    min_up_dur: float = 0.25,
    max_up_dur: float = 1.00,
    min_down_dur: float = 0.25,
    max_down_dur: float = 1.00
):
    """
    Slow Oscillation 事件检测（SHHS 对齐版）

    定义：
        一个周期 = 相邻 downcross [d_i, d_{i+1}]
        trough ∈ [d_i, u_i]
        peak   ∈ [u_i, d_{i+1}]
        CV_period 由 period 序列计算
    """

    # === 0) 单位统一：μV ===
    x_uv = x if x_in_uV else x * 1e6

    # === 1) 零交叉 ===
    down, up = _zero_crossings(x_uv)
    events = []

    if len(down) < 2 or len(up) == 0:
        return events

    u_ptr = 0
    n = len(x_uv)

    # === 2) 遍历相邻 downcross 周期 ===
    for i in range(len(down) - 1):
        left, right = down[i], down[i + 1]

        # 找该周期内的 upcross
        while u_ptr < len(up) and up[u_ptr] <= left:
            u_ptr += 1
        if u_ptr >= len(up) or up[u_ptr] >= right:
            continue
        u_i = up[u_ptr]

        # --- 周期约束 ---
        period = (right - left) / sf
        if not (Tmin <= period <= Tmax):
            continue

        # --- trough：负半周最小值 ---
        seg1 = x_uv[left:u_i + 1]
        if seg1.size < 3:
            continue
        trough_rel = np.argmin(seg1)
        trough_idx = left + trough_rel
        trough_val = x_uv[trough_idx]

        # --- peak：正半周最大值 ---
        seg2 = x_uv[u_i:right + 1]
        if seg2.size < 3:
            continue
        peak_rel = np.argmax(seg2)
        peak_idx = u_i + peak_rel
        peak_val = x_uv[peak_idx]

        # --- 振幅阈值 ---
        amp = float(peak_val - trough_val)
        if amp < min_pp_uv:
            continue

        # --- 上升段 ---
        up_dt = (peak_idx - trough_idx) / sf
        if not (min_up_dur <= up_dt <= max_up_dur):
            continue
        slope_up = amp / up_dt

        # --- 下降段 ---
        j = u_ptr + 1
        while j < len(up) and up[j] <= right:
            j += 1

        if j < len(up):
            search_end = up[j]
        else:
            search_end = min(n - 1, right + int(0.5 * period * sf))

        slope_down = np.nan
        if search_end - right >= 3:
            seg3 = x_uv[right:search_end + 1]
            next_trough_rel = np.argmin(seg3)
            next_trough_idx = right + next_trough_rel

            if next_trough_idx > peak_idx:
                down_dt = (next_trough_idx - peak_idx) / sf
                if min_down_dur <= down_dt <= max_down_dur:
                    slope_down = (x_uv[next_trough_idx] - peak_val) / down_dt
                else:
                    d = np.diff(x_uv[peak_idx:right + 1]) * sf
                    slope_down = float(np.min(d)) if d.size > 0 else np.nan

        events.append({
            "t0": left / sf,
            "t1": right / sf,
            "period": float(period),
            "amp_uv": amp,
            "slope_up_uvps": float(slope_up),
            "slope_down_uvps": float(slope_down),
            "trough_t": trough_idx / sf,
            "peak_t": peak_idx / sf,
        })

    return events

def detect_so_(
    x: np.ndarray,
    sf: float,
    Tmin: float,
    Tmax: float,
    *,
    # --- Stage-1 (time/frequency candidate) ---
    x_in_uV: bool = False,
    min_up_dur: float = 0.25,
    max_up_dur: float = 1.00,
    min_down_dur: float = 0.25,
    max_down_dur: float = 1.00,
    # relative amplitude threshold (robust, unit-free)
    min_pp_rel: float = 2.0,
    # robust scale choice: "mad" or "std"
    scale: str = "mad",
    # Optional: frequency gate info attached per call (not per-event)
    compute_psd_ratio: bool = False,
    so_band=(0.5, 1.25),
    total_band=(0.16, 4.0),
):
    """
    Slow Oscillation event detection (two-stage aligned with your pipeline)

    Stage-1 idea:
      - Do NOT threshold in absolute µV here.
      - Use time-structure constraints + relative amplitude (amp / robust_scale)
        to find plausible SO cycles.
      - Optionally compute SO bandpower ratio for window-level gating
        (P_SO / P_total), but do not hard-filter inside this function.

    Definitions (kept consistent with your SHHS-style implementation):
      - A cycle = adjacent downcross [d_i, d_{i+1}]
      - trough ∈ [d_i, u_i]
      - peak   ∈ [u_i, d_{i+1}]

    Inputs:
      x: time series (recommended: already bandpassed, detrended)
      sf: sampling frequency (Hz)
      Tmin/Tmax: period bounds (sec), e.g., 0.8–2.0 for classic SO
      x_in_uV: if True, x already in µV; otherwise interpreted as Volts and converted by *1e6

    Output:
      events: list[dict] with:
        - t0, t1, period
        - amp_uv (if x_in_uV else converted from V->uV; but use amp_rel for gating)
        - amp_rel (unit-free)
        - slope_up_uvps, slope_down_uvps (in uV/s)
        - trough_t, peak_t
      meta: dict with robust_scale and optional psd_ratio
    """
    x = np.asarray(x, dtype=float)
    if x.size < 10:
        return [], {"robust_scale": np.nan, "psd_ratio": np.nan}

    # === 0) Unit unify to µV for slope/amp reporting (still not used as threshold) ===
    x_uv = x if x_in_uV else x * 1e6

    # === 0.5) robust scale for relative amplitude gating ===
    if scale.lower() == "std":
        robust_scale = float(max(np.std(x_uv), 1e-12))
    else:
        robust_scale = _mad(x_uv)

    # Optional window-level frequency gate metric
    psd_ratio = np.nan
    if compute_psd_ratio:
        # note: pass original x (already in same units), ratio is scale-invariant anyway
        psd_ratio = so_bandpower_ratio(x_uv, sf, so_band=so_band, total_band=total_band)

    # === 1) zero crossings ===
    down, up = _zero_crossings(x_uv)
    events = []
    if len(down) < 2 or len(up) == 0:
        return events, {"robust_scale": robust_scale, "psd_ratio": psd_ratio}

    u_ptr = 0
    n = len(x_uv)

    # === 2) traverse adjacent downcross cycles ===
    for i in range(len(down) - 1):
        left, right = int(down[i]), int(down[i + 1])

        # find upcross within (left, right)
        while u_ptr < len(up) and up[u_ptr] <= left:
            u_ptr += 1
        if u_ptr >= len(up) or up[u_ptr] >= right:
            continue
        u_i = int(up[u_ptr])

        # period constraint
        period = (right - left) / float(sf)
        if not (Tmin <= period <= Tmax):
            continue

        # trough in negative half: [left, u_i]
        seg1 = x_uv[left:u_i + 1]
        if seg1.size < 3:
            continue
        trough_rel = int(np.argmin(seg1))
        trough_idx = left + trough_rel
        trough_val = float(x_uv[trough_idx])

        # peak in positive half: [u_i, right]
        seg2 = x_uv[u_i:right + 1]
        if seg2.size < 3:
            continue
        peak_rel = int(np.argmax(seg2))
        peak_idx = u_i + peak_rel
        peak_val = float(x_uv[peak_idx])

        # amplitude (peak-to-trough) in µV
        amp_uv = float(peak_val - trough_val)
        # relative amplitude (unit-free)
        amp_rel = float(amp_uv / robust_scale)

        # --- Stage-1 relative amplitude gate ---
        if amp_rel < float(min_pp_rel):
            continue

        # rise constraint
        up_dt = (peak_idx - trough_idx) / float(sf)
        if not (min_up_dur <= up_dt <= max_up_dur):
            continue
        slope_up = amp_uv / up_dt  # µV/s

        # fall constraint (estimate slope_down towards next trough after right)
        j = u_ptr + 1
        while j < len(up) and up[j] <= right:
            j += 1

        if j < len(up):
            search_end = int(up[j])
        else:
            search_end = int(min(n - 1, right + int(0.5 * period * sf)))

        slope_down = np.nan
        if search_end - right >= 3:
            seg3 = x_uv[right:search_end + 1]
            next_trough_rel = int(np.argmin(seg3))
            next_trough_idx = right + next_trough_rel

            if next_trough_idx > peak_idx:
                down_dt = (next_trough_idx - peak_idx) / float(sf)
                if min_down_dur <= down_dt <= max_down_dur:
                    slope_down = (float(x_uv[next_trough_idx]) - peak_val) / down_dt  # µV/s
                else:
                    # fallback: most negative derivative between peak and right
                    d = np.diff(x_uv[peak_idx:right + 1]) * float(sf)
                    slope_down = float(np.min(d)) if d.size > 0 else np.nan

        events.append({
            "t0": left / float(sf),
            "t1": right / float(sf),
            "period": float(period),
            "amp_uv": amp_uv,
            "amp_rel": amp_rel,
            "slope_up_uvps": float(slope_up),
            "slope_down_uvps": float(slope_down),
            "trough_t": trough_idx / float(sf),
            "peak_t": peak_idx / float(sf),
        })

    meta = {"robust_scale": robust_scale, "psd_ratio": psd_ratio}
    return events, meta
# ------------------------- 主流程 -------------------------
def detect_so_ve(
    ve: np.ndarray,
    fs: float,
    Tmin: float = 0.8,
    Tmax: float = 2.0,
    *,
    min_pp_rel: float = 6.0,       # 相对门槛：多少“稳健尺度”
    min_events: int = 3,
    compute_psd_ratio: bool = True,
    so_band=(0.16, 1.25),
):
    """
    在 ve 上做 SO 事件检测（单位 Hz，不涉及 µV）。
    思路：先在频段内找“像 SO 的周期”，再用相对幅度门槛筛掉噪声。
    返回：events, meta
    """
    # 1) 去均值（零交叉用）
    x = ve - np.mean(ve)

    # 2) SO 带通（建议 0.16–4 Hz 保留形状；PSD ratio 用 0.16–1.25 Hz）
    x_bp = bandpass(x, fs, lo=0.16, hi=4.0)

    # 3) 稳健尺度（避免 std 被极端波形影响）
    mad = np.median(np.abs(x_bp - np.median(x_bp)))
    robust = 1.4826 * mad + 1e-12

    # 4) 零交叉（以 0 为基准）
    s = x_bp
    sign = np.sign(s)
    sign[sign == 0] = 1
    zc = np.where(np.diff(sign) != 0)[0]  # crossing between i and i+1

    if len(zc) < 3:
        return [], {"psd_ratio": 0.0, "event_rate": 0.0, "robust": robust}

    # 5) 以相邻“同向零交叉”定义周期（更稳）：每两个 crossing 形成一个 half-cycle
    #    我们找 trough→peak 的完整周期：用 crossing 序列构造候选 [zc[k], zc[k+2]]
    events = []
    n = len(s)

    for k in range(len(zc) - 2):
        left = zc[k]
        mid  = zc[k + 1]
        right= zc[k + 2]

        period = (right - left) / fs
        if not (Tmin <= period <= Tmax):
            continue

        # 负半周与正半周：根据 s[mid] 的符号不可靠，因此直接用区间内极值
        seg = s[left:right+1]
        if seg.size < 5:
            continue

        trough_rel = int(np.argmin(seg))
        peak_rel   = int(np.argmax(seg))
        trough_idx = left + trough_rel
        peak_idx   = left + peak_rel
        trough_val = float(s[trough_idx])
        peak_val   = float(s[peak_idx])

        # 振幅：peak - trough（Hz）
        amp = peak_val - trough_val
        if amp < (min_pp_rel * robust):
            continue

        # 事件的“方向一致性”约束：要求 trough 在 peak 之前（否则是噪声乱跳）
        if trough_idx >= peak_idx:
            continue

        # 上升时间约束（可选）
        up_dt = (peak_idx - trough_idx) / fs
        if not (0.25 <= up_dt <= 1.2):
            continue

        events.append({
            "t0": left / fs,
            "t1": right / fs,
            "period": float(period),
            "amp_hz": float(amp),
            "trough_t": trough_idx / fs,
            "peak_t": peak_idx / fs,
        })

    # 6) 窗口级 meta
    dur_s = len(ve) / fs
    event_rate = len(events) / (dur_s / 60.0 + 1e-12)  # events per minute

    meta = {
        "robust": robust,
        "event_rate": float(event_rate),
    }
    if compute_psd_ratio:
        meta["psd_ratio"] = float(_psd_ratio(x_bp, fs, band=so_band))

    return events, meta

import numpy as np
from pathlib import Path
def cv_period_from_xbp_shhs_style(
    x_bp: np.ndarray,
    fs: float,
    Tmin: float = 0.8,
    Tmax: float = 2.0,
) -> tuple[float, int]:
    """
    完全复现 SHHS 的 period 定义：
      - period = (down[i+1] - down[i]) / fs
      - 要求存在 upcross：down[i] < up[j] < down[i+1]
      - 只对满足 Tmin/Tmax 的周期纳入 periods
    输入：
      x_bp: demean + bandpass 后的信号（相当于 SHHS 的 x_uv 版本）
      fs: 采样率（Hz）
    输出：
      cv_period, n_periods
    """
    x = np.asarray(x_bp, float)
    n = x.size
    if n < 10:
        return (np.nan, 0)

    # === 1) 找零交叉点（crossing between i and i+1） ===
    sign = np.sign(x)
    sign[sign == 0] = 1.0
    zc = np.where(np.diff(sign) != 0)[0]
    if zc.size < 3:
        return (np.nan, 0)

    # === 2) 分出 down / up crossing ===
    # down: 正 -> 负
    # up:   负 -> 正
    down = zc[(x[zc] > 0) & (x[zc + 1] < 0)]
    up   = zc[(x[zc] < 0) & (x[zc + 1] > 0)]

    if down.size < 2 or up.size < 1:
        return (np.nan, 0)

    # === 3) SHHS-style: 相邻 down 之间找一个 up ===
    periods = []
    u_ptr = 0

    for i in range(len(down) - 1):
        left = int(down[i])
        right = int(down[i + 1])

        # 找该周期中的 upcross：left < u_i < right
        while u_ptr < len(up) and up[u_ptr] <= left:
            u_ptr += 1
        if u_ptr >= len(up) or up[u_ptr] >= right:
            continue

        # 周期约束
        period = (right - left) / float(fs)
        if not (Tmin <= period <= Tmax):
            continue

        periods.append(period)

    if len(periods) < 3:
        return (np.nan, len(periods))

    periods = np.asarray(periods, float)
    cv = float(np.std(periods, ddof=1) / np.mean(periods))
    return (cv, int(periods.size))
def _corr_ignore_nan(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10:
        return np.nan
    x0 = x[m] - np.mean(x[m])
    y0 = y[m] - np.mean(y[m])
    den = (np.std(x0) * np.std(y0))
    if den <= 1e-12:
        return np.nan
    return float(np.mean(x0 * y0) / den)

def _zscore(x: np.ndarray, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    mu = np.mean(x, axis=axis, keepdims=True)
    sd = np.std(x, axis=axis, keepdims=True)
    return (x - mu) / (sd + eps)

def compute_global_synchrony(
    rateE: np.ndarray,
    fs: float,
    lo: float = 0.16,
    hi: float = 4.0,
    use_bandpass: bool = True,
) -> dict:
    """
    rateE: shape (T, R)  (T timepoints, R regions)
    输出：
      - sync_to_global_mean: 每个区与全脑均值信号的相关的平均
      - sync_pairwise_mean: 随机抽样的两两相关均值（避免 O(R^2)）
    """
    X = rateE.copy()

    # demean per region
    X = X - np.mean(X, axis=0, keepdims=True)

    # bandpass (建议用于 SO 同步性)
    if use_bandpass:
        # 你已有 bandpass(x, fs, lo, hi) 是 1D 的，这里逐列调用
        X_bp = np.zeros_like(X)
        for r in range(X.shape[1]):
            X_bp[:, r] = bandpass(X[:, r], fs, lo=lo, hi=hi)
        X = X_bp

    # global mean signal
    g = np.mean(X, axis=1)

    # 1) 与全脑均值的相关（更稳定、更接近“全脑一致性”）
    cors = []
    for r in range(X.shape[1]):
        cors.append(_corr_ignore_nan(X[:, r], g))
    sync_to_global_mean = float(np.nanmean(cors)) if np.any(np.isfinite(cors)) else np.nan

    # 2) 平均两两相关（抽样版，避免 68 区虽然不大但也可省事）
    R = X.shape[1]
    rng = np.random.default_rng(0)
    n_pairs = min(500, R * (R - 1) // 2)
    pairs = set()
    while len(pairs) < n_pairs:
        i = int(rng.integers(0, R))
        j = int(rng.integers(0, R))
        if i == j:
            continue
        if i > j:
            i, j = j, i
        pairs.add((i, j))
    pair_corrs = [_corr_ignore_nan(X[:, i], X[:, j]) for (i, j) in pairs]
    sync_pairwise_mean = float(np.nanmean(pair_corrs)) if np.any(np.isfinite(pair_corrs)) else np.nan

    return {
        "sync_to_global_mean": sync_to_global_mean,
        "sync_pairwise_mean": sync_pairwise_mean,
        "sync_band_lo": lo,
        "sync_band_hi": hi,
        "sync_use_bandpass": bool(use_bandpass),
    }
def poincare_var_zero_cross(x_bp, fs):
    x = np.asarray(x_bp, float)
    x = x[np.isfinite(x)]
    if len(x) < fs * 20:
        return np.nan

    dx = np.diff(x) * fs
    xc = x[:-1]
    xn = x[1:]

    idx = np.where((xc < 0) & (xn >= 0) & (dx > 0))[0]
    if len(idx) < 10:
        return np.nan

    v = dx[idx]
    return float(np.var(v))


def return_std_from_peaks(x_bp, fs, f0_hint=(0.5, 1.5)):
    x = np.asarray(x_bp, float)
    x = x[np.isfinite(x)]
    if len(x) < fs * 20:
        return np.nan

    min_dist = int(fs / f0_hint[1] * 0.6)
    peaks, _ = find_peaks(x, distance=min_dist)
    if len(peaks) < 10:
        return np.nan

    A = x[peaks]
    dA = A[1:] - A[:-1]
    return float(np.std(dA))


def hf_lf_ratio_welch(x, fs, lf=(0.5, 4.0), hf=(12.0, 30.0)):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < fs * 10:
        return np.nan

    f, Pxx = welch(x, fs=fs, nperseg=int(fs * 8), noverlap=int(fs * 4))

    def band_power(b):
        m = (f >= b[0]) & (f <= b[1])
        if not np.any(m):
            return np.nan
        return np.trapz(Pxx[m], f[m])

    p_lf = band_power(lf)
    p_hf = band_power(hf)
    if not np.isfinite(p_lf) or p_lf <= 1e-12:
        return np.nan
    return float(p_hf / p_lf)
def tvb_sim_single_ve_with_sync(

    b_e: float,
    sigma_ou: float,
    g_ee: float,
    sim_dur_s: float = 25.0,
    cut_transient_s: float = 5.0,
    region_idx: int = 5,
    seed: int = 0,
    out_root: str = "tmp_rl_tvb",
    keep: bool = False,
    nregions: int = 68,
    # 同步性参数
    sync_lo: float = 0.16,
    sync_hi: float = 4.0,
    sync_use_bandpass: bool = True,
    dump_params_only: bool = False,
) -> dict:
    print("🔥 ENTER WITH_SYNC FUNCTION")

    """
    扩展版：返回
      - 单区（region_idx）ve/vi/W 的 mean/std
      - 全脑（68区）ve/vi/W 的 mean/std（按区域再汇总）
      - 全脑同步性（在 ve 上计算）
    """

    parameters = get_base_parameters()
    pm = parameters.parameter_model
    pc = parameters.parameter_coupling

    # ===== DEBUG: dump ALL default AdEx parameters (before overwrite) =====
    print("\n=== [DEBUG] parameter_model DEFAULT (before overwrite) ===")
    for k in sorted(pm.keys()):
        print(f"{k} = {pm[k]}")

    # 设定参数
    #原来是BASE_WEIGHT_NOISE = 3e-04
    BASE_WEIGHT_NOISE = 2e-04
    BASE_COUPLING_A = 1.0

    pm["b_e"] = float(b_e)
    pm["weight_noise"] = BASE_WEIGHT_NOISE * float(sigma_ou)
    pc["parameter"]["a"] = BASE_COUPLING_A * float(g_ee)

    # 你现在的工作点（与前面一致）
    pm["E_L_e"] = -64.0
    pm["E_L_i"] = -64.0
    pm["tau_w_e"] = 500
    pm["T"] = 20.0
    pm["a_e"] = 0.0

    # ===== 只输出参数，不跑仿真 =====
    if dump_params_only:
        print("\n=== [DUMP] parameter_model (pm) ===")
        for k in sorted(pm.keys()):
            print(f"{k} = {pm[k]}")

        print("\n=== [DUMP] parameter_coupling (pc) ===")
        print(pc)

        print("\n=== [DUMP] parameter_simulation ===")
        print(parameters.parameter_simulation)

        print("\n=== [DUMP] parameter_integrator ===")
        print(parameters.parameter_integrator)

        return {
            "pm": dict(pm),
            "pc": pc,
            "parameter_simulation": dict(parameters.parameter_simulation),
            "parameter_integrator": dict(parameters.parameter_integrator),
        }

    # ===== 在这里加：打印最终参数（改之后）=====
    print("\n=== [DEBUG] parameter_model FINAL (after overwrite) ===")
    for k in sorted(pm.keys()):
        print(f"{k} = {pm[k]}")

    # 输出目录：建议 keep=False 时清空，避免读旧数据
    run_sim_ms = sim_dur_s * 1000.0
    out_root = Path(out_root); ensure_dir(out_root)
    folder = out_root / f"be{b_e:.2f}_s{sigma_ou:.2f}_gee{g_ee:.2f}_sd{seed}"


    ensure_dir(folder)
    parameters.parameter_simulation["path_result"] = str(folder)
    #zyz修改0413，以前是关闭刺激
    # 关闭刺激
    # parameters.parameter_stimulus["tau"] = 1e9
    # parameters.parameter_stimulus["T"] = 1e9
    # parameters.parameter_stimulus["weights"] = [0.0] * nregions
    # parameters.parameter_stimulus["variables"] = [0]
    # parameters.parameter_stimulus["onset"] = 1e9
    #打开刺激
    parameters.parameter_stimulus["tau"] = 1.0
    parameters.parameter_stimulus["T"] = 1e9
    weights = [0.0] * nregions #zyz修改，设定强度为0.5
    weights[region_idx] = 0.0006
    parameters.parameter_stimulus["weights"] = weights
    parameters.parameter_stimulus["variables"] = [0]
    parameters.parameter_stimulus["onset"] = 5000

    # init & run
    sim = tools.init(
        parameters.parameter_simulation,
        parameters.parameter_model,
        parameters.parameter_connection_between_region,
        parameters.parameter_coupling,
        parameters.parameter_integrator,
        parameters.parameter_monitor,
        parameter_stimulation=parameters.parameter_stimulus,
        my_seed=seed,
    )

    # 强校验：确认参数进入 sim.model（可保留/可注释）
    # print("SIM MODEL b_e:", getattr(sim.model, "b_e", None),
    #       "T:", getattr(sim.model, "T", None),
    #       "E_L_e:", getattr(sim.model, "E_L_e", None),
    #       "E_L_i:", getattr(sim.model, "E_L_i", None))

    tools.run_simulation(sim, run_sim_ms,
        parameters.parameter_simulation,
        parameters.parameter_monitor,
    )

    # ===== 新增：读取时间序列 =====
    nstep_guess = int(sim_dur_s)

    times_ms, rateE, rateI, W = load_concat_steps_re_ri_w(
        folder, nstep_guess=nstep_guess
    )

    print("rateE shape:", rateE.shape)

    # ===== 新增：保存 =====
    np.save(folder / "rateE.npy", rateE)
    np.save(folder / "time.npy", times_ms)

    print("🔥 SAVE EXECUTED:", folder)

    # ===== 防止空数据 =====
    if times_ms.size == 0 or rateE.size == 0:
        return {"status": "empty"}

    return {"status": "ok",

            }
    #zyz修改，加入了tims和rateE，以前都没有
    # ===== DEBUG: 只跑第一次 =====
    #return zyz修改


    # 读取 time series
    nstep_guess = int(sim_dur_s)
    times_ms, rateE, rateI, W = load_concat_steps_re_ri_w(folder, nstep_guess=nstep_guess)

    if times_ms.size == 0 or rateE.size == 0:
        return {"ok": False, "reason": "empty_timeseries"}

    # cut transient
    cut_ms = cut_transient_s * 1000.0
    mask = times_ms >= cut_ms
    if not np.any(mask):
        return {"ok": False, "reason": "no_data_after_cut"}

    t_use = times_ms[mask]
    rateE_use = rateE[mask, :]  # (T,R)
    rateI_use = rateI[mask, :]
    W_use     = W[mask, :]

    # fs
    dt_ms = float(np.median(np.diff(t_use)))
    fs = 1000.0 / dt_ms

    # 单区统计
    ve = rateE_use[:, region_idx]
    vi = rateI_use[:, region_idx]
    Wt = W_use[:, region_idx]

    single_stats = {
        "region_idx": int(region_idx),
        "ve_mean": float(np.mean(ve)), "ve_std": float(np.std(ve)),
        "vi_mean": float(np.mean(vi)), "vi_std": float(np.std(vi)),
        "W_mean":  float(np.mean(Wt)), "W_std":  float(np.std(Wt)),
    }

    # 全脑统计：先算每个区的 mean/std，再在区域维度汇总
    ve_mean_r = np.mean(rateE_use, axis=0)
    ve_std_r  = np.std(rateE_use, axis=0)
    vi_mean_r = np.mean(rateI_use, axis=0)
    vi_std_r  = np.std(rateI_use, axis=0)
    W_mean_r  = np.mean(W_use, axis=0)
    W_std_r   = np.std(W_use, axis=0)

    # 全脑同步性（在 ve 上）
    sync = compute_global_synchrony(
        rateE_use, fs,
        lo=sync_lo, hi=sync_hi,
        use_bandpass=sync_use_bandpass
    )

    # === ve/vi/W 全脑（T,R） ===
    ve_all = rateE[mask, :]  # (T, R)
    vi_all = rateI[mask, :]  # (T, R)
    W_all = W[mask, :]  # (T, R)

    # === 常数 ===
    pm = parameters.parameter_model
    tau_e = float(pm["tau_e"])
    tau_i = float(pm["tau_i"])
    Qe = float(pm["Q_e"])
    Qi = float(pm["Q_i"])
    gL = float(pm["g_L"])
    Ee = float(pm["E_e"])
    Ei = float(pm["E_i"])
    EL = float(pm["E_L_e"])  # 你已设置为 -64

    # === muV 全脑（T,R） ===
    muGe_all = tau_e * Qe * ve_all
    muGi_all = tau_i * Qi * vi_all
    den_all = np.maximum(gL + muGe_all + muGi_all, 1e-9)

    muV_all = (gL * EL + muGe_all * Ee + muGi_all * Ei - W_all) / den_all  # (T,R)

    # === muV_global(t) ===
    muV_global = muV_all.mean(axis=1)  # (T,)

    x = muV_global - np.mean(muV_global)
    x_bp = bandpass(x, fs, lo=0.16, hi=4.0)  # 或与你SHHS一致的频段

    amp_muV_global = float(np.quantile(x_bp, 0.95) - np.quantile(x_bp, 0.05))
    # muV_mean_regions_mean = float(np.mean(muV_global))
    # muV_std_regions_mean = float(np.std(muV_global, ddof=1))
    cv_period_muV_global, n_periods = cv_period_from_xbp_shhs_style(
        x_bp=x_bp, fs=fs, Tmin=0.8, Tmax=2.0
    )
    global_stats = {
        "ve_mean_across_regions_mean": float(np.mean(ve_mean_r)),
        "ve_mean_across_regions_std":  float(np.std(ve_mean_r)),
        "ve_std_across_regions_mean":  float(np.mean(ve_std_r)),
        "ve_std_across_regions_std":   float(np.std(ve_std_r)),

        "vi_mean_across_regions_mean": float(np.mean(vi_mean_r)),
        "vi_mean_across_regions_std":  float(np.std(vi_mean_r)),
        "vi_std_across_regions_mean":  float(np.mean(vi_std_r)),
        "vi_std_across_regions_std":   float(np.std(vi_std_r)),

        "W_mean_across_regions_mean": float(np.mean(W_mean_r)),
        "W_mean_across_regions_std":  float(np.std(W_mean_r)),
        "W_std_across_regions_mean":  float(np.mean(W_std_r)),
        "W_std_across_regions_std":   float(np.std(W_std_r)),

        "amp_muV_global": amp_muV_global,
        "cv_period_muV_global": cv_period_muV_global
    }
    # === 吸引子 / A-B 判据（不保存波形） ===
    poincare_var = poincare_var_zero_cross(x_bp, fs)
    return_std = return_std_from_peaks(x_bp, fs)
    hf_lf_ratio = hf_lf_ratio_welch(muV_global, fs)

    attractor = {
        "poincare_var": float(poincare_var),
        "return_std": float(return_std),
        "hf_lf_ratio": float(hf_lf_ratio),
    }

    if folder.exists() and (not keep):
        import shutil
        shutil.rmtree(folder)

    return {
        "ok": True,
        "params": {
            "b_e": float(b_e),
            "sigma_ou": float(sigma_ou),
            "g_ee": float(g_ee),
            "weight_noise": float(pm["weight_noise"]),
            "coupling_a": float(pc["parameter"]["a"]),
            "E_L_e": float(pm["E_L_e"]),
            "E_L_i": float(pm["E_L_i"]),
            "T": float(pm["T"]),
            "tau_w_e": float(pm["tau_w_e"]),
            "fs": float(fs),
            "dt_ms": float(dt_ms),
            "sim_dur_s": float(sim_dur_s),
            "cut_transient_s": float(cut_transient_s),
        },
        "single_region_stats": single_stats,
        "global_stats": global_stats,
        "synchrony": sync,
        "attractor": attractor,   #新增

    }


def main():
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    #----------1.10---------------------------------------------
    TVB_ADEX_REF = os.path.join(PROJECT_DIR, "tvb_adex_ref")  # 指向 tvb_adex_ref 根目录
    DATA_DIR = os.path.join(TVB_ADEX_REF, "tvb_model_reference", "data", "QL_20120814")
    print("TVB_ADEX_REF =", TVB_ADEX_REF)
    print("DATA_DIR     =", DATA_DIR)
    #---------------------------------------------------------
    parser = argparse.ArgumentParser(description="TVB-AdEx stimulus-aligned analysis")
    parser.add_argument("--out", type=str, default="results/",
                        help="结果根目录")
    parser.add_argument("--run-sim", action="store_true",default=True,
                        help="若加此参数，则实际运行模拟；否则只读已有结果进行分析")
    parser.add_argument("--nregions", type=int, default=68)
    parser.add_argument("--run-sim-ms", type=float, default=4000.0)
    parser.add_argument("--cut-transient-ms", type=float, default=2000.0)
    parser.add_argument("--t-analysis-ms", type=float, default=300.0)
    parser.add_argument("--timestep-ms", type=float, default=0.1)
    parser.add_argument("--nseeds", type=int, default=60)
    parser.add_argument("--bvals", type=float, nargs="+", default=[60]) # default=[0, 20, 40, 60]
    parser.add_argument("--stimvals", type=float, nargs="+", default=[1e-5, 1e-4, 1e-3]) # default=[1e-5, 1e-4, 1e-3]
    parser.add_argument("--stim_tau_ms", type=float, default=50.0)
    parser.add_argument("--stim_region", type=int, default=5, help="默认刺激 region 索引（请用 labels 确认）")
    parser.add_argument("--phase-gating", action="store_true", help="计算UP/DOWN相位门控",default=False)
    parser.add_argument("--propagation-map", action="store_true", help="计算传播达时地图",default=False)
    args = parser.parse_args()

    root = Path(args.out)
    ensure_dir(root)
    setup_logger(root / "run.log")

    # 1) 初始化参数容器
    parameters = Parameter()
    parameters.parameter_connection_between_region['path'] = \
    DATA_DIR
    nregions = args.nregions

    # 2) 生成每个 seed 的随机刺激时刻（保证稳态且前后各留 t_analysis）
    rng = np.random.default_rng(2025)  # 固定种子便于复现
    span = args.run_sim_ms - args.cut_transient_ms - 2*args.t_analysis_ms
    if span <= 0:
        raise ValueError("仿真时长不够放下过渡+分析窗口，请增大 run_sim_ms 或减小 cut_transient/t_analysis")
    stimtime_allseeds = rng.random(args.nseeds) * span + args.t_analysis_ms + args.cut_transient_ms
    np.save(root / "stimtimes.npy", stimtime_allseeds)

    # 3) 先 init 一次以打印 labels，确认 region 索引
    sim0 = tools.init(parameters.parameter_simulation,
                    parameters.parameter_model,
                    parameters.parameter_connection_between_region,
                    parameters.parameter_coupling,
                    parameters.parameter_integrator,
                    parameters.parameter_monitor)
    labels = list_region_labels(sim0)
    if labels:
        logging.info("Region labels loaded. Example (index: name):")
        for i, nm in enumerate(labels[:20]):  # 只打印前20个，完整可自行 print(labels)
            logging.info(f"{i:02d}: {nm}")
    else:
        logging.warning("无法读取 region_labels（如需确认听觉区索引，请检查工具链）")

    # 4) 预分配存放对齐后的矩阵: (b × stim × seeds × time × regions)
    t_plot = 2000.0  # ms，对齐窗口总宽度
    time_bins = int(t_plot / args.timestep_ms)
    means = np.full((len(args.bvals), len(args.stimvals), args.nseeds, time_bins, nregions), np.nan, dtype=float)

    # 5) 主循环：b_e × stim × seeds
    for ib, bval in enumerate(args.bvals):
        for isv, stim_amp in enumerate(args.stimvals):

            # 模型参数更新：状态开关 b_e
            parameters.parameter_model['b_e'] = bval

            # 构造刺激权重向量（只打在指定 region）
            weight = [0.0] * nregions
            weight[args.stim_region] = float(stim_amp)

            parameters.parameter_stimulus["tau"] = float(args.stim_tau_ms)   # 50 ms 脉冲
            parameters.parameter_stimulus["T"] = 1e9                          # 大间隔=只打一次
            parameters.parameter_stimulus["weights"] = weight
            parameters.parameter_stimulus["variables"] = [0]                  # 0=刺激兴奋性发放率通道

            for seedy in range(args.nseeds):
                onset = float(stimtime_allseeds[seedy])

                # 1) 先确定输出目录，并写回仿真参数
                folder = root / f"stim_{stim_amp}_b_{bval}_seed_{seedy}"
                ensure_dir(folder)
                parameters.parameter_simulation['path_result'] = str(folder)

                # 2) 写入“本次刺激”的 onset（必须在 init 之前）
                parameters.parameter_stimulus['onset'] = onset

                if args.run_sim:
                    # 3) init 时把 stimulation & seed 带进去（会读取上面两项）
                    sim = tools.init(parameters.parameter_simulation,
                                   parameters.parameter_model,
                                   parameters.parameter_connection_between_region,
                                   parameters.parameter_coupling,
                                   parameters.parameter_integrator,
                                   parameters.parameter_monitor,
                                   parameter_stimulation=parameters.parameter_stimulus,
                                   my_seed=seedy)

                    # 可选：单独保存一下本 trial 的 onset
                    np.save(folder / "stimtime.npy", np.array([onset]))

                    # 4) 真正运行并写入 step_*.npy 到 folder
                    tools.run_simulation(sim,
                                   args.run_sim_ms,
                                   parameters.parameter_simulation,
                                   parameters.parameter_monitor)

                # 5) 无论是否运行，都从 folder 读取 step_*.npy
                nstep_guess = int(args.run_sim_ms / 1000.0)
                times_ms, rateE = load_concat_steps(folder, nstep_guess=nstep_guess)
                if times_ms.size == 0:
                    logging.warning(f"[empty] {folder} , 可能未运行或无输出文件")
                    continue

                aligned, idx_start, idx_stop = align_trials(times_ms, rateE, onset, t_plot_ms=t_plot, dt_ms=args.timestep_ms)
                # 存入 means（rate 单位换 Hz：很多实现里 rate 是 kHz，所以乘 1e3；如果你输出本来就是 Hz，就去掉）
                means[ib, isv, seedy, idx_start:idx_stop, :] = aligned * 1e3

            # 画刺激区平均±std
            # 画刺激区平均±std
            fig = plt.figure(figsize=(7, 4))
            ax = fig.add_subplot(111)
            ax.set_title(f"Region {args.stim_region} | b_e={bval} | stim={stim_amp}")
            ax.axvline(t_plot / 2, color='k', linestyle=':', lw=1)

            mean_tr = np.nanmean(means[ib, isv, :, :, args.stim_region], axis=0)  # (ntime,)
            std_tr = np.nanstd(means[ib, isv, :, :, args.stim_region], axis=0)  # (ntime,)
            t_axis = np.arange(mean_tr.size) * args.timestep_ms

            # 先画，再让 Matplotlib 自动缩放
            line, = ax.plot(t_axis, mean_tr, lw=2)
            ax.fill_between(t_axis, mean_tr - std_tr, mean_tr + std_tr, alpha=0.3)

            # 只把下限固定在 0，上限按数据自动
            ax.relim();
            ax.autoscale(axis='y', tight=False)
            ylim = ax.get_ylim()
            ax.set_ylim(bottom=0, top=max(ylim[1], float(np.nanmax(mean_tr + std_tr)) * 1.1))

            ax.set_xlabel("Time (ms, aligned to stimulus)")
            ax.set_ylabel("Firing rate (Hz)")
            fig.tight_layout()
            # plt.show(block=True)
            fig.savefig(root / f"sig_aligned_mean_seed{args.nseeds}_b{bval}_stim{stim_amp}_reg{args.stim_region}.pdf")
            plt.close(fig)

    # ---------------- 可选分析 1：相位门控（UP vs DOWN） ----------------
    if args.phase_gating:
        logging.info("Running phase-gating analysis (UP vs DOWN at stimulus)…")
        fs = 1000.0 / args.timestep_ms  # 采样率 Hz
        lo, hi = 0.5, 4.0               # δ 带
        reg = args.stim_region

        for ib, bval in enumerate(args.bvals):
            for isv, stim_amp in enumerate(args.stimvals):
                X = means[ib, isv, :, :, reg]  # (seeds, time)
                valid = ~np.isnan(X).all(axis=1)
                X = X[valid, :]
                if X.size == 0:
                    continue
                # 刺激时刻索引（窗口正中）
                idx0 = int((t_plot/2) / args.timestep_ms)

                # 取相位
                Xf = bandpass(X, fs, lo=lo, hi=hi)
                phi = np.angle(hilbert(Xf, axis=1)[:, idx0])  # 每个 trial 在刺激时刻的相位
                up_mask = np.cos(phi) > 0
                dn_mask = ~up_mask
                if up_mask.sum() < 3 or dn_mask.sum() < 3:
                    logging.warning(f"Not enough trials for UP/DOWN split: b={bval}, stim={stim_amp}")
                    continue

                up_mean = np.nanmean(X[up_mask, :], axis=0)
                dn_mean = np.nanmean(X[dn_mask, :], axis=0)
                t_axis = np.arange(X.shape[1]) * args.timestep_ms

                fig = plt.figure(figsize=(7, 4))
                ax = fig.add_subplot(111)
                ax.set_title(f"Phase gating @ region {reg} | b_e={bval} | stim={stim_amp}")
                ax.axvline(t_plot/2, color='k', linestyle=':', lw=1)
                ax.plot(t_axis, up_mean, label="UP-phase hit", lw=2)
                ax.plot(t_axis, dn_mean, label="DOWN-phase hit", lw=2)
                ax.set_xlabel("Time (ms)")
                ax.set_ylabel("Firing rate (Hz)")
                ax.legend()
                fig.tight_layout()
                fig.savefig(root / f"phase_gating_b{bval}_stim{stim_amp}_reg{reg}.pdf")
                plt.close(fig)

    # ---------------- 可选分析 2：传播达时地图 ----------------
    if args.propagation_map:
        logging.info("Running propagation map (earliest significant time per region)…")
        # 用基线: 刺激前 300 ms
        base_ms = 300.0
        for ib, bval in enumerate(args.bvals):
            for isv, stim_amp in enumerate(args.stimvals):
                # 取跨 seed 均值轨迹 (time × region)
                mean_tr_allreg = np.nanmean(means[ib, isv, :, :, :], axis=0)  # (time, regions)
                if np.isnan(mean_tr_allreg).all():
                    continue
                t_axis = np.arange(mean_tr_allreg.shape[0]) * args.timestep_ms
                baseline_mask = t_axis < (t_plot/2 - base_ms/2)  # 保守一点取前半段

                etimes = []
                for r in range(nregions):
                    et = earliest_significant_time(t_axis, mean_tr_allreg[:, r], baseline_mask)
                    etimes.append(et)
                etimes = np.array(etimes)

                # 简单热力图（用时间，无响应用 NaN）
                fig = plt.figure(figsize=(8, 4))
                ax = fig.add_subplot(111)
                im = ax.imshow(etimes[None, :], aspect="auto", cmap="viridis",
                               extent=[0, nregions, 0, 1], vmin=np.nanmin(etimes), vmax=np.nanmax(etimes))
                ax.set_yticks([])
                ax.set_xlabel("Region index")
                cbar = fig.colorbar(im, ax=ax, shrink=0.7)
                cbar.set_label("Earliest significant time (ms, aligned)")
                ax.set_title(f"Propagation map | b_e={bval} | stim={stim_amp}")
                fig.tight_layout()
                fig.savefig(root / f"propagation_map_b{bval}_stim{stim_amp}.pdf")
                plt.close(fig)

    logging.info("Done.")


if __name__ == "__main__":
    main()
