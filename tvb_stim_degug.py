#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TVB-AdEx: state (b_e) × stimulus amplitude 扫描 + 刺激对齐分析
适配 PyCharm Debug：
- 无命令行参数也可跑（检测 PYCHARM_HOSTED，使用小规模默认配置）
- 修复 step_init.npy 排序异常
- 在 init 前写好 path_result 与 onset
- 读不到数据时跳过绘图
"""

import os
import re
import sys
import json
import time
import math
import glob
import logging
import warnings
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Debug 时建议也用非交互后端，避免 GUI 干扰
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
sys.path.append("/tvb_adex_ref")
from tvb_adex_ref.tvb_model_reference.simulation_file.parameter.parameter_M_Berlin import Parameter
import tvb_adex_ref.tvb_model_reference.src.tools_simulation as tools
from tvb_adex_ref.tvb_model_reference.simulation_file.parameter.parameter_M_Berlin import Parameter
# ---- 屏蔽 TVB 读取空行的提示 ----
warnings.filterwarnings(
    "ignore",
    message="Input line 1 contained no data",
    category=UserWarning,
    module="tvb.basic.readers"
)


# ===================== 工具函数 =====================

def setup_logger(log_path: Path):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def bandpass(x, fs, lo=0.5, hi=4.0, order=2):
    b, a = butter(order, [lo/(fs/2), hi/(fs/2)], btype='band')
    return filtfilt(b, a, x, axis=0)

def list_region_labels(simulator):
    try:
        return list(simulator.connectivity.region_labels)
    except Exception:
        return []

def load_concat_steps(folder_path: Path, nstep_guess: int):
    """
    只加载形如 step_<number>.npy 的文件，忽略 step_init.npy 等。
    返回 times_l: (nt,), rateE_m: (nt, nregions)
    """
    all_files = list(folder_path.glob("step_*.npy"))

    step_num_files = []
    for fp in all_files:
        m = re.match(r"^step_(\d+)\.npy$", fp.name)
        if m:
            step_num_files.append((int(m.group(1)), fp))
        # 其他文件（如 step_init.npy）忽略

    if not step_num_files:
        # 回退：如果没有编号文件，尝试按猜测的 nstep 读取
        fallback = []
        for i in range(nstep_guess):
            f = folder_path / f"step_{i}.npy"
            if f.exists():
                fallback.append((i, f))
        step_num_files = fallback

    step_num_files.sort(key=lambda t: t[0])
    files = [fp for _, fp in step_num_files]

    print(f"[loader] {folder_path.name} -> found {len(files)} step files:",
          [f.name for f in files[:5]], "..." if len(files) > 5 else "")

    times_l, rateE_m = [], []
    for fp in files:
        raw = np.load(fp, allow_pickle=True, encoding="latin1")
        # 建议在此处打断点，展开 raw 看具体结构
        for i_time in range(len(raw[0])):
            t_ms = raw[0][i_time][0]
            e_vec = np.array(raw[0][i_time][1][0])  # excitatory across regions
            times_l.append(t_ms)
            rateE_m.append(e_vec)

    if len(times_l) == 0:
        return np.array([]), np.array([[]])
    return np.array(times_l), np.vstack(rateE_m)

def align_trials(times_ms, data_mat, onset_ms, t_plot_ms, dt_ms):
    """
    以刺激时刻为中心裁剪窗口 [onset-TP/2, onset+TP/2]。
    """
    mask = (times_ms > onset_ms - t_plot_ms/2.) & (times_ms < onset_ms + t_plot_ms/2.)
    trial_aligned = data_mat[mask, :]
    idx_start = int(max(0.0, t_plot_ms/2. - onset_ms) / dt_ms)
    idx_stop = int((t_plot_ms - max(0.0, onset_ms + t_plot_ms/2. - times_ms[-1])) / dt_ms) + 1
    return trial_aligned, idx_start, idx_stop


# ===================== 参数获取（PyCharm 友好） =====================

def get_args():
    parser = argparse.ArgumentParser(description="TVB-AdEx stimulus-aligned analysis (PyCharm-friendly)")
    parser.add_argument("--out", type=str, default=None, help="结果根目录")
    parser.add_argument("--run-sim", action="store_true", help="实际运行仿真，否则只读已有结果")
    parser.add_argument("--nregions", type=int, default=68)
    parser.add_argument("--run-sim-ms", type=float, default=4000.0)
    parser.add_argument("--cut-transient-ms", type=float, default=2000.0)
    parser.add_argument("--t-analysis-ms", type=float, default=300.0)
    parser.add_argument("--timestep-ms", type=float, default=0.1)
    parser.add_argument("--nseeds", type=int, default=None)
    parser.add_argument("--bvals", type=float, nargs="+", default=None)
    parser.add_argument("--stimvals", type=float, nargs="+", default=None)
    parser.add_argument("--stim_tau_ms", type=float, default=50.0)
    parser.add_argument("--stim_region", type=int, default=5)
    parser.add_argument("--phase-gating", action="store_true", help="计算UP/DOWN相位门控")
    parser.add_argument("--propagation-map", action="store_true", help="计算传播达时地图")

    args = parser.parse_args()

    # 如果跑在 PyCharm（常见环境变量），且没有传入参数，则给一套小规模默认，方便 F5 直接调试
    in_pycharm = os.environ.get("PYCHARM_HOSTED") == "1"

    if args.out is None:
        args.out = "results_debug" if in_pycharm else "results"

    if args.nseeds is None:
        args.nseeds = 1 if in_pycharm else 40

    if args.bvals is None:
        args.bvals = [0, 60] if in_pycharm else [0, 20, 40, 60]

    if args.stimvals is None:
        args.stimvals = [1e-4] if in_pycharm else [1e-5, 1e-4, 1e-3]

    # 在 PyCharm 下默认也运行仿真，避免“没有数据可读”
    if in_pycharm and not args.run_sim:
        args.run_sim = True

    return args


# ===================== 主流程 =====================

def main():
    args = get_args()

    root = Path(args.out)
    ensure_dir(root)
    setup_logger(root / "run.log")

    # ---- 1) 初始化参数容器 ----
    parameters = Parameter()
    nregions = args.nregions

    # ---- 2) 生成随机刺激时刻（每个 seed 不同），保证有稳态与分析余量 ----
    rng = np.random.default_rng(2025)  # 可改以改变随机性
    span = args.run_sim_ms - args.cut_transient_ms - 2*args.t_analysis_ms
    if span <= 0:
        raise ValueError("仿真时长不够放下过渡+分析窗口，请增大 run_sim_ms 或减小 cut_transient/t_analysis")
    stimtime_allseeds = rng.random(args.nseeds) * span + args.t_analysis_ms + args.cut_transient_ms
    np.save(root / "stimtimes.npy", stimtime_allseeds)

    # ---- 3) init 一次，只为获取 labels（建议在下方打断点看 labels 顺序）----
    sim0 = tools.init(parameters.parameter_simulation,
                    parameters.parameter_model,
                    parameters.parameter_connection_between_region,
                    parameters.parameter_coupling,
                    parameters.parameter_integrator,
                    parameters.parameter_monitor)
    labels = list_region_labels(sim0)
    if labels:
        logging.info(f"Loaded {len(labels)} region labels. First 10: " + ", ".join([f"{i}:{nm}" for i, nm in enumerate(labels[:10])]))
    else:
        logging.warning("无法读取 region_labels（如需确认听觉区索引，请检查工具链）")

    # ---- 4) 预分配对齐矩阵 ----
    t_plot = 2000.0  # ms
    time_bins = int(t_plot / args.timestep_ms)
    means = np.full((len(args.bvals), len(args.stimvals), args.nseeds, time_bins, nregions), np.nan, dtype=float)

    # ---- 5) 主循环：b_e × stim × seed ----
    for ib, bval in enumerate(args.bvals):
        # 全脑状态（睡/醒旋钮）
        parameters.parameter_model['b_e'] = float(bval)

        for isv, stim_amp in enumerate(args.stimvals):
            # 刺激配置
            weight = [0.0] * nregions
            weight[args.stim_region] = float(stim_amp)

            parameters.parameter_stimulus["tau"] = float(args.stim_tau_ms)   # 50 ms 脉冲
            parameters.parameter_stimulus["T"] = 1e9                          # 大间隔=只打一次
            parameters.parameter_stimulus["weights"] = weight
            parameters.parameter_stimulus["variables"] = [0]                  # 0=兴奋性发放率通道

            for seedy in range(args.nseeds):
                onset = float(stimtime_allseeds[seedy])

                # ---- 输出目录与参数写回（顺序很重要！建议在这里打断点检查）----
                folder = root / f"stim_{stim_amp}_b_{bval}_seed_{seedy}"
                ensure_dir(folder)
                parameters.parameter_simulation['path_result'] = str(folder)
                parameters.parameter_stimulus['onset'] = onset

                if args.run_sim:
                    sim = tools.init(parameters.parameter_simulation,
                                   parameters.parameter_model,
                                   parameters.parameter_connection_between_region,
                                   parameters.parameter_coupling,
                                   parameters.parameter_integrator,
                                   parameters.parameter_monitor,
                                   parameter_stimulation=parameters.parameter_stimulus,
                                   my_seed=seedy)

                    # 可选：保存本 trial 的 onset
                    np.save(folder / "stimtime.npy", np.array([onset]))

                    # 真正运行（仿真时会看到 console 输出 simulation time : 1000, 2000...）
                    tools.run_simulation(sim,
                                   args.run_sim_ms,
                                   parameters.parameter_simulation,
                                   parameters.parameter_monitor)

                # ---- 读取 step_*.npy 并对齐 ----
                nstep_guess = int(args.run_sim_ms / 1000.0)
                times_ms, rateE = load_concat_steps(folder, nstep_guess=nstep_guess)

                if times_ms.size == 0 or rateE.size == 0:
                    logging.warning(f"[empty-after-run] {folder} 没有有效 step 文件，跳过此 trial")
                    continue

                aligned, idx_start, idx_stop = align_trials(times_ms, rateE, onset, t_plot_ms=t_plot, dt_ms=args.timestep_ms)
                # 注意：有些实现 rate 单位是 kHz，这里统一乘 1e3 变 Hz；若你的输出本来就是 Hz，就去掉 *1e3
                means[ib, isv, seedy, idx_start:idx_stop, :] = aligned * 1e3

            # ---- 刺激区均值±std 绘图（仅当有有效数据时）----
            stim_reg = args.stim_region
            valid_trials = ~np.isnan(means[ib, isv, :, :, stim_reg]).all(axis=1)
            if not np.any(valid_trials):
                logging.warning(f"[plot-skip] b_e={bval}, stim={stim_amp}: 无有效 trial，跳过绘图")
            else:
                m = np.nanmean(means[ib, isv, valid_trials, :, stim_reg], axis=0)
                s = np.nanstd(means[ib, isv, valid_trials, :, stim_reg], axis=0)
                t_axis = np.arange(m.size) * args.timestep_ms

                fig = plt.figure(figsize=(7, 4))
                ax = fig.add_subplot(111)
                ax.set_title(f"Region {stim_reg} | b_e={bval} | stim={stim_amp} | n={valid_trials.sum()}")
                ax.axvline(t_plot/2, color='k', linestyle=':', lw=1)
                ax.plot(t_axis, m, lw=2)
                ax.fill_between(t_axis, m - s, m + s, alpha=0.3)
                ax.set_xlabel("Time (ms, aligned to stimulus)")
                ax.set_ylabel("Firing rate (Hz)")
                ax.set_ylim(bottom=0)
                fig.tight_layout()
                fig.savefig(root / f"sig_aligned_mean_seed{valid_trials.sum()}_b{bval}_stim{stim_amp}_reg{stim_reg}.pdf")
                plt.close(fig)

    logging.info("Done.")


if __name__ == "__main__":
    main()
