# -*- coding: utf-8 -*-
#此副本用于保存，下一步正在尝试在第60秒作为第一个波加刺激
"""
TVB 参数扫描 + BO 标定 + 可视化

假设同一文件中已经有：
    - dataclass Subject
    - load_subjects_from_csv
    - split_subjectsF
    - 函数 tvb_sim_single(b_e, sigma_ou, g_ee, ...)

建议用途：
1. 对单个 subject 扫描 (sigma, g_ee) → (f, slope) 的地形；
2. 对多个 subject 用 BO 做 per-subject 标定 (sigma*, g_ee*)；
3. 画出 heatmap，看清楚在哪一块参数区域有合理的 SO 行为。
"""

import os
import pandas as pd
#zyz修改，删除绘制csv文件部分后脚本之间没有连接起来，此import用于连接，如果要绘制csv文件需要删掉这部分import，前半段panda也需要删除

from detect_peaks import detect_from_file, compute_dt
from tvb_stim_copy_5_19 import tvb_sim_single_ve_with_sync


def sweep_be_sigma_for_subject(
    subj: str,
    gee_fixed: float = 0.4,
    be_range=(59.0, 60.0),
    sigma_range=(0.5, 0.6),
    num_be: int = 2,
    num_sigma: int = 2,
    sim_dur_s: float = 20.0,
    cut_transient_s: float = 5.0,
    region_idx: int = 5,
    seed: int = 0,
    out_root: str = "runs_scan_be_sigma",
) -> pd.DataFrame:
    """
    对单个 subject 扫描 (b_e, sigma_ou) 网格，记录 f_sim, slope_sim。
    g_ee 固定为 gee_fixed。

    返回：
        df，列包括：
            b_e, sigma, f_sim, slope_sim, f_target, slope_target
    """
    os.makedirs(out_root, exist_ok=True)

    bes    = np.linspace(be_range[0],    be_range[1],    num_be)
    sigmas = np.linspace(sigma_range[0], sigma_range[1], num_sigma)

    rows = []
    for be in bes:
        for s in sigmas:
            cv_period_sim, amp_sim, counts = tvb_sim_single_eeg(
                b_e=float(be),
                sigma_ou=float(s),
                g_ee=float(gee_fixed),
                sim_dur_s=sim_dur_s,
                cut_transient_s=cut_transient_s,
                region_idx=region_idx,
                seed=seed,
                out_root=os.path.join(
                    out_root,
                    f"sid_{subj.sid}_be_{be:.2f}_s_{s:.3f}"
                ),
            )
            rows.append(
                {
                    "sid": subj.sid,
                    "axis_joint": subj.axis_joint,
                    "b_e": float(be),
                    "sigma": float(s),
                    "g_ee": float(gee_fixed),
                    "cv_period_sim": float(cv_period_sim),
                    "amp_so_sim": float(amp_sim),
                    "so_count_sim": int(counts),
                    "cv_period_target": float(subj.cv_period_target),
                    "amp_so_target": float(subj.amp_so_target),
                }
            )

    df = pd.DataFrame(rows)
    csv_path = os.path.join(
        out_root,
        f"sweep_be_sigma_sid_{subj.sid}_gee_{gee_fixed:.2f}.csv"
    )
    df.to_csv(csv_path, index=False)
    print(f"[SWEEP be–sigma] 保存扫描结果到: {csv_path}")
    return df

def sweep_be_sigma_for_subject_(
    subj: str,
    gee_fixed: float = 0.4,
    be_range=(59.0, 60.0),
    sigma_range=(0.5, 0.6),
    num_be: int = 2,
    num_sigma: int = 2,
    sim_dur_s: float = 20.0,
    cut_transient_s: float = 5.0,
    region_idx: int = 5,
    seed: int = 0,
    out_root: str = "runs_scan_be_sigma",
) -> pd.DataFrame:
    """
    对单个 subject 扫描 (b_e, sigma_ou) 网格，记录 f_sim, slope_sim。
    g_ee 固定为 gee_fixed。

    返回：
        df，列包括：
            b_e, sigma, f_sim, slope_sim, f_target, slope_target
    """
    os.makedirs(out_root, exist_ok=True)

    bes    = np.linspace(be_range[0],    be_range[1],    num_be)
    sigmas = np.linspace(sigma_range[0], sigma_range[1], num_sigma)

    rows = []
    for be in bes:
        for s in sigmas:
            res = tvb_sim_single_ve(
                b_e=float(be),
                sigma_ou=float(s),
                g_ee=float(gee_fixed),
                sim_dur_s=sim_dur_s,
                cut_transient_s=cut_transient_s,
                region_idx=region_idx,
                seed=seed,
                out_root=os.path.join(
                    out_root,
                    f"sid_{subj.sid}_be_{be:.2f}_s_{s:.3f}"
                ),
            )

            rows.append(
                {
                    # === subject / axis ===
                    "sid": subj.sid,
                    "axis_joint": float(subj.axis_joint),

                    # === parameters ===
                    "b_e": float(be),
                    "sigma": float(s),
                    "g_ee": float(gee_fixed),

                    # === simulation outputs (ve-based) ===
                    "so_flag_sim": bool(res["so_flag"]),
                    "so_count_sim": int(res["n_events"]),
                    "event_rate_sim": float(res["event_rate"]),
                    "psd_ratio_sim": float(res["psd_ratio"]),
                    "cv_period_sim": float(res["CV_period"]),
                    "amp_so_hz_sim": float(res["amp_hz"]),

                    # === targets (from SHHS / subject stats) ===
                    "cv_period_target": float(subj.cv_period_target),
                    "amp_so_target": float(subj.amp_so_target),
                }
            )

    df = pd.DataFrame(rows)
    csv_path = os.path.join(
        out_root,
        f"sweep_be_sigma_sid_{subj.sid}_gee_{gee_fixed:.2f}.csv"
    )
    df.to_csv(csv_path, index=False)
    print(f"[SWEEP be–sigma] 保存扫描结果到: {csv_path}")
    return df

# ============================================================
# 1. 扫描 (sigma, g_ee) → (f_sim, slope_sim)
# ============================================================

def sweep_sigma_gee_for_subject(
    subj: str,
    be_star: float,
    sigma_range=(0.9, 1.1),
    gee_range=(0.2, 0.5),
    num_sigma: int = 21,
    num_gee: int = 31,
    sim_dur_s: float = 20.0,
    cut_transient_s: float = 5.0,
    region_idx: int = 5,
    seed: int = 0,
    out_root: str = "runs_scan",
) -> pd.DataFrame:
    """
    对单个 subject 扫描 (sigma, g_ee) 网格，记录对应的 f_sim, slope_sim。

    返回：
        df，包含列：
            sigma, g_ee, f_sim, slope_sim
    """

    os.makedirs(out_root, exist_ok=True)
    sigmas = np.linspace(sigma_range[0], sigma_range[1], num_sigma)
    gees   = np.linspace(gee_range[0],   gee_range[1],   num_gee)

    rows = []
    for s in sigmas:
        for g in gees:
            f_sim, s_sim = tvb_sim_single(
                b_e=be_star,
                sigma_ou=float(s),
                g_ee=float(g),
                sim_dur_s=sim_dur_s,
                cut_transient_s=cut_transient_s,
                region_idx=region_idx,
                seed=seed,
                out_root=os.path.join(out_root, f"sid_{subj.sid}_s_{s:.3f}_g_{g:.3f}")
            )
            rows.append(
                {
                    "sid": subj.sid,
                    "axis_joint": subj.axis_joint,
                    "sigma": float(s),
                    "g_ee": float(g),
                    "f_sim": float(f_sim),
                    "slope_sim": float(s_sim),
                    "f_target": float(subj.f_target),
                    "slope_target": float(subj.slope_target),
                }
            )

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_root, f"sweep_sid_{subj.sid}_be_{be_star:.2f}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[SWEEP] 保存扫描结果到: {csv_path}")
    return df
import os
import numpy as np
import pandas as pd

def sweep_be_sigma_for_subject_veviw(
    subj,
    gee_fixed: float = 0.4,
    be_range=(59.0, 60.0),
    sigma_range=(0.5, 0.6),
    num_be: int = 2,
    num_sigma: int = 2,
    sim_dur_s: float = 20.0,
    cut_transient_s: float = 5.0,
    region_idx: int = 5,
    seed: int = 0,
    out_root: str = "runs_scan_be_sigma_be0_60",
    keep: bool = True,
    # synchrony config
    sync_lo: float = 0.16,
    sync_hi: float = 4.0,
    sync_use_bandpass: bool = True,
) -> pd.DataFrame:
    """
    对单个 subject 扫描 (b_e, sigma_ou) 网格，记录：
      - 单区 ve/vi/W mean/std
      - 全脑 ve/vi/W 统计（按区域汇总）
      - 全脑同步性（在 ve 上计算）

    注意：不计算任何 SO/事件特征。
    """
    os.makedirs(out_root, exist_ok=True)

    bes    = np.linspace(be_range[0],    be_range[1],    num_be)
    sigmas = np.linspace(sigma_range[0], sigma_range[1], num_sigma)


    rows = []
    for be in bes:
        for s in sigmas:
            run_dir = os.path.join(
                out_root,
                f"sid_{subj.sid}_be_{be:.2f}_s_{s:.3f}_gee_{gee_fixed:.3f}_sd_{seed}"
            )


            res = tvb_sim_single_ve_with_sync(
                b_e=float(be),
                sigma_ou=float(s),
                g_ee=float(gee_fixed),
                sim_dur_s=sim_dur_s,
                cut_transient_s=cut_transient_s,
                region_idx=region_idx,
                seed=seed,
                out_root=run_dir,
                keep=keep,
                nregions=68,
                sync_lo=sync_lo,
                sync_hi=sync_hi,
                sync_use_bandpass=sync_use_bandpass,
            )
            # ===== zyz修改，此处用于检测加入刺激后peak会发生什么变化 =====
            from detect_peaks import detect_from_file, compute_dt


            # 找真正存数据的子目录
            subdirs = [d for d in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, d))]
            if len(subdirs) == 0:
                raise ValueError("❌ 没有找到仿真输出目录")

            folder = os.path.join(run_dir, subdirs[0])

            # detect
            peak_times, trough_times = detect_from_file(folder)

            # 计算 Δt
            dt = compute_dt(peak_times, trough_times)

            print("mean dt:", dt.mean())


            # ===== 到这里结束检测波峰波谷分布（用于对比刺激产生的效果是否真正产生相位提前） =====
            #zyz修改，以下部分是检测波峰波谷
            #from detect_peaks import detect_from_file

            #folder = "/Users/zhangyuanzhuo/PycharmProjects/hybrid_brain_mac1/runs_tvb_be_sigma/scan_seed1/sid_201774_be_60.00_s_0.500_gee_0.400_sd_1/be60.00_s0.50_gee0.40_sd1"

            #peak_times, trough_times = detect_from_file(folder)

            #print("peaks:", peak_times[:5])

            # ===== 只有成功才做 detect =====
            peak_times, trough_times = detect_from_file(folder)
            dt = compute_dt(peak_times, trough_times)

            print("mean dt:", dt.mean())

            # ⭐ 保存
            np.save(os.path.join(folder, "peak_times.npy"), peak_times)
            np.save(os.path.join(folder, "trough_times.npy"), trough_times)
            np.save(os.path.join(folder, "dt.npy"), dt)

            # 判断 simulation 是否成功
            if res is None:
                rows.append({
                    "sid": subj.sid,
                    "b_e": float(be),
                    "sigma": float(s),
                    "g_ee": float(gee_fixed),
                    "ok": False,
                    "reason": "res is None",
                })
                continue

            if not res.get("ok", False):
                rows.append({
                    "sid": subj.sid,
                    "b_e": float(be),
                    "sigma": float(s),
                    "g_ee": float(gee_fixed),
                    "ok": False,
                    "reason": res.get("reason", "unknown"),
                })
                continue

            rows.append({
                "sid": subj.sid,
                "axis_joint": float(subj.axis_joint),
                "b_e": float(be),
                "sigma": float(s),
                "g_ee": float(gee_fixed),
                "ok": True,
                "mean_dt": float(dt.mean()),
            })
#zyz修改，这一段会影响detect，以及下部分rows，rows是作表用的，是我手动注释的
            # single = res["single_region_stats"]
            # glob   = res["global_stats"]
            # sync   = res["synchrony"]
            # par    = res["params"]
            # attr = res["attractor"]
            #
            #
            # rows.append({
            #     # subject meta
            #     "sid": subj.sid,
            #     "axis_joint": float(subj.axis_joint),
            #
            #     # parameters
            #     "b_e": float(be),
            #     "sigma": float(s),
            #     "g_ee": float(gee_fixed),
            #
            #     # run meta
            #     "ok": True,
            #     "region_idx": int(single["region_idx"]),
            #     "fs": float(par.get("fs", np.nan)),
            #     "dt_ms": float(par.get("dt_ms", np.nan)),
            #     "sim_dur_s": float(par.get("sim_dur_s", sim_dur_s)),
            #     "cut_transient_s": float(par.get("cut_transient_s", cut_transient_s)),
            #
            #     # single-region ve/vi/W mean/std
            #     "ve_mean_r": float(single["ve_mean"]),
            #     "ve_std_r":  float(single["ve_std"]),
            #     "vi_mean_r": float(single["vi_mean"]),
            #     "vi_std_r":  float(single["vi_std"]),
            #     "W_mean_r":  float(single["W_mean"]),
            #     "W_std_r":   float(single["W_std"]),
            #
            #     # global ve/vi/W stats (region-summary)
            #     "ve_mean_regions_mean": float(glob["ve_mean_across_regions_mean"]),
            #     "ve_mean_regions_std":  float(glob["ve_mean_across_regions_std"]),
            #     "ve_std_regions_mean":  float(glob["ve_std_across_regions_mean"]),
            #     "ve_std_regions_std":   float(glob["ve_std_across_regions_std"]),
            #
            #     "vi_mean_regions_mean": float(glob["vi_mean_across_regions_mean"]),
            #     "vi_mean_regions_std":  float(glob["vi_mean_across_regions_std"]),
            #     "vi_std_regions_mean":  float(glob["vi_std_across_regions_mean"]),
            #     "vi_std_regions_std":   float(glob["vi_std_across_regions_std"]),
            #
            #     "W_mean_regions_mean":  float(glob["W_mean_across_regions_mean"]),
            #     "W_mean_regions_std":   float(glob["W_mean_across_regions_std"]),
            #     "W_std_regions_mean":   float(glob["W_std_across_regions_mean"]),
            #     "W_std_regions_std":    float(glob["W_std_across_regions_std"]),
            #     "amp_muV_global": float(glob["amp_muV_global"]),
            #     "cv_period_muV_global": float(glob["cv_period_muV_global"]),
            #
            #     # synchrony
            #     "sync_to_global_mean": float(sync["sync_to_global_mean"]),
            #     "sync_pairwise_mean":  float(sync["sync_pairwise_mean"]),
            #     "sync_lo": float(sync.get("sync_band_lo", sync_lo)),
            #     "sync_hi": float(sync.get("sync_band_hi", sync_hi)),
            #     "sync_use_bandpass": bool(sync.get("sync_use_bandpass", sync_use_bandpass)),
            #     # attractor
            #
            #     "poincare_var": float(attr.get("poincare_var", np.nan)),
            #     "return_std": float(attr.get("return_std", np.nan)),
            #     "hf_lf_ratio": float(attr.get("hf_lf_ratio", np.nan))
            # })

    df = pd.DataFrame(rows)

    csv_path = os.path.join(
        out_root,
        f"sweep_be_sigma_sid_{subj.sid}_gee_{gee_fixed:.2f}.csv"
    )
    df.to_csv(csv_path, index=False)
    print(f"[SWEEP be–sigma] 保存扫描结果到: {csv_path}")
    return df

def sweep_sigma_gee_for_subject(
    subj: str,
    be_star: float,
    sigma_range=(0.9, 1.1),
    gee_range=(0.2, 0.5),
    num_sigma: int = 21,
    num_gee: int = 31,
    sim_dur_s: float = 20.0,
    cut_transient_s: float = 5.0,
    region_idx: int = 5,
    seed: int = 0,
    out_root: str = "runs_scan",
) -> pd.DataFrame:
    """
    对单个 subject 扫描 (sigma, g_ee) 网格，记录对应的 f_sim, slope_sim。

    返回：
        df，包含列：
            sigma, g_ee, f_sim, slope_sim
    """

    os.makedirs(out_root, exist_ok=True)
    sigmas = np.linspace(sigma_range[0], sigma_range[1], num_sigma)
    gees   = np.linspace(gee_range[0],   gee_range[1],   num_gee)

    rows = []
    for s in sigmas:
        for g in gees:
            f_sim, s_sim = tvb_sim_single(
                b_e=be_star,
                sigma_ou=float(s),
                g_ee=float(g),
                sim_dur_s=sim_dur_s,
                cut_transient_s=cut_transient_s,
                region_idx=region_idx,
                seed=seed,
                out_root=os.path.join(out_root, f"sid_{subj.sid}_s_{s:.3f}_g_{g:.3f}")
            )
            rows.append(
                {
                    "sid": subj.sid,
                    "axis_joint": subj.axis_joint,
                    "sigma": float(s),
                    "g_ee": float(g),
                    "f_sim": float(f_sim),
                    "slope_sim": float(s_sim),
                    "f_target": float(subj.f_target),
                    "slope_target": float(subj.slope_target),
                }
            )

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_root, f"sweep_sid_{subj.sid}_be_{be_star:.2f}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[SWEEP] 保存扫描结果到: {csv_path}")
    return df

# ============================================================
# 2. 扫描结果可视化（f 和 slope 的 heatmap）
# ============================================================
def plot_be_sigma_heatmaps(
    df: pd.DataFrame,
    subj: str,
    gee_fixed: float,
    out_root: str = "runs_scan_be_sigma",
    f_clim=None,
    slope_clim=None,
):
    """
    根据 sweep_be_sigma_for_subject 的 df，
    画两个 heatmap：
        - f_sim(b_e, sigma)
        - slope_sim(b_e, sigma)
    行：sigma，列：b_e
    """
    os.makedirs(out_root, exist_ok=True)

    pivot_f = df.pivot(index="sigma", columns="b_e", values="cv_period_sim")
    pivot_slope = df.pivot(index="sigma", columns="b_e", values="amp_so_sim")

    bes    = pivot_f.columns.values
    sigmas = pivot_f.index.values

    # f_sim heatmap
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    im1 = ax1.imshow(
        pivot_f.values,
        origin="lower",
        aspect="auto",
        extent=[bes[0], bes[-1], sigmas[0], sigmas[-1]],
        cmap="viridis",
    )
    if f_clim is not None:
        im1.set_clim(*f_clim)
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label("f_sim (Hz)")

    ax1.set_xlabel("b_e")
    ax1.set_ylabel("sigma")
    ax1.set_title(
        f"sid={subj.sid}  g_ee={gee_fixed:.2f}\n"
        f"target f={subj.cv_period_target:.3f} Hz"
    )
    f_png = os.path.join(
        out_root,
        f"heatmap_f_be_sigma_sid_{subj.sid}_gee_{gee_fixed:.2f}.png"
    )
    fig1.tight_layout()
    fig1.savefig(f_png, dpi=200)
    plt.close(fig1)
    print(f"[PLOT] 保存 f_sim(be,sigma) heatmap: {f_png}")

    # slope_sim heatmap
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    im2 = ax2.imshow(
        pivot_slope.values,
        origin="lower",
        aspect="auto",
        extent=[bes[0], bes[-1], sigmas[0], sigmas[-1]],
        cmap="viridis",
    )
    if slope_clim is not None:
        im2.set_clim(*slope_clim)
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label("slope_sim")

    ax2.set_xlabel("b_e")
    ax2.set_ylabel("sigma")
    ax2.set_title(
        f"sid={subj.sid}  g_ee={gee_fixed:.2f}\n"
        f"target slope={subj.amp_so_target:.3f}"
    )
    slope_png = os.path.join(
        out_root,
        f"heatmap_slope_be_sigma_sid_{subj.sid}_gee_{gee_fixed:.2f}.png"
    )
    fig2.tight_layout()
    fig2.savefig(slope_png, dpi=200)
    plt.close(fig2)
    print(f"[PLOT] 保存 slope_sim(be,sigma) heatmap: {slope_png}")
def summarize_f_slope_stats(df: pd.DataFrame, tag: str = ""):
    """
    对 sweep 结果里的 f_sim / slope_sim 做一个简单统计：
        - 有效样本数
        - min / max / mean / std
        - 若干分位数
    """
    import numpy as np

    # 去掉 inf / nan
    df_clean = df.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.dropna(subset=["cv_period_sim", "amp_so_sim"])

    print("\n" + "=" * 60)
    print(f"[STATS] f_sim / slope_sim 统计  {tag}")
    print(f"有效网格点数: {len(df_clean)} / 原始: {len(df)}")

    for col in ["cv_period_sim", "amp_so_sim"]:
        vals = df_clean[col].values
        if vals.size == 0:
            print(f"  {col}: 无有效数据")
            continue

        print(f"\n  {col}:")
        print(f"    min   = {np.min(vals):.4f}")
        print(f"    max   = {np.max(vals):.4f}")
        print(f"    mean  = {np.mean(vals):.4f}")
        print(f"    std   = {np.std(vals):.4f}")
        print(f"    p10   = {np.percentile(vals, 10):.4f}")
        print(f"    p50   = {np.percentile(vals, 50):.4f}")
        print(f"    p90   = {np.percentile(vals, 90):.4f}")
    print("=" * 60 + "\n")

def summarize_so_feature_stats(df: pd.DataFrame, tag: str = ""):
    """
    对 sweep 结果里的 SO 事件级特征做统计：
        - CV_period_sim
        - Amp_SO_sim (μV)

    注意：
        - 这些量来自 detect_so 的事件统计
        - NaN 通常表示该参数组合下没有稳定 SO
    """
    import numpy as np

    print("\n" + "=" * 70)
    print(f"[STATS] SO feature statistics  {tag}")
    print(f"原始 grid 点数: {len(df)}")

    # --- 只保留至少检测到 SO 的 ---
    df_clean = (
        df.replace([np.inf, -np.inf], np.nan)
          .dropna(subset=["cv_period_sim", "amp_so_sim"])
    )

    print(f"检测到 SO 的 grid 点数: {len(df_clean)}")
    print(f"SO 检出率: {len(df_clean) / max(len(df),1) * 100:.1f}%")

    for col, note in [
        ("cv_period_sim", "（SHHS 常见范围 ~0.15–0.25）"),
        ("amp_so_sim",    "（SHHS 常见范围 ~40–80 μV）"),
    ]:
        vals = df_clean[col].values
        if vals.size == 0:
            print(f"\n  {col}: 无有效 SO 事件")
            continue

        print(f"\n  {col} {note}")
        print(f"    min   = {np.min(vals):.4f}")
        print(f"    max   = {np.max(vals):.4f}")
        print(f"    mean  = {np.mean(vals):.4f}")
        print(f"    std   = {np.std(vals):.4f}")
        print(f"    p10   = {np.percentile(vals, 10):.4f}")
        print(f"    p50   = {np.percentile(vals, 50):.4f}")
        print(f"    p90   = {np.percentile(vals, 90):.4f}")

    print("=" * 70 + "\n")


def plot_sweep_heatmaps(
    df: pd.DataFrame,
    subj: str,
    be_star: float,
    out_root: str = "runs_scan",
    f_clim=None,
    slope_clim=None,
):
    """
    对 sweep_sigma_gee_for_subject 的结果画两个 heatmap：
        - f_sim(sigma, g_ee)
        - slope_sim(sigma, g_ee)

    默认用 imshow；如果你有美化需求可以自己再改。
    """
    os.makedirs(out_root, exist_ok=True)

    # pivot 成 2D 网格
    # 行：g_ee，列：sigma
    pivot_f = df.pivot(index="g_ee", columns="sigma", values="cv_period_sim")
    pivot_slope = df.pivot(index="g_ee", columns="sigma", values="amp_sim")

    sigmas = pivot_f.columns.values
    gees   = pivot_f.index.values

    S, G = np.meshgrid(sigmas, gees)

    # --- f_sim heatmap ---
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    im1 = ax1.imshow(
        pivot_f.values,
        origin="lower",
        aspect="auto",
        extent=[sigmas[0], sigmas[-1], gees[0], gees[-1]],
        cmap="viridis",
    )
    if f_clim is not None:
        im1.set_clim(*f_clim)
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label("cv_period")

    ax1.set_xlabel("sigma")
    ax1.set_ylabel("g_ee")
    ax1.set_title(
        f"sid={subj.sid}  be*={be_star:.2f}  target f={subj.f_target:.3f} Hz"
    )

    # 在图中画一条表示 target f 的 contour 或者只是写个文本即可
    # 这里简单写在标题里就够了

    f_png = os.path.join(out_root, f"heatmap_f_sid_{subj.sid}_be_{be_star:.2f}.png")
    fig1.tight_layout()
    fig1.savefig(f_png, dpi=200)
    plt.close(fig1)
    print(f"[PLOT] 保存 f_sim heatmap: {f_png}")

    # --- slope_sim heatmap ---
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    im2 = ax2.imshow(
        pivot_slope.values,
        origin="lower",
        aspect="auto",
        extent=[sigmas[0], sigmas[-1], gees[0], gees[-1]],
        cmap="viridis",
    )
    if slope_clim is not None:
        im2.set_clim(*slope_clim)
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label("slope_sim")

    ax2.set_xlabel("sigma")
    ax2.set_ylabel("g_ee")
    ax2.set_title(
        f"sid={subj.sid}  be*={be_star:.2f}  target amp={subj.slope_target:.3f}"
    )

    slope_png = os.path.join(out_root, f"heatmap_slope_sid_{subj.sid}_be_{be_star:.2f}.png")
    fig2.tight_layout()
    fig2.savefig(slope_png, dpi=200)
    plt.close(fig2)
    print(f"[PLOT] 保存 slope_sim heatmap: {slope_png}")

def plot_sweep_heatmaps_cv_amp(
    df: pd.DataFrame,
    subj,
    be_star: float,
    out_root: str = "runs_scan",
    cv_clim=None,
    amp_clim=None,
):
    """
    画两个 heatmap（只要 CV 和 Amp）：
      - cv_period_sim(sigma, g_ee)
      - amp_sim(sigma, g_ee)

    df 必须包含列：
      sigma, g_ee, cv_period_sim, amp_sim, cv_period_target, amp_so_target
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(out_root, exist_ok=True)

    # 为了 extent 对齐：强制排序后再 pivot
    df2 = df.copy().sort_values(["g_ee", "sigma"])

    # 取 target（同一 subject 扫描应当是常数）
    cv_t = float(df2["cv_period_target"].dropna().iloc[0]) if df2["cv_period_target"].notna().any() else np.nan
    amp_t = float(df2["amp_so_target"].dropna().iloc[0]) if df2["amp_so_target"].notna().any() else np.nan

    # 行：g_ee，列：sigma
    pivot_cv  = df2.pivot(index="g_ee", columns="sigma", values="cv_period_sim").sort_index(axis=0).sort_index(axis=1)
    pivot_amp = df2.pivot(index="g_ee", columns="sigma", values="amp_sim").sort_index(axis=0).sort_index(axis=1)

    sigmas = pivot_cv.columns.values
    gees   = pivot_cv.index.values

    # -------------------------
    # 1) CV_period heatmap
    # -------------------------
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    im1 = ax1.imshow(
        pivot_cv.values,
        origin="lower",
        aspect="auto",
        extent=[sigmas[0], sigmas[-1], gees[0], gees[-1]],
        cmap="viridis",
    )
    if cv_clim is not None:
        im1.set_clim(*cv_clim)

    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label("CV_period (sim)")

    ax1.set_xlabel("sigma_ou")
    ax1.set_ylabel("g_ee")
    ax1.set_title(
        f"sid={subj.sid}  b_e={be_star:.2f}\n"
        f"target CV_period={cv_t:.3f}"
    )

    cv_png = os.path.join(out_root, f"heatmap_cv_sid_{subj.sid}_be_{be_star:.2f}.png")
    fig1.tight_layout()
    fig1.savefig(cv_png, dpi=200)
    plt.close(fig1)
    print(f"[PLOT] 保存 CV_period heatmap: {cv_png}")

    # -------------------------
    # 2) Amp heatmap
    # -------------------------
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    im2 = ax2.imshow(
        pivot_amp.values,
        origin="lower",
        aspect="auto",
        extent=[sigmas[0], sigmas[-1], gees[0], gees[-1]],
        cmap="viridis",
    )
    if amp_clim is not None:
        im2.set_clim(*amp_clim)

    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label("Amp_SO (sim, μV)")

    ax2.set_xlabel("sigma_ou")
    ax2.set_ylabel("g_ee")
    ax2.set_title(
        f"sid={subj.sid}  b_e={be_star:.2f}\n"
        f"target Amp_SO={amp_t:.1f} μV"
    )

    amp_png = os.path.join(out_root, f"heatmap_amp_sid_{subj.sid}_be_{be_star:.2f}.png")
    fig2.tight_layout()
    fig2.savefig(amp_png, dpi=200)
    plt.close(fig2)
    print(f"[PLOT] 保存 Amp heatmap: {amp_png}")

# ============================================================
# 3. BO：对单个 subject 标定 (sigma, g_ee)
# ============================================================

def bo_calibrate_micro_params_for_subject(
    subj: str,
    be_star: float,
    bounds=None,
    feature_weights=(1.0, 0.1),
    n_trials: int = 40,
    sim_dur_s: float = 20.0,
    cut_transient_s: float = 5.0,
    region_idx: int = 5,
    out_root: str = "runs_bo_single",
    seed: int = 0,
):
    """
    对单个 subject，在固定 b_e = be_star 下，
    用 BO 寻找 (sigma, g_ee)，最小化：
        L = w_f (f_sim - f_target)^2 + w_s (s_sim - slope_target)^2

    返回:
        sigma_star, g_ee_star, best_loss
    """
    os.makedirs(out_root, exist_ok=True)

    if bounds is None:
        bounds = {
            "sigma": (0.9, 1.1),
            "g_ee": (0.2, 0.5),
        }
    w_f, w_s = feature_weights

    # --- 优先尝试 optuna ---
    try:
        import optuna
        from optuna.samplers import TPESampler

        def objective(trial):
            sigma = trial.suggest_float("sigma", *bounds["sigma"])
            g_ee  = trial.suggest_float("g_ee",  *bounds["g_ee"])

            out_dir = os.path.join(
                out_root,
                f"sid_{subj.sid}_be_{be_star:.2f}_trial_{trial.number}",
            )

            f_sim, s_sim = tvb_sim_single(
                b_e=be_star,
                sigma_ou=sigma,
                g_ee=g_ee,
                sim_dur_s=sim_dur_s,
                cut_transient_s=cut_transient_s,
                region_idx=region_idx,
                seed=seed,
                out_root=out_dir,
            )

            if not np.isfinite(f_sim) or not np.isfinite(s_sim):
                return 1e9

            loss = (
                w_f * (f_sim - subj.f_target) ** 2
                + w_s * (s_sim - subj.slope_target) ** 2
            )
            return float(loss)

        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=seed),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best = study.best_params
        sigma_star = float(best["sigma"])
        g_ee_star  = float(best["g_ee"])
        best_loss  = float(study.best_value)

        print(
            f"[BO] sid={subj.sid}, be*={be_star:.2f}, "
            f"sigma*={sigma_star:.4f}, g_ee*={g_ee_star:.4f}, loss={best_loss:.4f}"
        )
        return sigma_star, g_ee_star, best_loss

    except Exception as e:
        print(f"[BO] Optuna 不可用或失败 ({e})，退化为随机搜索。")

    # --- 兜底：随机搜索 ---
    rng = np.random.default_rng(seed)
    best_sigma, best_gee, best_loss = None, None, np.inf

    for t in range(n_trials):
        sigma = rng.uniform(*bounds["sigma"])
        g_ee  = rng.uniform(*bounds["g_ee"])

        out_dir = os.path.join(
            out_root,
            f"sid_{subj.sid}_be_{be_star:.2f}_rand_{t}",
        )

        f_sim, s_sim = tvb_sim_single(
            b_e=be_star,
            sigma_ou=sigma,
            g_ee=g_ee,
            sim_dur_s=sim_dur_s,
            cut_transient_s=cut_transient_s,
            region_idx=region_idx,
            seed=seed,
            out_root=out_dir,
        )
        if not np.isfinite(f_sim) or not np.isfinite(s_sim):
            continue

        loss = (
            w_f * (f_sim - subj.f_target) ** 2
            + w_s * (s_sim - subj.slope_target) ** 2
        )
        if loss < best_loss:
            best_loss = loss
            best_sigma = sigma
            best_gee   = g_ee

    if best_sigma is None or best_gee is None:
        best_sigma = sum(bounds["sigma"]) * 0.5
        best_gee   = sum(bounds["g_ee"]) * 0.5
        best_loss  = 1e9

    print(
        f"[BO-RAND] sid={subj.sid}, be*={be_star:.2f}, "
        f"sigma*={best_sigma:.4f}, g_ee*={best_gee:.4f}, loss={best_loss:.4f}"
    )
    return float(best_sigma), float(best_gee), float(best_loss)


# ============================================================
# 4. 对一批 subjects 跑 BO（可作为 80% train 的内层最优化）
# ============================================================

def run_stage2_bo_for_subjects(
    subjects: dict,
    be_lookup: dict,
    id_list: list,
    bounds=None,
    feature_weights=(1.0, 0.1),
    n_trials: int = 40,
    sim_dur_s: float = 20.0,
    cut_transient_s: float = 5.0,
    region_idx: int = 5,
    out_root: str = "runs_bo_batch",
    seed: int = 0,
) -> pd.DataFrame:
    """
    对一批 subjects（典型用法：train_ids）跑 BO 标定 (sigma*, g_ee*)。

    返回：
        df，包含 sid, axis_joint, age, ahi, bmi, b_e_star, sigma_star, g_ee_star, loss
    """
    os.makedirs(out_root, exist_ok=True)
    logs = []

    for i, sid in enumerate(id_list):
        subj = subjects[sid]
        be_star = be_lookup.get(sid, 60.0)

        sigma_star, gee_star, best_loss = bo_calibrate_micro_params_for_subject(
            subj=subj,
            be_star=be_star,
            bounds=bounds,
            feature_weights=feature_weights,
            n_trials=n_trials,
            sim_dur_s=sim_dur_s,
            cut_transient_s=cut_transient_s,
            region_idx=region_idx,
            out_root=os.path.join(out_root, "raw"),
            seed=seed + i,  # 稍微变一下 seed
        )

        logs.append(
            {
                "sid": sid,
                "axis_joint": subj.axis_joint,
                "age": subj.age,
                "ahi": subj.ahi,
                "bmi": subj.bmi,
                "b_e_star": be_star,
                "sigma_star": sigma_star,
                "g_ee_star": gee_star,
                "loss": best_loss,
            }
        )

    df = pd.DataFrame(logs)
    csv_path = os.path.join(out_root, "bo_micro_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"[BO-BATCH] 保存 BO 汇总到: {csv_path}")
    return df


# ============================================================
# 5. 一个简单的 main 示例：先做扫描，再做 BO
# ============================================================
def demo_scan_be_sigma():
    input_csv = "./data/shhs_so_features_axis.csv"
    out_root = "runs_tvb_be_sigma"
    os.makedirs(out_root, exist_ok=True)

    subjects = load_subjects_from_csv(input_csv)
    train_ids, _ = split_subjects(subjects, train_ratio=0.8, seed=2025)

    # 随便挑一个 subject 做演示
    sid_demo = train_ids[0]
    subj_demo = subjects[sid_demo]

    gee_fixed = 0.4  # 你可以后面再试 0.2 / 0.4 看差异

    df = sweep_be_sigma_for_subject_(
        subj=subj_demo,
        gee_fixed=gee_fixed,
        be_range=(60.0, 60.0),
        sigma_range=(0.4,0.5),
        num_be=1, #20
        num_sigma=2, # 4
        sim_dur_s=20.0,
        cut_transient_s=5.0,
        region_idx=5,
        seed=0,
        out_root=os.path.join(out_root, "scan"),
    )

    plot_be_sigma_heatmaps(
        df,
        subj=subj_demo,
        gee_fixed=gee_fixed,
        out_root=os.path.join(out_root, "scan"),
        # 例如你可以先看看 f 在 [0,2] 整体分布，
        # 如果太花再手动设 f_clim=(0.3,1.5)
        f_clim=None,
        slope_clim=None,
    )
    summarize_so_feature_stats(df, tag=f"sid={subj_demo.sid}, gee={gee_fixed:.2f}")


    print("be–sigma 扫描完成。")

def main_scan_and_bo():
    """
    示例流程：
        1. 读入 subject 聚合表；
        2. 对所有 subject 或子集做 Stage1 b_e 标定（或者用固定 b_e）；
        3. 挑一个代表 subject 做 (sigma,g_ee) 扫描 + heatmap；
        4. 对 train_ids 跑 BO 做 per-subject 标定。
    """
    input_csv = "./data/shhs_so_features_axis.csv"
    out_root  = "runs_tvb_scan_bo"
    os.makedirs(out_root, exist_ok=True)

    # 1) 读 subject
    subjects = load_subjects_from_csv(input_csv)
    train_ids, val_ids = split_subjects(subjects, train_ratio=0.8, seed=2025)

    # 2) b_e：这里先用固定值，等你确认 TVB 行为以后再切回 Stage1 标定也可以
    be_const = 57.0
    be_lookup = {sid: be_const for sid in (train_ids + val_ids)}

    # 3) 挑一个代表 subject 做扫描 + 可视化（你可以随便改 sid）
    sid_demo = train_ids[0]
    subj_demo = subjects[sid_demo]
    be_demo = be_lookup[sid_demo]

    df_sweep = sweep_sigma_gee_for_subject(
        subj=subj_demo,
        be_star=be_demo,
        sigma_range=(0.9, 1.1),
        gee_range=(0.2, 0.5),
        num_sigma=21,
        num_gee=31,
        sim_dur_s=20.0,
        cut_transient_s=5.0,
        region_idx=5,
        seed=0,
        out_root=os.path.join(out_root, "scan"),
    )
    plot_sweep_heatmaps_cv_amp(
        df_sweep,
        subj=subj_demo,
        be_star=be_demo,
        out_root=os.path.join(out_root, "scan"),
        f_clim=None,
        slope_clim=None,
    )

    # 4) 对 train_ids 跑 BO（可作为后面 meta-learning 的 train set）
    # _ = run_stage2_bo_for_subjects(
    #     subjects=subjects,
    #     be_lookup=be_lookup,
    #     id_list=train_ids,
    #     bounds={"sigma": (0.9, 1.1), "g_ee": (0.2, 0.5)},
    #     feature_weights=(1.0, 0.1),
    #     n_trials=40,
    #     sim_dur_s=20.0,
    #     cut_transient_s=5.0,
    #     region_idx=5,
    #     out_root=os.path.join(out_root, "bo"),
    #     seed=0,
    # )
    # print("Scan + BO 完成。")
def demo_scan_be_sigma_for_veviw():
    input_csv = "./data/shhs_so_features_axis.csv"   # 你现在仍用它来取 subject/axis
    out_root = "runs_tvb_be_sigma"
    os.makedirs(out_root, exist_ok=True)

    subjects = load_subjects_from_csv(input_csv)
    train_ids, _ = split_subjects(subjects, train_ratio=0.8, seed=2025)

    sid_demo = train_ids[0]
    subj_demo = subjects[sid_demo]

    gee_fixed = 0.4  # 你可后续试 0.2 / 0.4 / 0.5

    df = sweep_be_sigma_for_subject_veviw(
        subj=subj_demo,
        gee_fixed=gee_fixed,
        be_range=(30.0, 60.0),
        sigma_range=(0.5, 0.5),
        num_be=31,
        num_sigma=1,
        sim_dur_s=25.0,
        cut_transient_s=5.0,
        region_idx=5,
        seed=1,
        out_root=os.path.join(out_root, "scan"),
        keep=True,
        sync_lo=0.16,
        sync_hi=4.0,
        sync_use_bandpass=True,
    )

    # ===== 文本级快速总结（不画图）=====
    # 1) 只保留成功的点
    df_ok = df[df["ok"] == True].copy()
    df_ok = df_ok.sort_values(["b_e", "sigma"]).reset_index(drop=True)

    # 2) 打印关键信息：最小/中位/最大 b_e 对应的指标
    key_cols = [
        "b_e", "sigma", "g_ee",
        "ve_mean_r", "ve_std_r", "vi_mean_r", "vi_std_r", "W_mean_r", "W_std_r",
        "sync_to_global_mean", "sync_pairwise_mean",
        "ve_mean_regions_mean", "vi_mean_regions_mean", "W_mean_regions_mean",
    ]
    key_cols = [c for c in key_cols if c in df_ok.columns]

    print(f"[DEMO] sid={subj_demo.sid}, axis_joint={float(subj_demo.axis_joint):.4f}, gee={gee_fixed:.3f}")
    print(f"[DEMO] ok runs: {len(df_ok)}/{len(df)}")

    if len(df_ok) == 0:
        print("[DEMO] No successful simulations. Check logs / output folders.")
        return df

    # head/tail（看趋势）
    print("\n[DEMO] first 5 rows (low b_e):")
    print(df_ok[key_cols].head(5).to_string(index=False))

    print("\n[DEMO] last 5 rows (high b_e):")
    print(df_ok[key_cols].tail(5).to_string(index=False))

    # 3) 可选：与 b_e 的线性相关（快速判读方向性）
    def _safe_corr(x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 5:
            return np.nan
        return float(np.corrcoef(x[m], y[m])[0, 1])

    summary_corr = {}
    for col in ["ve_mean_r", "vi_mean_r", "W_mean_r", "sync_to_global_mean", "sync_pairwise_mean",
                "ve_mean_regions_mean", "vi_mean_regions_mean", "W_mean_regions_mean"]:
        if col in df_ok.columns:
            summary_corr[f"corr(b_e,{col})"] = _safe_corr(df_ok["b_e"], df_ok[col])

    print("\n[DEMO] correlations vs b_e (quick sanity check):")
    for k, v in summary_corr.items():
        print(f"  {k}: {v:.4f}" if np.isfinite(v) else f"  {k}: nan")

    print("\n[DEMO] be–sigma sweep finished.")
    return df

def demo_scan_be_sigma_for_veviw_():
    input_csv = "./data/shhs_so_features_axis.csv"
    out_root = "runs_tvb_be_sigma"
    os.makedirs(out_root, exist_ok=True)

    #subjects = load_subjects_from_csv(input_csv) #zyz修改，此处用于不读csv
    # train_ids, _ = split_subjects(subjects, train_ratio=0.8, seed=2025)
    #
    # sid_demo = train_ids[0]
    # subj_demo = subjects[sid_demo]

    #------zyz修改，以下到gee fixed结束--------
    class DummySubj:
        sid = 0
        axis_joint = 0

    subj_demo = DummySubj()
    #---------------------------------------


    gee_fixed = 0.4

    # ====== 关键改动 1：延长时长 + 更充分的 warm-up ======
    sim_dur_s = 240.0
    cut_transient_s = 20.0

    # ====== 关键改动 2：多 seed 重复，以检验吸引性/多稳态 ======
    seeds = [ 1 ]   # 你也可以用 10 个
    # 原来：be_vals = np.linspace(0.0, 10.0, 11)
    #be_vals = np.arange(0.0, 60.0 + 1e-9, 1.0)  # 0,1,2,...,60 共 61 个
    be_vals = np.array([60], dtype=float)

    all_runs = []
    print("[CHECK] be_vals =", be_vals, "len=", len(be_vals))
    for seed in seeds:
        df = sweep_be_sigma_for_subject_veviw(
            subj=subj_demo,
            gee_fixed=gee_fixed,
            be_range=(float(be_vals.min()), float(be_vals.max())),
            sigma_range=(0.5, 0.5),
            num_be=len(be_vals),
            num_sigma=1,
            sim_dur_s=sim_dur_s,
            cut_transient_s=cut_transient_s,
            region_idx=5,
            seed=seed,
            out_root=os.path.join(out_root, f"scan_seed1"),
            keep=True,   # ====== 关键改动 3：保留波形/中间文件，方便提取 Poincaré 等 ======
            sync_lo=0.16,
            sync_hi=4.0,
            sync_use_bandpass=True,
        )
        df["seed"] = seed
        all_runs.append(df)

    df_all = pd.concat(all_runs, axis=0, ignore_index=True)
    df_ok = df_all[df_all["ok"] == True].copy()

    if len(df_ok) == 0:
        print("[DEMO] No successful simulations. Check logs / output folders.")
        return df_all

    # ===================== 吸引子判定：跨 seed 的几何一致性 =====================
    feat_cols = ["poincare_var", "return_std", "hf_lf_ratio"]
    feat_cols = [c for c in feat_cols if c in df_ok.columns]

    if len(feat_cols) < 2:
        print("[WARN] Not enough attractor metrics in df_ok. "
              "Please add them in tvb_sim_single_ve_with_sync() and sweep output.")
        return df_all

    grp = df_ok.groupby("b_e")
    rows = []

    for be, g in grp:
        x = g[feat_cols].to_numpy(float)
        m = np.isfinite(x).all(axis=1)
        x = x[m]
        n = len(x)

        if n < 3:
            rows.append({"b_e": float(be), "n_runs": int(n), "attractor_flag": "insufficient_runs"})
            continue

        mu = np.mean(x, axis=0)
        sd = np.std(x, axis=0)
        rel = sd / (np.abs(mu) + 1e-9)

        # 几何一致性：seed 间是否收敛（可按需要调阈值）
        geom_stable = np.all(rel < 0.30)

        # 用均值来表示该 b_e 的典型几何尺度
        pv = mu[feat_cols.index("poincare_var")] if "poincare_var" in feat_cols else np.nan
        rs = mu[feat_cols.index("return_std")] if "return_std" in feat_cols else np.nan

        if not geom_stable:
            flag = "no_attractor_or_noise"
        else:
            flag = "attractor_candidate"

        rows.append({
            "b_e": float(be),
            "n_runs": int(n),
            "geom_relstd_mean": float(np.mean(rel)),
            "poincare_var_mean": float(pv),
            "return_std_mean": float(rs),
            "attractor_flag": flag
        })

    df_attr = pd.DataFrame(rows).sort_values("b_e").reset_index(drop=True)

    # 第二步：在 attractor_candidate 中区分 deep / shallow（相对阈值，稳健）
    mask = df_attr["attractor_flag"] == "attractor_candidate"
    if mask.any():
        pv_med = np.nanmedian(df_attr.loc[mask, "poincare_var_mean"])
        rs_med = np.nanmedian(df_attr.loc[mask, "return_std_mean"])

        shallow = (
                (df_attr["poincare_var_mean"] > pv_med) &
                (df_attr["return_std_mean"] > rs_med) &
                mask
        )
        df_attr.loc[shallow, "attractor_flag"] = "shallow_attractor (B-like)"
        df_attr.loc[mask & ~shallow, "attractor_flag"] = "deep_attractor (A-like)"

    print(f"[DEMO] sid={subj_demo.sid}, axis_joint={float(subj_demo.axis_joint):.4f}, gee={gee_fixed:.3f}")
    print(f"[DEMO] total ok runs: {len(df_ok)}/{len(df_all)}")
    print("\n[ATTRACTOR] summary by b_e (head):")
    print(df_attr.head(10).to_string(index=False))
    print("\n[ATTRACTOR] summary by b_e (tail):")
    print(df_attr.tail(10).to_string(index=False))

    # 保存汇总
    df_all.to_csv(os.path.join(out_root, "scan_all_runs.csv"), index=False)
    df_attr.to_csv(os.path.join(out_root, "scan_attractor_summary.csv"), index=False)

    return df_all, df_attr

# 如果你想单独跑这个流程，可以在文件末尾加：
if __name__ == "__main__":
    # main_scan_and_bo()
    # demo_scan_be_sigma()
    # demo_scan_be_sigma_for_veviw()
    demo_scan_be_sigma_for_veviw_()