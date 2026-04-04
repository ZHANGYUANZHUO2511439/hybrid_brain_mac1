# -*- coding: utf-8 -*-
"""
bifurcation_fixed.py

用途：
- 不修改 pipeline 的前提下，对 runs_tvb_be_sigma 扫描输出做后处理：
  1) 读取每个 run 的 step_*.npy，提取 ve (T, 68)
  2) 计算全脑平均 ve_global (T,)
  3) SO 带通得到 xbp
  4) 用“上升沿过 0 的下一个点”作为 Poincaré 截面取点
  5) 汇总 (b_e, point) 并画散点“分岔图”
"""

import os
import re
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt

from tvb_adex_ref.postprocess.so_filter import ve_to_xbp_so


# =========================
# 1) 基本配置
# =========================

RUNS_ROOT = "runs_tvb_be_sigma"

# 你的扫描范围（横坐标）
BE_MIN, BE_MAX = 0.0, 60.0

# SO 带通参数
F_LOW, F_HIGH = 0.5, 1.25
ORDER = 4

# 采样时间步长（ms）
DT_MS = 5.0

# ve 在 state 里的索引
VE_IDX = 1

# debug 开关
VERBOSE = True


# =========================
# 2) 自动切换到项目根目录
# =========================

print("[DEBUG] CWD =", os.getcwd())
print("[DEBUG] exists runs_tvb_be_sigma? ", os.path.exists("runs_tvb_be_sigma"))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)

print("[DEBUG] CWD switched to =", os.getcwd())
print("[DEBUG] exists runs_tvb_be_sigma? ", os.path.exists("runs_tvb_be_sigma"))


# =========================
# 3) 工具函数
# =========================

def parse_be_from_path(path: str):
    """从目录名解析 b_e，例如 be_30.00"""
    m = re.search(r"be_(\d+(?:\.\d+)?)", path)
    return float(m.group(1)) if m else None


def step_number(step_path: str) -> int:
    """从 step_XX.npy 提取数字 XX，用于数字排序"""
    m = re.search(r"step_(\d+)\.npy$", os.path.basename(step_path))
    return int(m.group(1)) if m else -1


def find_run_dirs():
    """找到所有包含 step_0.npy 的 run 目录"""
    step0_list = glob.glob(os.path.join(RUNS_ROOT, "**", "step_0.npy"), recursive=True)

    print("[DEBUG] step_0.npy found:", len(step0_list))
    if step0_list:
        print("[DEBUG] first step_0.npy:", step0_list[0])

    run_dirs = sorted({os.path.dirname(p) for p in step0_list})
    if VERBOSE:
        print("[DEBUG] run_dirs example:", run_dirs[:3])

    return run_dirs


def load_ve_chunk_from_step(step_path: str, ve_idx: int = VE_IDX):
    """
    从单个 step_*.npy 读取 ve，返回 shape=(K,68)
    若 step 文件为空/结构异常，返回 None（并跳过）
    """
    arr = np.load(step_path, allow_pickle=True)

    # ✅ 空 step：例如 shape=(1,0) 或 size==0
    if isinstance(arr, np.ndarray) and arr.size == 0:
        if VERBOSE:
            print("[DEBUG] empty step, skip:", step_path, "shape=", getattr(arr, "shape", None))
        return None

    # ✅ 正常 step 应为 object ndarray, ndim=3, shape=(1,K,2)
    if not (isinstance(arr, np.ndarray) and arr.dtype == object and arr.ndim == 3):
        if VERBOSE:
            print("[DEBUG] unexpected step format, skip:", step_path, "shape=", getattr(arr, "shape", None))
        return None

    if arr.shape[0] != 1 or arr.shape[2] != 2:
        if VERBOSE:
            print("[DEBUG] unexpected step shape, skip:", step_path, "shape=", arr.shape)
        return None

    K = arr.shape[1]
    if K <= 0:
        if VERBOSE:
            print("[DEBUG] step has zero samples, skip:", step_path, "shape=", arr.shape)
        return None

    ve = np.empty((K, 68), dtype=float)
    for k in range(K):
        state = arr[0, k, 1]  # (8,68,1)
        ve[k, :] = state[ve_idx, :, 0]
    return ve


def load_ve_from_run(run_dir: str, ve_idx: int = VE_IDX) -> np.ndarray:
    """读取一个 run 的所有有效 step_数字.npy 并拼接为 (T, 68)"""

    # ✅ 只匹配 step_0.npy / step_1.npy / ... 排除 step_init.npy
    step_files = glob.glob(os.path.join(run_dir, "step_[0-9]*.npy"))
    step_files = sorted(step_files, key=step_number)

    if VERBOSE:
        print("[DEBUG] run_dir =", run_dir)
        print("[DEBUG] step_files found =", len(step_files))
        if step_files:
            print("[DEBUG] first step_file =", os.path.basename(step_files[0]))

    if not step_files:
        raise RuntimeError(f"No numeric step files in {run_dir}")

    chunks = []
    for p in step_files:
        ve_chunk = load_ve_chunk_from_step(p, ve_idx=ve_idx)
        if ve_chunk is None:
            continue
        chunks.append(ve_chunk)

    if not chunks:
        raise RuntimeError(f"All step files empty or invalid in {run_dir}")

    return np.concatenate(chunks, axis=0)


def sample_after_upcross_1d(x: np.ndarray) -> np.ndarray:
    """
    Poincaré 截面取点：上升沿过 0 的下一个点

    返回
    ----
    points : ndarray, shape (N,)
        若没有过零点，返回空数组
    """
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return np.array([], dtype=float)

    up = np.where((x[:-1] < 0.0) & (x[1:] >= 0.0))[0]
    idx = up + 1

    if idx.size == 0:
        return np.array([], dtype=float)

    return x[idx]


# =========================
# 4) 主流程
# =========================

def main():
    run_dirs = find_run_dirs()
    print(f"[INFO] Found run dirs: {len(run_dirs)}")

    rows = []

    for rd in run_dirs:
        be = parse_be_from_path(rd) or parse_be_from_path(os.path.dirname(rd))
        if be is None or not (BE_MIN <= be <= BE_MAX):
            continue

        try:
            ve = load_ve_from_run(rd, ve_idx=VE_IDX)   # (T,68)
            ve_global = ve.mean(axis=1)                # (T,)

            if VERBOSE:
                print(f"[DEBUG] be={be:.2f} ve_global shape={ve_global.shape} "
                      f"min/max={ve_global.min():.6g}/{ve_global.max():.6g}")

            xbp = ve_to_xbp_so(
                ve_global,
                dt_ms=DT_MS,
                f_low=F_LOW,
                f_high=F_HIGH,
                order=ORDER,
                demean=True,
                do_detrend=False,
            )

            # 再去一次均值，保证穿 0 更稳定（不改变动力学，只消除偏置）
            xbp = xbp - np.nanmean(xbp)

            if VERBOSE:
                print(f"[DEBUG] be={be:.2f} xbp shape={xbp.shape} "
                      f"min/max={np.nanmin(xbp):.6g}/{np.nanmax(xbp):.6g} "
                      f"nan={np.isnan(xbp).sum()}")

            points = sample_after_upcross_1d(xbp)

            if VERBOSE:
                print(f"[DEBUG] be={be:.2f} points={len(points)}")

            if len(points) == 0:
                continue

            for p in points:
                rows.append((be, float(p), rd))

            print(f"[OK] be={be:.2f} points={len(points)}")

        except Exception as e:
            print(f"[WARN] failed: {rd}\n  -> {e}")

    os.makedirs("postprocess_out", exist_ok=True)

    out_csv = "postprocess_out/bifurcation_points_be0_60.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["b_e", "point", "run_dir"])
        w.writerows(rows)

    print(f"[DONE] CSV saved: {out_csv} (rows={len(rows)})")

    if rows:
        be_vals = [r[0] for r in rows]
        pts = [r[1] for r in rows]

        plt.figure()
        plt.scatter(be_vals, pts, s=3)
        plt.xlabel("b_e")
        plt.ylabel("Poincaré points on xbp (SO band-passed global ve)")
        plt.title("Bifurcation-like scatter (be 0–60)")
        out_png = "postprocess_out/bifurcation_be0_60.png"
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        print(f"[DONE] Figure saved: {out_png}")


if __name__ == "__main__":
    main()
