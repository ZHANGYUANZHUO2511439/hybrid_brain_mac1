# -*- coding: utf-8 -*-
#读取ve
import os
import glob
import numpy as np

VE_IDX = 1  #确认的 ve

def load_ve_chunk_from_step(step_path: str, ve_idx: int = VE_IDX) -> np.ndarray:
    """
    从单个 step_*.npy 中读取 ve，返回 shape=(K, 68)
    其中 K=200（由 step 文件内部记录点数决定）
    """
    arr = np.load(step_path, allow_pickle=True)  # shape (1, K, 2), dtype=object
    K = arr.shape[1]

    # 每个采样点：arr[0, k, 0] 是 time(float)；arr[0, k, 1] 是 state ndarray (8,68,1)
    ve_list = []
    for k in range(K):
        state = arr[0, k, 1]          # (8,68,1)
        ve_k = state[ve_idx, :, 0]    # (68,)
        ve_list.append(ve_k)

    ve = np.stack(ve_list, axis=0)   # (K,68)
    return ve


def load_ve_from_run(run_dir: str, ve_idx: int = VE_IDX) -> np.ndarray:
    """
    从一个 run 输出目录（包含 step_*.npy）拼接完整 ve_full，返回 shape=(T,68)
    """
    step_files = sorted(glob.glob(os.path.join(run_dir, "step_*.npy")))
    if not step_files:
        raise FileNotFoundError(f"No step_*.npy found in: {run_dir}")

    chunks = [load_ve_chunk_from_step(p, ve_idx=ve_idx) for p in step_files]
    ve_full = np.concatenate(chunks, axis=0)  # (T,68)
    return ve_full
