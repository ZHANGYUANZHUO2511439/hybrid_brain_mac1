# -*- coding: utf-8 -*-
"""
poincare.py
-----------
Poincaré 截面 / 取点逻辑
输入是：已经带通后的信号（xbp）
"""

import numpy as np


def zero_crossings_up(x):
    """
    找到上升沿过 0 点的位置 i：
        x[i] < 0 且 x[i+1] >= 0

    Parameters
    ----------
    x : ndarray, shape (T,)

    Returns
    -------
    up_idx : ndarray
        上升沿索引 i
    """
    x = np.asarray(x)
    assert x.ndim == 1, "zero_crossings_up 只接受 1D 信号"

    return np.where((x[:-1] < 0) & (x[1:] >= 0))[0]


def sample_after_upcross(x):
    """
    你的分岔取点规则：
        在上升沿过 0 点后，取下一个采样点

    Parameters
    ----------
    x : ndarray, shape (T,)
        已带通的 SO 信号（xbp）

    Returns
    -------
    points : ndarray
        取样点集合，用作分岔图纵坐标
    """
    up = zero_crossings_up(x)
    idx = up + 1
    idx = idx[idx < len(x)]
    return x[idx]
