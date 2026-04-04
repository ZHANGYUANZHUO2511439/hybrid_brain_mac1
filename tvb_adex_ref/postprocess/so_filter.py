# -*- coding: utf-8 -*-
"""
so_filter.py
------------
    ve (原始群体活动) → xbp (SO 频段带通信号)

"""

import numpy as np
from scipy.signal import butter, filtfilt, detrend


def ve_to_xbp_so(
    ve,
    dt_ms=5.0,
    f_low=0.5,
    f_high=1.25,
    order=4,
    demean=True,
    do_detrend=False,
):
    """
    将原始 ve 转换为 SO 频段带通信号 xbp

    Parameters
    ----------
    ve : ndarray, shape (T,) or (T, N)
        原始 TVB 输出信号
    dt_ms : float
        积分步长（毫秒）
    f_low, f_high : float
        SO 带通频段 (Hz)
    order : int
        Butterworth 阶数
    demean : bool
        是否去均值（强烈建议 True）
    do_detrend : bool
        是否线性去趋势（一般 False）

    Returns
    -------
    xbp : ndarray
        SO 带通信号，shape 与 ve 相同
    """
    ve = np.asarray(ve, dtype=float)

    # 采样率
    fs = 1000.0 / dt_ms

    # ---------- 预处理 ----------
    x = ve.copy()
    if do_detrend:
        x = detrend(x, axis=0, type="linear")
    if demean:
        x = x - np.mean(x, axis=0, keepdims=True)

    # ---------- 带通滤波 ----------
    b, a = butter(order, [f_low, f_high], btype="bandpass", fs=fs)

    # 零相位滤波
    xbp = filtfilt(b, a, x, axis=0)

    return xbp
