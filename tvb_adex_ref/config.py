# -*- coding: utf-8 -*-
"""
集中管理工程路径配置（兼容旧代码版）

设计原则：
1. 对外全部暴露为 str，兼容旧 pipeline 中的：
   path + "/xxx"  这种字符串拼接
2. 内部仍然基于 pathlib.Path，便于以后逐步规范化
3. 不依赖运行时 cwd，从文件位置自动推断项目根目录
"""

from __future__ import annotations
from pathlib import Path
import os

# ============================================================
# 1. 项目根目录
# ------------------------------------------------------------
# 本文件位置：
# hybrid_brain_mac1/tvb_adex_ref/config.py
# 所以项目根目录是它的上一级
# ============================================================

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 对外暴露为字符串（兼容旧代码）
PROJECT_ROOT = str(_PROJECT_ROOT)

# ============================================================
# 2. 数据目录（Connectivity、模板、zip 等）
# ============================================================

_DATA_DIR = _PROJECT_ROOT / "data"
DATA_DIR = str(_DATA_DIR)

# ============================================================
# 3. 扫描输出根目录（pipeline 扫描结果）
# ============================================================

_RUNS_ROOT = _PROJECT_ROOT / "runs_tvb_be_sigma"
RUNS_ROOT = str(_RUNS_ROOT)

# ============================================================
# 4. 后处理输出目录
# ============================================================

_POSTPROCESS_OUT = _PROJECT_ROOT / "postprocess_out"
POSTPROCESS_OUT = str(_POSTPROCESS_OUT)

# ============================================================
# 5. 工具函数
# ============================================================

def ensure_dirs() -> None:
    """
    确保常用输出目录存在
    （避免每次手动 mkdir）
    """
    os.makedirs(POSTPROCESS_OUT, exist_ok=True)


def p(*parts: str) -> str:
    """
    以项目根目录为基准拼路径（返回 str）

    用法示例：
    p("data", "Connectivity.zip")
    """
    return str(_PROJECT_ROOT.joinpath(*parts))


# ============================================================
# 6. 调试用（可选）
# ============================================================

if __name__ == "__main__":
    print("[CONFIG DEBUG]")
    print("PROJECT_ROOT =", PROJECT_ROOT)
    print("DATA_DIR     =", DATA_DIR)
    print("RUNS_ROOT    =", RUNS_ROOT)
    print("POSTPROCESS_OUT =", POSTPROCESS_OUT)
