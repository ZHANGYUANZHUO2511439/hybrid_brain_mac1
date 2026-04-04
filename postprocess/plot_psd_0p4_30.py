# plot_psd_be_scan_auto.py
import os
import re
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# =========================
# User config
# =========================
SEED_DIR = "/Users/zhangyuanzhuo/PycharmProjects/hybrid_brain_mac1/runs_tvb_be_sigma/scan_seed1"  # 改成你的 a_e_10_seed1_scan
OUT_DIR  = "/Users/zhangyuanzhuo/PycharmProjects/hybrid_brain_mac1/postprocess_psd_out/pse_per_be_weight_noise_5e-5"  # 输出目录

BE_MIN, BE_MAX = 0.0, 60.0

# 二选一：DT_MS(毫秒) 或 FS(Hz)
DT_MS = 5.0
FS = None

# 频谱范围
F_LO, F_HI = 0.4, 30.0

# Welch 参数
WELCH_NPERSEG_SEC = 5000
WELCH_NOVERLAP_FRAC = 0.5

# 数据维度常量
NREGIONS = 68
VE_INDEX_PREFER = 0  # 优先取 state 的第1个变量当 ve（如果不存在就自动选方差最大）

# 预处理
REMOVE_MEAN = True
DETREND = True


# =========================
# Utils
# =========================
def infer_fs(dt_ms, fs):
    if fs is not None:
        return float(fs)
    if dt_ms is None:
        raise ValueError("DT_MS 和 FS 至少填一个。")
    dt = float(dt_ms) / 1000.0
    if dt <= 0:
        raise ValueError("DT_MS 必须 > 0")
    return 1.0 / dt

def natural_step_key(p):
    m = re.search(r"step_(\d+)\.npy$", os.path.basename(p))
    return int(m.group(1)) if m else 10**18

def parse_be_from_path(path: str):
    # 适配：.../sid_201774_be_55.00_s_0.500_... 或 .../be55.00_s0.50...
    m = re.search(r"(?:^|[\/_])be[_=]?(\d+(?:\.\d+)?)", path)
    return float(m.group(1)) if m else None

def find_run_dirs(seed_dir: str):
    # “包含 step_*.npy 的目录”就是 run_dir
    step_paths = glob.glob(os.path.join(seed_dir, "**", "step_*.npy"), recursive=True)
    return sorted({os.path.dirname(p) for p in step_paths})

def compute_psd_welch(x, fs, nperseg, noverlap_frac):
    nperseg = int(nperseg)                 # 现在是点数
    noverlap = int(round(noverlap_frac * nperseg))

    # 安全：数据太短时处理
    if len(x) < nperseg:
        # 方案1：严格跳过（可比性最好）
        # return None, None

        # 方案2：降级到 len(x)（能画出来，但不再固定5000）
        nperseg = len(x)
        noverlap = int(round(noverlap_frac * nperseg))

    f, pxx = signal.welch(
        x,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        scaling="density",
        average="mean",
    )
    print(f"[DEBUG] len(x)={len(x)}, fs={fs}, nperseg={nperseg}, noverlap={noverlap}")
    return f, pxx

def crop_band(f, pxx, f_lo, f_hi):
    m = (f >= f_lo) & (f <= f_hi)
    return f[m], pxx[m]
#----------1月17日 将每个be都生成psd----------------
import os
import numpy as np
import matplotlib.pyplot as plt

def save_psd_one_be(out_dir: str, be: float, f: np.ndarray, pxx: np.ndarray,
                    f_lo: float, f_hi: float, idx: int, run_name: str,
                    logy: bool = True):
    """每个 be 保存一张 PSD 图"""
    os.makedirs(out_dir, exist_ok=True)

    # 频段裁剪
    m = (f >= f_lo) & (f <= f_hi)
    ff = f[m]
    pp = pxx[m]

    plt.figure(figsize=(7, 4))
    if logy:
        plt.semilogy(ff, pp + 1e-30)
        ylab = "PSD (log scale)"
    else:
        plt.plot(ff, pp)
        ylab = "PSD"

    plt.xlabel("Frequency (Hz)")
    plt.ylabel(ylab)
    plt.title(f"Welch PSD | be={be:.2f} | idx={idx} | {run_name}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 文件名：带 be / idx / 频段
    fn = f"psd_be_{be:06.2f}_idx{idx}_f{f_lo:g}-{f_hi:g}.png"
    plt.savefig(os.path.join(out_dir, fn), dpi=180)
    plt.close()
    #--------------------------------------------------------------

# =========================
# Auto parsing for TVB object npy
# =========================
def _unwrap_to_ndarray(x):
    """把 object 里的内容尽量解包成 ndarray；返回 ndarray 或 None"""
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, (list, tuple)) and len(x) > 0:
        # 常见：第0个就是 ndarray
        if isinstance(x[0], np.ndarray):
            return x[0]
        # 或者里面某个是 ndarray
        for v in x:
            if isinstance(v, np.ndarray):
                return v
    if isinstance(x, dict):
        for key in ["state", "x", "data", "arr"]:
            if key in x and isinstance(x[key], np.ndarray):
                return x[key]
        for v in x.values():
            if isinstance(v, np.ndarray):
                return v
    return None

def _find_state_array(obj, n_regions=NREGIONS):
    """
    在 obj（可能是 ndarray/tuple/dict）里寻找“包含 68 脑区维”的 ndarray
    """
    a = _unwrap_to_ndarray(obj)
    if a is None:
        return None
    a = np.asarray(a)
    if n_regions in a.shape:
        return a
    return None

def _extract_region_vector(state_arr, ve_idx_prefer=VE_INDEX_PREFER, n_regions=NREGIONS):
    """
    从 state_arr（单个采样点的 state）抽一个 (68,) 的向量（各脑区）
    自动识别 region 轴 + 自动选 ve 变量
    """
    a = np.asarray(state_arr)
    # 去掉长度为1的维度（常见最后一个 1）
    a = np.squeeze(a)

    if n_regions not in a.shape:
        return None

    # 找 region 轴
    ridx = list(a.shape).index(n_regions)
    # 把 region 移到最后 => (..., 68)
    a = np.moveaxis(a, ridx, -1)

    # 现在最后一维是 68
    if a.ndim == 1:
        # (68,)
        return a.astype(np.float64)

    # 最常见 (V, 68)
    if a.ndim == 2:
        V = a.shape[0]
        if V > ve_idx_prefer:
            return a[ve_idx_prefer, :].astype(np.float64)

        # 不再兜底：越界就报错，避免 idx=3/4 实际拿到同一个变量
        raise IndexError(f"ve_idx_prefer={ve_idx_prefer} out of range, V={V}")

    # 更高维：先展平成 (M, 68)，选方差最大的一行
    flat = a.reshape(-1, n_regions)
    vars_ = np.var(flat, axis=1)
    return flat[int(np.argmax(vars_)), :].astype(np.float64)

def _extract_ve_global_from_step_arr(arr, n_regions=NREGIONS):
    """
    从单个 step 文件加载出来的 arr 中提取 ve_global 时间序列（长度 K）
    兼容你当前的 step_0.npy: (1, K, 2) dtype=object
    返回 ve_k (K,) 或 None（如果该 step 是空/不支持）
    """
    # 空块：shape (1,0)
    if arr.ndim == 2 and arr.shape == (1, 0):
        return None

    # 你的主要格式：(1, K, 2) object
    if arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[-1] == 2:
        K = arr.shape[1]
        if K == 0:
            return None

        ve_k = np.empty((K,), dtype=np.float64)

        for k in range(K):
            # 每个采样点两个槽位：slot 0/1
            # 我们不写死哪个是 state，而是两个都试，谁能解析出 68 脑区就用谁
            cand0 = arr[0, k, 0]
            cand1 = arr[0, k, 1]

            state0 = _find_state_array(cand0, n_regions=n_regions)
            state1 = _find_state_array(cand1, n_regions=n_regions)

            state = state1 if state1 is not None else state0
            if state is None:
                raise ValueError(
                    f"Cannot find state array with {n_regions} regions at k={k}. "
                    f"type(slot0)={type(cand0)}, type(slot1)={type(cand1)}"
                )

            reg_vec = _extract_region_vector(state, n_regions=n_regions)
            if reg_vec is None:
                raise ValueError(f"Found state but cannot extract region vector. state_shape={getattr(state,'shape',None)}")

            ve_k[k] = float(np.mean(reg_vec))  # 全脑平均

        return ve_k

    # 其它格式：你之前遇到过 step_init.npy (14,1,68,1)，我们在外层已经排除
    return None


def load_ve_global_from_run(run_dir: str):
    """
    读取 run_dir 下 step_数字.npy，拼接 ve_global(t)
    """
    # 先抓 step_*.npy，再用正则筛掉 step_init.npy 等
    all_steps = glob.glob(os.path.join(run_dir, "step_*.npy"))
    step_files = []
    for fp in all_steps:
        base = os.path.basename(fp)
        if re.match(r"^step_\d+\.npy$", base):
            step_files.append(fp)
    step_files = sorted(step_files, key=natural_step_key)

    if not step_files:
        return None

    chunks = []
    for fp in step_files:
        arr = np.load(fp, allow_pickle=True)

        ve_k = _extract_ve_global_from_step_arr(arr, n_regions=NREGIONS)
        if ve_k is None:
            # 空块或不支持的块
            # 你看到的 step_240.npy (1,0) 就会走这里
            # print(f"[WARN] skip empty/unsupported step: {fp} shape={arr.shape}")
            continue

        chunks.append(ve_k)

    if not chunks:
        return None

    ve_global = np.concatenate(chunks, axis=0)

    if REMOVE_MEAN:
        ve_global = ve_global - np.mean(ve_global)

    if DETREND:
        ve_global = signal.detrend(ve_global, type="linear")

    return ve_global


# =========================
# Main
# =========================
def main():
    fs = infer_fs(DT_MS, FS)

    if not os.path.isdir(SEED_DIR):
        raise FileNotFoundError(f"SEED_DIR not found: {SEED_DIR}")

    os.makedirs(OUT_DIR, exist_ok=True)

    run_dirs = find_run_dirs(SEED_DIR)

    def load_json(p):
        import json
        with open(p, "r") as f:
            return json.load(f)

    def get_param_json(rd: str) -> str | None:
        # 优先找 run_dir 里自己的 parameter.json
        import os
        cand = os.path.join(rd, "parameter.json")
        if os.path.exists(cand):
            return cand
        return None

    runs = []
    skipped_nojson = 0
    skipped_tau = 0

    for rd in run_dirs:
        pj = get_param_json(rd)
        if pj is None:
            skipped_nojson += 1
            continue
        obj = load_json(pj)
        pm = obj.get("parameter_model", {})
        be = pm.get("b_e", None)


        if be is None:
            continue

        runs.append((float(be), rd, pj))
    print("[DEBUG] run_dirs found:", len(run_dirs))
    print("[DEBUG] example run_dir:", run_dirs[0] if run_dirs else "NONE")

    runs.sort(key=lambda x: x[0])  # 按 b_e 排序
    be_lo = runs[0][0]
    be_hi = runs[-1][0]
    tag = f"be_{be_lo:.0f}_{be_hi:.0f}_n{len(runs)}"

    print(
        f"Runs total: {len(run_dirs)}, usable: {len(runs)}, skipped_nojson: {skipped_nojson}, skipped_tau: {skipped_tau}")
    print("First 10 be:", [b for b, _, _ in runs[:10]])
    print("Last 10 be:", [b for b, _, _ in runs[-10:]])


    results = []
    n_total = 0
    n_used = 0

    for rd in run_dirs:
        n_total += 1

        be = parse_be_from_path(rd)
        if be is None:
            # 兜底：如果你某些 run 没写进路径，就尝试读 json
            for cand in ["params.json", "config.json", "run_params.json", "meta.json"]:
                jp = os.path.join(rd, cand)
                if os.path.exists(jp):
                    try:
                        with open(jp, "r", encoding="utf-8") as f:
                            obj = json.load(f)
                        for k in ["b_e", "be", "b_e_value", "be_value"]:
                            if k in obj:
                                be = float(obj[k])
                                break
                        if be is not None:
                            break
                    except Exception:
                        pass

        if be is None:
            continue
        if not (BE_MIN <= be <= BE_MAX):
            continue

        ve_global = load_ve_global_from_run(rd)
        if ve_global is None or len(ve_global) < 16:
            continue

        f, pxx = compute_psd_welch(
            ve_global,
            fs=fs,
            nperseg=5000,
            noverlap_frac=0.5,
        )
        f, pxx = crop_band(f, pxx, F_LO, F_HI)
        results.append((be, f, pxx))
        n_used += 1
#---------------1月17日-----------------------------------------
        per_dir = os.path.join(OUT_DIR, "psd_per_be")
        save_psd_one_be(per_dir, be, f, pxx, F_LO, F_HI, VE_INDEX_PREFER, os.path.basename(rd), logy=True)
#----------------------------------------------------------------

        print(f"[OK] be={be:g}  T={len(ve_global)}  run={os.path.basename(rd)}")

    if not results:
        raise RuntimeError(
            "没有收集到任何 PSD 结果。\n"
            "说明：\n"
            "1) be 虽然解析到了，但 ve_global 没有成功拼出来（解析 state 失败或 step 全空）\n"
            "2) 或者 DT_MS/FS 不对导致后续异常被你中断（一般会报错而不是 results 为空）\n"
            "你可以把某个 run 的 step_0.npy 的内部结构再 probe 一下，我也可以继续适配。\n"
        )

    results.sort(key=lambda t: t[0])

    # ---------- Figure 1: all PSD curves ----------
    plt.figure()
    for be, f, pxx in results:
        plt.plot(f, pxx, linewidth=1.0)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (a.u.^2/Hz)")
    plt.title(f"Welch PSD of ve_global (seed=1, be {BE_MIN:g}-{BE_MAX:g})")
    plt.tight_layout()
    fp1 = os.path.join(OUT_DIR, f"psd_lines_be_{BE_MIN:g}_{BE_MAX:g}.png")
    plt.savefig(fp1, dpi=200)
    plt.close()

    # ---------- Figure 2: be-frequency heatmap ----------
    f_ref = results[0][1]
    be_list = np.array([r[0] for r in results], dtype=float)

    P = np.zeros((len(results), len(f_ref)), dtype=float)
    for i, (be, f, pxx) in enumerate(results):
        if np.allclose(f, f_ref):
            P[i] = pxx
        else:
            P[i] = np.interp(f_ref, f, pxx)

    P_log = np.log10(P + 1e-30)

    plt.figure()
    plt.imshow(
        P_log,
        aspect="auto",
        origin="lower",
        extent=[f_ref[0], f_ref[-1], be_list[0], be_list[-1]],
    )
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("b_e")
    plt.title("PSD heatmap (log10) of ve_global (seed=1)")
    plt.colorbar(label="log10(PSD)")
    plt.tight_layout()
    fp2 = os.path.join(OUT_DIR, f"psd_heatmap_be_{BE_MIN:g}_{BE_MAX:g}.png")
    plt.savefig(fp2, dpi=200)
    plt.close()

    print("\nDone.")
    print(f"Runs total: {n_total}, used: {n_used}")
    print("Saved:")
    print(" ", fp1)
    print(" ", fp2)

if __name__ == "__main__":
    main()
