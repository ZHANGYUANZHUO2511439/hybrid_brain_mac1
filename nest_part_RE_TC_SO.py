# NEST 3.x
import nest
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import math
from scipy.signal import hilbert

# ---------- 小工具 ----------
def bandpass(x, fs, lo, hi, order=2):
    b, a = butter(order, [lo/(fs/2), hi/(fs/2)], btype="band")
    return filtfilt(b, a, x)

def randomize_vm(pop, vmin=-70.0, vmax=-55.0):
    vals = np.random.uniform(vmin, vmax, size=len(pop))
    nest.SetStatus(pop, [{"V_m": float(v)} for v in vals])
def lfp_from_mm(events, dt, T_ms):
    """把 multimeter 的 times/V_m/senders 对齐到规则时标后做均值，得到 LFP-like。"""
    t = np.asarray(events["times"]); v = np.asarray(events["V_m"]); s = np.asarray(events["senders"])
    nbin = int(T_ms/dt)
    grid = np.arange(nbin) * dt
    ids = np.unique(s)
    V = np.full((len(ids), nbin), np.nan, float)
    idx = np.clip(np.round(t/dt).astype(int), 0, nbin-1)
    for i, nid in enumerate(ids):
        m = (s == nid)
        V[i, idx[m]] = v[m]
        # 前向填充
        row = V[i]
        nz = np.where(~np.isnan(row))[0]
        if nz.size:
            row[:nz[0]] = row[nz[0]]
            for k in range(nz[0]+1, nbin):
                if np.isnan(row[k]): row[k] = row[k-1]
    lfp = np.nan_to_num(np.nanmean(V, axis=0))
    lfp = np.clip(lfp, -90.0, 50.0)

    return grid, lfp

import math

# ---- NEST→TVB：从 TC 的 LFP-like 提取 A_TC、latency、sigma_ms（半高宽） ----
def extract_tc_pulse_features_from_trace(t, lfp,
                                         stim_on_ms,
                                         baseline_ms=200.0,
                                         search_lo_ms=40.0,
                                         search_hi_ms=300.0,
                                         min_prominence=0.2):
    """
    t, lfp: 直接用你已有的 t_tc, v_tc（均匀网格上的 LFP-like）
    stim_on_ms: 你的声音刺激开始时刻（aud_on）
    baseline_ms: 刺激前用于估计基线
    search_lo/hi_ms: 刺激后找峰范围
    min_prominence: 峰显著性阈值（与 lfp 量级一致）
    """
    t = np.asarray(t); y = np.asarray(lfp)

    # 1) 基线
    pre_mask = (t >= max(0.0, stim_on_ms - baseline_ms)) & (t < stim_on_ms)
    if not np.any(pre_mask):
        return {"ok": False, "reason": "no_baseline"}
    baseline = float(np.mean(y[pre_mask]))

    # 2) 搜索窗口（刺激后）
    win_mask = (t >= (stim_on_ms + search_lo_ms)) & (t <= (stim_on_ms + search_hi_ms))
    if not np.any(win_mask):
        return {"ok": False, "reason": "no_search_window"}

    t_win = t[win_mask]
    y_win = y[win_mask] - baseline

    # 3) 峰值（简单稳健：直接取 y_win 最大点；你也可以换成 find_peaks）
    p_rel = int(np.argmax(y_win))
    A_TC  = float(y_win[p_rel])
    t_peak= float(t_win[p_rel])
    latency = t_peak - float(stim_on_ms)

    # 4) 半高宽（用 (peak+baseline)/2 的水平找左右交点）
    half = baseline + A_TC/2.0
    # 找全局索引
    p_abs = np.where(t == t_peak)[0]
    if p_abs.size == 0:
        return {"ok": False, "reason": "no_global_peak_match"}
    p_abs = int(p_abs[0])

    # 往左找第一个 <= half 的点
    li = p_abs
    while li > 0 and y[li] > half:
        li -= 1
    # 往右找第一个 <= half 的点
    ri = p_abs
    while ri < len(y)-1 and y[ri] > half:
        ri += 1

    if li == p_abs or ri == p_abs:
        sigma_ms = 50.0  # 兜底
    else:
        sigma_ms = float(t[ri] - t[li])

    return {
        "ok": True,
        "reason": "ok",
        "A_TC": A_TC,
        "latency_ms": float(latency),
        "sigma_ms": float(sigma_ms),
        "t_peak_ms": float(t_peak),
        "baseline": float(baseline),
    }

def angdiff(a):
    # wrap 到 [-pi, pi]
    return (a + math.pi) % (2*math.pi) - math.pi

def apply_gate(t):
    phi  = 2.0*math.pi*so_hz*(t/1000.0) + phi0
    dphi = angdiff(phi - phi_opt)
    c    = math.cos(dphi)
    s    = math.sin(dphi)

    # 只允许“峰值附近 & 上升沿（sin<0）”
    in_window   = (abs(dphi) <= phase_win)
    on_upslope  = (s < 0.0)   # 若想要下降沿，改为 (s > 0.0)

    if in_window and on_upslope:
        amp = A_dc_pos * c
        std_now = std_re0 * (1.0 + eta_pos * max(0.0, c))
    else:
        amp = A_dc_neg * (1.0 - abs(c))
        std_now = max(2.5, std_re0 * (1.0 - eta_neg))  # 降到2.5左右

    # nest.SetStatus(dc_re, {"amplitude": float(amp)})
    # nest.SetStatus(ng_re, {"std": float(std_now)})

def connect_pb(pre, post, p, w, d):
    """概率连接 + 固定延迟（Izhikevich 用电流权重，单位“相对 pA”）"""
    nest.Connect(pre, post,
                 conn_spec={"rule": "pairwise_bernoulli", "p": float(p)},
                 syn_spec={"weight": float(w), "delay": float(d)})

# ---------- 全局参数 ----------
dt     = 0.1     # ms
T_ms   = 25000.0  # 总时长
N_RE   = 120
N_TC   = 120
rng    = 1234

# 刺激（可选）
use_auditory_pulse = True
aud_on   = 10000.0  # ms
aud_dur  = 60.0    # ms
aud_w    = 18.0     # 等效电流脉冲权重（再小再大都可调）

# ====== SO 相位门控参数 ======
so_hz      = 0.8             # SO 频率（0.5–1 Hz 区间选一个）
phi0       = 0.0             # 初相
gate_dt    = 50.0           # 每隔多少 ms 更新一次门控（慢一点即可）
A_dc       = 8.0             # RE 的 DC 门控振幅（pA 量级，6~10 逐步试）
eta_noise  = 0.25            # RE 噪声门控幅度（±15%）
phi_opt    = 0.0             # 目标相位（例如 SO 上相/峰值）
std_re0    = 5.5             # 你的 RE 噪声基线
phi_opt   = 0.0             # 目标相位（把它设为SO峰值相位）
phase_win = math.pi         # 窗宽 180°
A_dc_pos  = 10.0            # 窗内兴奋偏置
A_dc_neg  = 12.0            # 窗外抑制偏置
eta_pos   = 0.35            # 窗内增噪
eta_neg   = 0.80            # 窗外降噪
std_floor = 3.5
std_re0   = 5.5

# ---------- 初始化内核 ----------
nest.ResetKernel()
nest.SetKernelStatus({"resolution": dt, "local_num_threads": 2, "rng_seed": rng})

# ---------- 细胞模型 ----------
# TC：burst 倾向；RE：fast-spiking 倾向
nest.CopyModel("izhikevich", "tc_burst", {"a": 0.02, "b": 0.25, "c": -65.0, "d": 6.0})
nest.CopyModel("izhikevich", "re_fs",    {"a": 0.10, "b": 0.20, "c": -65.0, "d": 2.0})

TC = nest.Create("tc_burst", N_TC)
RE = nest.Create("re_fs",    N_RE)

# 随机初值，避免完全同步或沉寂
randomize_vm(TC)
randomize_vm(RE)

# ---------- 噪声 ----------
# 轻度噪声维持可兴奋态；数值太大会冲掉节律
ng_tc = nest.Create("noise_generator", 1, {"mean": 0.0, "std": 7.0})
ng_re = nest.Create("noise_generator", 1, {"mean": 0.0, "std": 9.0})
nest.Connect(ng_tc, TC, "all_to_all", {"weight": 1.0})
nest.Connect(ng_re, RE, "all_to_all", {"weight": 1.0})

# -----------模拟SO到DC ---------
dc_re = nest.Create("dc_generator", 1, {"amplitude": 0.0})
nest.Connect(dc_re, RE, "all_to_all")  # 直连 RE 全体
# ---------- 丘脑环路 ----------
# 权重量级很关键：如果电压爆到上千 mV，说明权重太大
# TC → RE（兴奋）
connect_pb(TC, RE, p=0.35, w=1.8,  d=2.0)    # 原 p=0.4, w=1.8
connect_pb(RE, TC, p=0.50, w=-2.2, d=1.6)    # 快抑制 略弱
connect_pb(RE, TC, p=0.5, w=-3.8, d=5)
connect_pb(RE, RE, p=0.18, w=-2, d=1.5)    # RE 内抑制稍强（去同步）
# ---------- 可选：听觉脉冲（模拟声→TC） ----------
if use_auditory_pulse:
    pg = nest.Create("poisson_generator", 1, {"rate": 0.0})
    nest.Connect(pg, TC, "all_to_all", {"weight": aud_w, "delay": 1.0})

# ---------- 记录 ----------
mm_tc = nest.Create("multimeter", 1, {"record_from": ["V_m"], "interval": dt})
mm_re = nest.Create("multimeter", 1, {"record_from": ["V_m"], "interval": dt})
nest.Connect(mm_tc, TC[:100])
nest.Connect(mm_re, RE[:100])

# ---------- 模拟 ----------
t = 0.0
while t < T_ms:
    # 在每个段开始，根据当前时间更新门控
    apply_gate(t)

    # 若这一段会跨越听觉刺激窗口，就在中间切一刀
    seg = min(gate_dt, T_ms - t)

    # 如果你有听觉脉冲（aud_on, aud_dur, pg），就判断是否在此段内需要开/关：
    if use_auditory_pulse:
        # 开窗
        if (t < aud_on) and (t + seg >= aud_on):
            # 先跑到 aud_on
            nest.Simulate(aud_on - t)
            nest.SetStatus(pg, {"rate": 6000.0})  # 你可选 3000–4000
            t = aud_on
            continue
        # 关窗
        if (t < aud_on + aud_dur) and (t + seg >= aud_on + aud_dur):
            nest.Simulate(aud_on + aud_dur - t)
            nest.SetStatus(pg, {"rate": 0.0})
            t = aud_on + aud_dur
            continue

    # 正常推进
    nest.Simulate(seg)
    t += seg


# ---------- 提取与绘图 ----------
fs = 1000.0/dt
t_tc, v_tc = lfp_from_mm(mm_tc.get("events"), dt, T_ms)
t_re, v_re = lfp_from_mm(mm_re.get("events"), dt, T_ms)
v_tc_sigma = bandpass(v_tc, fs, 10.0, 16.0)
v_re_sigma = bandpass(v_re, fs, 10.0, 16.0)

plt.figure(figsize=(10,6))
plt.subplot(3,1,1)
plt.plot(t_tc, v_tc, label="TC raw", lw=0.8)
plt.plot(t_re, v_re, label="RE raw", lw=0.8, alpha=0.7)
if use_auditory_pulse:
    plt.axvspan(aud_on, aud_on+aud_dur, color="grey", alpha=0.2)
plt.title("Raw membrane potentials (avg over sampled neurons)")
plt.ylabel("Vm (mV)"); plt.legend()

plt.subplot(3,1,2)
plt.plot(t_tc, v_tc_sigma, lw=1.0)
if use_auditory_pulse:
    plt.axvspan(aud_on, aud_on+aud_dur, color="grey", alpha=0.2)
plt.ylabel("TC 10–16 Hz")

plt.subplot(3,1,3)
plt.plot(t_re, v_re_sigma, lw=1.0)
if use_auditory_pulse:
    plt.axvspan(aud_on, aud_on+aud_dur, color="grey", alpha=0.2)
plt.ylabel("RE 10–16 Hz"); plt.xlabel("Time (ms)")
plt.tight_layout(); plt.show()

print("建议：无纺锤时，依次微调 -> RE→TC 慢抑制权重/延迟、TC→RE 权重、噪声 std、连接概率 p。")


# 提取 σ 波段能量包络
env = np.abs(hilbert(v_tc_sigma))

# 归一化阈值
th = np.mean(env) + 2 * np.std(env)

# 检测峰值区间
above = env > th
spindle_onsets = np.where(np.diff(above.astype(int)) == 1)[0]
spindle_offsets = np.where(np.diff(above.astype(int)) == -1)[0]

print(f"检测到 {len(spindle_onsets)} 段纺锤波活动")
if len(spindle_onsets):
    for i in range(len(spindle_onsets)):
        t1 = t_tc[spindle_onsets[i]]
        t2 = t_tc[spindle_offsets[i]] if i < len(spindle_offsets) else t_tc[-1]
        print(f"  Spindle {i+1}: {t1:.1f}–{t2:.1f} ms, duration = {t2 - t1:.1f} ms")

# ------- 与 tvb功能接通 -------
# ======== NEST→TVB 三元组提取（新增） ========
feats = extract_tc_pulse_features_from_trace(
    t_tc, v_tc,
    stim_on_ms=aud_on,
    baseline_ms=200.0,
    search_lo_ms=40.0,
    search_hi_ms=300.0,
    min_prominence=0.2
)

if feats.get("ok", False):
    print(f"[NEST→TVB] A_TC={feats['A_TC']:.3f}, latency={feats['latency_ms']:.1f} ms, sigma={feats['sigma_ms']:.1f} ms, t_peak={feats['t_peak_ms']:.1f} ms")
else:
    print(f"[NEST→TVB] 提取失败：{feats.get('reason')}")
