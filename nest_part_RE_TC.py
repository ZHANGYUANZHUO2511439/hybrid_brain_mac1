# NEST 3.x
import nest
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

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

def connect_pb(pre, post, p, w, d):
    """概率连接 + 固定延迟（Izhikevich 用电流权重，单位“相对 pA”）"""
    nest.Connect(pre, post,
                 conn_spec={"rule": "pairwise_bernoulli", "p": float(p)},
                 syn_spec={"weight": float(w), "delay": float(d)})

# ---------- 全局参数 ----------
dt     = 0.1     # ms
T_ms   = 15000.0  # 总时长
N_RE   = 120
N_TC   = 120
rng    = 1234

# 刺激（可选）
use_auditory_pulse = True
aud_on   = 1200.0  # ms
aud_dur  = 60.0    # ms
aud_w    = 8.0     # 等效电流脉冲权重（再小再大都可调）

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
ng_re = nest.Create("noise_generator", 1, {"mean": 0.0, "std": 7.0})
nest.Connect(ng_tc, TC, "all_to_all", {"weight": 1.0})
nest.Connect(ng_re, RE, "all_to_all", {"weight": 1.0})

# ---------- 丘脑环路 ----------
# 权重量级很关键：如果电压爆到上千 mV，说明权重太大
# TC → RE（兴奋）
connect_pb(TC, RE, p=0.4, w=1.8,  d=2.0)   # TC→RE
# RE → TC（抑制）：快 + 慢（慢抑制支持纺锤维持）
connect_pb(RE, TC, p=0.50, w=-2.2, d=1.6)   # RE→TC 快 (GABA_A-like)
connect_pb(RE, TC, p=0.50, w=-3.8, d=8.5)   # RE→TC 慢 (GABA_B-like) —— 关键
# RE 内部少量抑制，避免全场硬同步
connect_pb(RE, RE, p=0.12, w=-0.8, d=1.5)   # RE 内部轻抑制（防止全场硬同步）

# ---------- 可选：听觉脉冲（模拟声→TC） ----------
if use_auditory_pulse:
    pg = nest.Create("poisson_generator", 1, {"rate": 0.0})
    nest.Connect(pg, TC, "all_to_all", {"weight": aud_w, "delay": 1.0})

# ---------- 记录 ----------
mm_tc = nest.Create("multimeter", 1, {"record_from": ["V_m"], "interval": dt})
mm_re = nest.Create("multimeter", 1, {"record_from": ["V_m"], "interval": dt})
nest.Connect(mm_tc, TC[:40])
nest.Connect(mm_re, RE[:40])

# ---------- 模拟 ----------
if use_auditory_pulse:
    # 先到刺激前
    nest.Simulate(aud_on)
    nest.SetStatus(pg, {"rate": 6000.0})
    nest.Simulate(aud_dur)
    nest.SetStatus(pg, {"rate": 0.0})
    # 到结束
    nest.Simulate(T_ms - aud_on - aud_dur)
else:
    nest.Simulate(T_ms)

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
