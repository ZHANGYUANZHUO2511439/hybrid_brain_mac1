import nest
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, hilbert
import matplotlib.pyplot as plt

# ========= 连接（概率稀疏连接）=========
def connect(pre, post, prob, syn):
    conn = {"rule": "pairwise_bernoulli", "p": prob}
    nest.Connect(pre, post, conn_spec=conn, syn_spec=syn)

# ========= Poisson 背景（睡眠态可调） =========
def bg_gen(rate):
    return nest.Create("poisson_generator", 1, {"rate": rate})

def bandpass(x, fs_hz, lo, hi, order=2):
    b, a = butter(order, [lo/(fs_hz/2), hi/(fs_hz/2)], btype="band")
    return filtfilt(b, a, x)

def lfp_from_mm(ev, sim_T_ms, dt_ms):
    # 把 multimeter 的 (times, V_m, senders) 对齐到规则时间网格并对所有神经元取平均
    t = np.asarray(ev["times"]); v = np.asarray(ev["V_m"]); s = np.asarray(ev["senders"])
    nb = int(np.floor(sim_T_ms / dt_ms))
    t_grid = np.arange(nb) * dt_ms
    senders_unique = np.unique(s)
    V_stack = np.full((len(senders_unique), nb), np.nan, dtype=float)
    idx_grid = np.clip(np.round(t / dt_ms).astype(int), 0, nb-1)
    for i, nid in enumerate(senders_unique):
        mask = (s == nid)
        V_stack[i, idx_grid[mask]] = v[mask]
    # 前向填充
    for i in range(V_stack.shape[0]):
        row = V_stack[i]
        nz = np.where(~np.isnan(row))[0]
        if nz.size == 0: continue
        first = nz[0]
        row[:first] = row[first]
        for k in range(first+1, nb):
            if np.isnan(row[k]): row[k] = row[k-1]
    lfp = np.nanmean(V_stack, axis=0)
    return t_grid, lfp

def show_kc_spindle_proxies(ev, params, stim_on_ms=None, stim_dur_ms=None):
    t, lfp = lfp_from_mm(ev, params["sim_T"], params["dt"])
    fs = 1000.0 / params["dt"]  # Hz

    # δ/σ 分解
    lfp_delta = bandpass(lfp, fs, 0.5, 4.0)        # K-complex 观测
    lfp_sigma = bandpass(lfp, fs, 11.0, 16.0)      # 纺锤观测
    env_sigma = np.abs(hilbert(lfp_sigma))         # 纺锤包络

    # 画图
    fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(t, lfp, lw=1.0);          ax[0].set_ylabel("mean Vm (mV)"); ax[0].set_title("LFP-like")
    ax[1].plot(t, lfp_delta, lw=1.0);    ax[1].set_ylabel("delta (0.5–4 Hz)")
    ax[2].plot(t, env_sigma, lw=1.0);    ax[2].set_ylabel("sigma envelope"); ax[2].set_xlabel("Time (ms)")

    if stim_on_ms is not None and stim_dur_ms is not None:
        for a in ax:
            a.axvspan(stim_on_ms, stim_on_ms+stim_dur_ms, color="grey", alpha=0.15)
            # KC 典型评估窗：0.3–1.0 s
            a.axvspan(stim_on_ms+300, stim_on_ms+1000, color="orange", alpha=0.08)

    plt.tight_layout(); plt.show()

    # 简单阈值判定（可作为 reward/指标）
    pre_mask  = (t >= max(0, (stim_on_ms or 0) - 200)) & (t < (stim_on_ms or 0))
    win_mask  = (t >= (stim_on_ms or 0) + 300) & (t <= (stim_on_ms or 0) + 1000)
    mu, sd    = np.mean(lfp_delta[pre_mask]), np.std(lfp_delta[pre_mask] + 1e-9)
    kc_flag   = np.min(lfp_delta[win_mask]) < (mu - 2.0*sd)
    sp_gain   = np.mean(env_sigma[win_mask]) / (np.mean(env_sigma[pre_mask]) + 1e-9)
    return {"kc_flag": bool(kc_flag), "spindle_envelope_gain": float(sp_gain)}
def plot_mm_voltage(ev, sim_T_ms, dt_ms, title="LFP-like (mean Vm across CxE)"):
    """
    从 multimeter 提取并绘制前 n_cells 的膜电位随时间变化
    """
    t = np.asarray(ev["times"])
    v = np.asarray(ev["V_m"])
    s = np.asarray(ev["senders"])

    nb = int(np.floor(sim_T_ms / dt_ms))  # 取整，避免越界
    t_grid = np.arange(nb) * dt_ms

    senders_unique = np.unique(s)
    V_stack = np.full((len(senders_unique), nb), np.nan, dtype=float)
    idx_grid = np.clip(np.round(t / dt_ms).astype(int), 0, nb-1)

    for i, nid in enumerate(senders_unique):
        mask = (s == nid)
        V_stack[i, idx_grid[mask]] = v[mask]

    # 前向填充缺失
    for i in range(V_stack.shape[0]):
        row = V_stack[i]
        nz = np.where(~np.isnan(row))[0]
        if nz.size == 0:
            continue
        first = nz[0]
        row[:first] = row[first]
        for k in range(first+1, nb):
            if np.isnan(row[k]):
                row[k] = row[k-1]

    lfp = np.nanmean(V_stack, axis=0)

    plt.figure(figsize=(10,4))
    plt.plot(t_grid, lfp, lw=1.2)
    plt.title(title)
    plt.xlabel("Time (ms)")
    plt.ylabel("Mean membrane potential (mV)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# ========= 简单的RL环境钩子 =========
def get_obs_and_metrics():
    # LFP 近似：取 CxE 前 50 个膜电位均值
    ev = mm_cxe.get("events")
    V = ev["V_m"]; t = ev["times"]
    mask = t > (params["sim_T"] - 1000.0)
    v_last = np.array(V)[mask]
    obs = {
        "cx_mean_vm": float(np.mean(v_last)) if v_last.size else np.nan,
        "cx_min_vm":  float(np.min(v_last)) if v_last.size else np.nan,
        "cx_max_vm":  float(np.max(v_last)) if v_last.size else np.nan,
    }
    k_complex_flag = (v_last.size>0 and obs["cx_min_vm"] < -80.0 and obs["cx_max_vm"] > -55.5)
    return obs, {"k_complex": bool(k_complex_flag)}

def reset_env(personalized_params=None):
    # 可选：根据个性化参数调整
    if personalized_params:
        for k, v in personalized_params.items():
            params[k] = v
    # 软重置：不清空内核与网络，只把刺激源关掉，时间推进0，记录器可选择新建
    nest.Simulate(0.0)                       # 将仿真时间对齐（NEST 3 推荐）
    try:
        nest.SetStatus(auditory, {"rate": 0.0})
    except Exception:
        pass
    return {"note": "soft reset (kept network alive)"}


def step_env(action):
    """
    action 示例：{"aud_start_ms": 1200, "aud_burst_ms": 250, "aud_burst_rate": 5000}
    """
    start = action.get("aud_start_ms",  params["aud_start_ms"])
    dur   = action.get("aud_burst_ms",  params["aud_burst_ms"])
    rate  = action.get("aud_burst_rate", params["aud_burst_rate"])

    # 运行至刺激开始前
    nest.Simulate(start)
    # 刺激窗：提高 auditory rate
    nest.SetStatus(auditory, [{"rate": rate}])
    nest.Simulate(dur)
    # 关闭刺激
    nest.SetStatus(auditory, [{"rate": 0.0}])
    # 运行到结束
    nest.Simulate(params["sim_T"] - start - dur)

    obs, info = get_obs_and_metrics()
    # plot_mm_voltage(mm_cxe.get("events"), sim_T_ms=params["sim_T"], dt_ms=params["dt"])
    metrics = show_kc_spindle_proxies(mm_cxe.get("events"),params,stim_on_ms=params["aud_start_ms"],stim_dur_ms=params["aud_burst_ms"])
    print(metrics)

    reward = (1.0 if info["k_complex"] else 0.0)
    done = True
    return obs, reward, done, info

# ========= 个性化/睡眠状态 参数 =========
params = {
    "sleep_state": "N2",
    "bg_rate_cortex": 100.0,    # Hz
    "bg_rate_thalamus": 500.0,  # Hz
    "re_inh_g_fast": 2.5,       # nS  (conductance-based 权重)
    "re_inh_g_slow": 4.5,       # nS
    "tc_exc_g": 2.1,            # nS
    "cx_ei_balance": 1.0,
    "aud_burst_rate": 8000.0,   # Hz
    "aud_burst_ms": 80.0,      # ms
    "aud_start_ms": 1000.0,     # ms
    "sim_T": 3000.0,            # ms
    "dt": 0.1                   # ms
}

# ========= 神经元数量 =========
N_CXE, N_CXI, N_TC, N_RE = 800, 400, 400, 400

# ========= Kernel =========
nest.ResetKernel()
nest.SetKernelStatus({
    "resolution": params["dt"],
    "rng_seed": 12345,            # 可复现
    "local_num_threads": 2        # 视你的 CPU 调整
})

# ========= 神经元模型与默认参数（电导型 I&F）=========
neuron_model = "iaf_cond_alpha"
base = {
    "E_ex": 0.0,        # mV
    "E_in": -80.0,      # mV
    "E_L": -70.0,       # mV 静息电位（建议显式给）
    "V_reset": -70.0,
    "V_m": -70.0,       # 初始膜电位
    "V_th": -55.0,
    "tau_syn_ex": 1.5,  # ms
    "tau_syn_in": 10.0, # ms
    "C_m": 200.0,       # pF
    "g_L": 10.0         # nS  -> tau_m = C_m/g_L = 20 ms
}
nest.CopyModel(neuron_model, "cx_e", base) # name of the neurons is "cx_e"
nest.CopyModel("aeif_cond_exp", "cx_e_adapt", {
    "E_ex": 0.0, "E_in": -80.0, "E_L": -70.0,
    "V_reset": -70.0, "V_th": -50.0, "Delta_T": 2.0,   # 稍易兴奋
    "tau_syn_ex": 1.5, "tau_syn_in": 10.0,
    "C_m": 200.0, "g_L": 10.0,
    # 适应参数（关键）
    "a": 5.0,       # nS  亚阈适应
    "b": 100.0,      # pA  放电时适应步进
    "tau_w": 400.0  # ms  适应时间常数
})
nest.CopyModel(neuron_model, "cx_i", base)
nest.CopyModel(neuron_model, "tc",   base)
nest.CopyModel(neuron_model, "re",   {**base, "tau_syn_in": 12.0})

# CxE = nest.Create("cx_e", N_CXE) # create the neuron population
CxE = nest.Create("cx_e_adapt", N_CXE)
CxI = nest.Create("cx_i", N_CXI)
TC  = nest.Create("tc",   N_TC)
RE  = nest.Create("re",   N_RE)


bg_cx = bg_gen(params["bg_rate_cortex"])
bg_th = bg_gen(params["bg_rate_thalamus"])
# add random noise to the background
for gen in [bg_cx, bg_th]:
    rate = nest.GetStatus(gen, "rate")[0]
    nest.SetStatus(gen, {"rate": rate * np.random.uniform(0.8, 1.2)})


# ========= 声音刺激：短时加高TC噪声 =========
auditory = nest.Create("poisson_generator", 1, {"rate": 0.0})

# ========= 突触（NEST 3.8：不要放 "model":"static_synapse"）=========
exc      = {"weight": params["tc_exc_g"],        "delay": 1.5}  # nS
inh_fast = {"weight": -params["re_inh_g_fast"],  "delay": 1.5}
inh_slow = {"weight": -params["re_inh_g_slow"],  "delay": 8.0}



# 背景 → 皮层/丘脑（默认 static_synapse）
connect(bg_cx, CxE, 1.0, {"weight": 1.0, "delay": 1.0})
connect(bg_cx, CxI, 1.0, {"weight": 1.0, "delay": 1.0})
connect(bg_th, TC,  1.0, {"weight": 1.0, "delay": 1.0})

# 声音刺激（默认先连上，rate=0，step里再注入）
connect(auditory, TC, 1.0, {"weight": 1.0, "delay": 1.0})

# 丘脑环路
connect(TC, RE, 0.2, exc)          # TC → RE (兴奋)
connect(RE, TC, 0.2, inh_fast)     # RE → TC (快抑制)
connect(RE, TC, 0.2, inh_slow)     # RE → TC (慢抑制，支持纺锤样)

# 皮层环路（E/I）
ei = params["cx_ei_balance"]
connect(CxE, CxI, 0.2, {"weight":  1.2*ei, "delay": 1.5})
connect(CxI, CxE, 0.2, {"weight": -1.5*ei, "delay": 1.5})
connect(TC,  CxE, 0.2, {"weight":  params["tc_exc_g"], "delay": 2.0})
connect(CxE, RE, 0.05, {"weight":  0.6, "delay": 2.5})  # 可选皮层→RE 回馈

# ========= 记录器 =========
mm_cxe = nest.Create("multimeter", 1, {"record_from": ["V_m"], "interval": params["dt"]})
mm_tce = nest.Create("multimeter", 1, {"record_from": ["V_m"], "interval": params["dt"]})

nest.Connect(mm_cxe, CxE[:50])  # 采部分细胞
nest.Connect(mm_tce, TC[:50])

sd_all = nest.Create("spike_recorder")
nest.Connect(CxE + CxI + TC + RE, sd_all)



# ======== 演示：一次 step ========

obs, reward, done, info = step_env({
    "aud_start_ms": 1200,
    "aud_burst_ms": 250,
    "aud_burst_rate": 4500
})
print("OBS:", obs, "REWARD:", reward, "INFO:", info)
