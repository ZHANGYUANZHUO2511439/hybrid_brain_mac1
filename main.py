# ===== Phase-locked Hybrid (TVB + NEST) with Visualization =====
# If TVB isn't installed, uses a sine slow-wave placeholder but still runs NEST
# Produces: raster + LFP proxy + (optional) TVB EEG proxy + PSD pre/post

import numpy as np
import nest
from scipy.signal import butter, filtfilt, hilbert, welch
import matplotlib.pyplot as plt

from tvb.simulator.lab import (models, connectivity, coupling, noise,
                               integrators, monitors, simulator, patterns, equations)
import numpy as np
# ----------------------------
# Config
# ----------------------------
CFG = {
    "sim_T_ms": 3000.0,
    "tvb_dt_ms": 1.0,
    "slowwave_f_hz": 0.8,
    "slowwave_noise": 0.05,
    "phase_center_deg": 0.0,       # 0° ~ 上升沿
    "phase_half_deg": 30.0,
    "phase_refractory_ms": 800.0,
    "export_prefix": "drive_tvb",
    "plot_sample_spikes": 2000,    # raster最多画这么多 spike 点，避免太密
    "fs_plot": 1000.0,             # 假设 multimeter interval≈1ms 时的采样率（用于PSD）
    "pre_ms": 1000.0,              # PSD 前窗长度
    "post_ms": 1000.0,             # PSD 后窗长度
    "sigma_band": (12.0, 15.0),    # spindle 频带
}

# ----------------------------
# 0) Slow-wave & phase helpers
# ----------------------------
def try_make_slowwave_from_tvb(total_ms=3000.0, dt_ms=1.0, noise_sigma=0.03):
    jr = models.JansenRit(
        A=np.array([3.25]), a=np.array([30.0]),
        B=np.array([22.0]), b=np.array([15.0]),
        mu=np.array([0.05]), p_min=np.array([0.0]), p_max=np.array([0.0])
    )

    # 单区连接矩阵
    conn = connectivity.Connectivity.from_file(
        "/Users/alex-mhuang/opt/anaconda3/envs/nest310/lib/python3.10/site-packages/tvb_data/connectivity/connectivity_68.zip")
    conn.configure()

    coup = coupling.Linear(a=np.array([0.0]))
    integ = integrators.HeunStochastic(dt=dt_ms,
                                       noise=noise.Additive(nsig=np.array([noise_sigma])))

    mon = (monitors.TemporalAverage(period=dt_ms),)

    sim = simulator.Simulator(model=jr, connectivity=conn, coupling=coup,
                              integrator=integ, monitors=mon)
    sim.configure()

    (time, data), = sim.run(simulation_length=total_ms)
    t = time.squeeze()
    x = data[:, 0, 0]
    x = (x - np.mean(x)) / (np.std(x) + 1e-9)
    return t, x


def make_sine_slowwave(total_ms=3000.0, dt_ms=1.0, freq_hz=0.8, noise_sigma=0.05, seed=7):
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, total_ms, dt_ms)
    x = np.sin(2*np.pi*freq_hz * (t/1000.0)) + rng.normal(0.0, noise_sigma, size=t.size)
    return t, x

def get_slowwave(total_ms, dt_ms, freq_hz, noise_sigma):
    tvb_try = try_make_slowwave_from_tvb(total_ms, dt_ms, noise_sigma*0.6)
    if tvb_try is not None:
        return tvb_try
    return make_sine_slowwave(total_ms, dt_ms, freq_hz, noise_sigma)

def bandpass_hilbert_phase(x, fs_hz, lo=0.5, hi=1.5, order=3):
    nyq = 0.5*fs_hz
    b, a = butter(order, [lo/nyq, hi/nyq], btype="band")
    xf = filtfilt(b, a, x)
    phase = np.angle(hilbert(xf))  # (-pi, pi]
    return xf, phase

def find_phase_triggers(t_ms, phase_rad, center_deg=0.0, half_width_deg=30.0, refractory_ms=800.0):
    center = np.deg2rad(center_deg)
    half = np.deg2rad(half_width_deg)
    def wrap(a): return (a + np.pi) % (2*np.pi) - np.pi
    dist = np.abs(wrap(phase_rad - center))
    in_win = dist <= half
    triggers = []
    last_t = -1e9
    for i in range(1, len(t_ms)):
        if in_win[i] and not in_win[i-1]:
            if t_ms[i] - last_t >= refractory_ms:
                triggers.append(float(t_ms[i]))
                last_t = t_ms[i]
    return triggers

# ----------------------------
# 1) Build NEST network (NEST 3.x)
# ----------------------------
params = {
    "sleep_state": "N2",
    "bg_rate_cortex": 500.0,
    "bg_rate_thalamus": 400.0,
    "re_inh_g_fast": 2.5,
    "re_inh_g_slow": 1.0,
    "tc_exc_g": 1.8,
    "cx_ei_balance": 1.0,
    "aud_burst_rate": 4000.0,
    "aud_burst_ms": 200.0,
    "dt": 0.1
}
N_CXE, N_CXI, N_TC, N_RE = 800, 400, 400, 400

def build_nest_network():
    nest.ResetKernel()
    nest.SetKernelStatus({
        "resolution": params["dt"],
        "rng_seed": 12345,
        "local_num_threads": 2
    })
    base = {
        "E_ex": 0.0, "E_in": -80.0, "E_L": -70.0,
        "V_reset": -70.0, "V_m": -70.0, "V_th": -55.0,
        "tau_syn_ex": 1.5, "tau_syn_in": 10.0,
        "C_m": 200.0, "g_L": 10.0
    }
    neuron_model = "iaf_cond_alpha"
    nest.CopyModel(neuron_model, "cx_e", base)
    nest.CopyModel(neuron_model, "cx_i", base)
    nest.CopyModel(neuron_model, "tc",   base)
    nest.CopyModel(neuron_model, "re",   {**base, "tau_syn_in": 12.0})

    CxE = nest.Create("cx_e", N_CXE)
    CxI = nest.Create("cx_i", N_CXI)
    TC  = nest.Create("tc",   N_TC)
    RE  = nest.Create("re",   N_RE)

    def bg_gen(rate): return nest.Create("poisson_generator", 1, {"rate": rate})
    bg_cx = bg_gen(params["bg_rate_cortex"])
    bg_th = bg_gen(params["bg_rate_thalamus"])
    auditory = nest.Create("poisson_generator", 1, {"rate": 0.0})

    exc      = {"weight": params["tc_exc_g"],        "delay": 1.5}
    inh_fast = {"weight": -params["re_inh_g_fast"],  "delay": 1.5}
    inh_slow = {"weight": -params["re_inh_g_slow"],  "delay": 6.0}

    def connect(pre, post, prob, syn):
        conn = {"rule": "pairwise_bernoulli", "p": prob}
        nest.Connect(pre, post, conn_spec=conn, syn_spec=syn)

    # 背景 & 刺激
    connect(bg_cx, CxE, 1.0, {"weight": 1.0, "delay": 1.0})
    connect(bg_cx, CxI, 1.0, {"weight": 1.0, "delay": 1.0})
    connect(bg_th, TC,  1.0, {"weight": 1.0, "delay": 1.0})
    connect(auditory, TC, 1.0, {"weight": 1.0, "delay": 1.0})

    # 丘脑环路
    connect(TC, RE, 0.2, exc)
    connect(RE, TC, 0.2, inh_fast)
    connect(RE, TC, 0.2, inh_slow)

    # 皮层环路
    ei = params["cx_ei_balance"]
    connect(CxE, CxI, 0.2, {"weight":  1.2*ei, "delay": 1.5})
    connect(CxI, CxE, 0.2, {"weight": -1.5*ei, "delay": 1.5})
    connect(TC,  CxE, 0.2, {"weight":  params["tc_exc_g"], "delay": 2.0})
    connect(CxE, RE, 0.05, {"weight":  0.6, "delay": 2.5})

    # 记录器
    mm_cxe = nest.Create("multimeter", 1, {"record_from": ["V_m"], "interval": params["dt"]})
    mm_tce = nest.Create("multimeter", 1, {"record_from": ["V_m"], "interval": params["dt"]})
    nest.Connect(mm_cxe, CxE[:50])
    nest.Connect(mm_tce, TC[:50])

    sd_all = nest.Create("spike_recorder")
    nest.Connect(CxE + CxI + TC + RE, sd_all)

    return dict(CxE=CxE, CxI=CxI, TC=TC, RE=RE,
                bg_cx=bg_cx, bg_th=bg_th, auditory=auditory,
                mm_cxe=mm_cxe, mm_tce=mm_tce, sd_all=sd_all)

# ----------------------------
# 2) Phase-locked triggers
# ----------------------------
def schedule_phase_locked_triggers(total_ms, tvb_dt_ms, f_hz, noise_sigma,
                                   center_deg, half_deg, refractory_ms):
    t_ms, x = get_slowwave(total_ms, tvb_dt_ms, f_hz, noise_sigma)
    fs_hz = 1000.0 / tvb_dt_ms
    xf, phase = bandpass_hilbert_phase(x, fs_hz, lo=0.5, hi=1.5, order=3)
    triggers = find_phase_triggers(t_ms, phase, center_deg, half_deg, refractory_ms)
    return t_ms, x, xf, phase, triggers

# ----------------------------
# 3) Run NEST with triggers
# ----------------------------
def run_nest_with_triggers(handles, triggers_ms, sim_T):
    auditory = handles["auditory"]
    nest.SetStatus(auditory, [{"rate": 0.0}])
    prev_t = 0.0
    for trig in triggers_ms:
        if trig < prev_t:
            continue
        nest.Simulate(trig - prev_t)
        nest.SetStatus(auditory, [{"rate": params["aud_burst_rate"]}])
        nest.Simulate(params["aud_burst_ms"])
        nest.SetStatus(auditory, [{"rate": 0.0}])
        prev_t = trig + params["aud_burst_ms"]
    if prev_t < sim_T:
        nest.Simulate(sim_T - prev_t)

def make_lfp_proxy_from_multimeter(mm_handle):
    ev = mm_handle.get("events")
    t_all = ev["times"]; v_all = ev["V_m"]
    uniq_t = np.unique(t_all)
    vm_mean = np.empty_like(uniq_t)
    for i, ti in enumerate(uniq_t):
        vm_mean[i] = np.mean(v_all[t_all == ti])
    return uniq_t, vm_mean

def resample_to_tvb_grid(t_ms, x, tvb_dt_ms=1.0, total_ms=None):
    if total_ms is None:
        total_ms = t_ms[-1] - t_ms[0]
    T = int(np.round(total_ms / tvb_dt_ms))
    t0 = t_ms[0]
    t_tvb = np.arange(T) * tvb_dt_ms + t0
    x_tvb = np.interp(t_tvb, t_ms, x)
    x_tvb = (x_tvb - np.mean(x_tvb)) / (np.std(x_tvb) + 1e-9)
    return t_tvb, x_tvb

# ----------------------------
# 4) Optional: run TVB with drive (if TVB already installed)
# ----------------------------
def run_tvb_with_nest_drive(drive_tvb, tvb_dt_ms=1.0,
                            target_nodes=("l_caudalanteriorcingulate", "r_caudalanteriorcingulate"),
                            G=0.0, noise_sigma=0.02, conduction_vel=3.0,
                            gain_drive = 0.002):

    conn = connectivity.Connectivity.from_file("/Users/alex-mhuang/opt/anaconda3/envs/nest310/lib/python3.10/site-packages/tvb_data/connectivity/connectivity_68.zip")
    conn.configure()
    labels = [lb.decode() if isinstance(lb, bytes) else lb for lb in conn.region_labels]
    idx = [labels.index(n) for n in target_nodes if n in labels]
    if len(idx) == 0:
        idx = [0]

    jr = models.JansenRit(
        A=np.array([3.25]),
        a=np.array([100.0]),
        B=np.array([22.0]),
        b=np.array([50.0]),
        mu=np.array([0.22]),
        p_min=np.array([0.1]),
        p_max=np.array([0.2]),
    )

    #  coupling 的增益也用数组
    coup = coupling.Linear(a=np.array([G]))
    no = noise.Additive(nsig=np.array([noise_sigma]))
    integ = integrators.HeunStochastic(
        dt=tvb_dt_ms,
        noise=no)

    drive_tvb = np.asarray(drive_tvb, dtype=np.float64) * gain_drive
    temporal = equations.ExternalEquation(array=drive_tvb,dt=tvb_dt_ms,t0=0.0)

    temporal.parameters["array"] = drive_tvb.astype(np.float64)
    stim = patterns.StimuliRegion(temporal=temporal, connectivity=conn)
    stim.weight = np.zeros((1, conn.number_of_regions))
    stim.weight[0, idx] = 1.0
    stim.configure()

    mon = (monitors.TemporalAverage(period=tvb_dt_ms),)
    sim = simulator.Simulator(model=jr, connectivity=conn, coupling=coup,
                              integrator=integ, stimulus=stim,
                              monitors=mon, conduction_speed=conduction_vel)
    sim.configure()
    (time, data), = sim.run(simulation_length=len(drive_tvb) * tvb_dt_ms)
    eeg_proxy = data[:, :, 0]
    return time.squeeze(), eeg_proxy

# ----------------------------
# 5) Visualization helpers
# ----------------------------
def plot_raster(sd_handle, triggers_ms=None, max_points=2000, title="Spike raster"):
    ev = sd_handle.get("events")
    t = ev["times"]; s = ev["senders"]
    n = t.size
    if n > max_points:
        idx = np.random.default_rng(0).choice(n, size=max_points, replace=False)
        t = t[idx]; s = s[idx]
    plt.figure(figsize=(8, 3))
    plt.scatter(t, s, s=1)
    if triggers_ms:
        for x in triggers_ms:
            plt.axvline(x, linestyle="--", alpha=0.4)
    plt.xlabel("Time (ms)"); plt.ylabel("Neuron ID"); plt.title(title); plt.tight_layout()

def plot_lfp(t, lfp, triggers_ms=None, title="CxE mean Vm (LFP proxy)"):
    plt.figure(figsize=(8, 3))
    plt.plot(t, lfp)
    if triggers_ms:
        for x in triggers_ms:
            plt.axvline(x, linestyle="--", alpha=0.4)
    plt.xlabel("Time (ms)"); plt.ylabel("mV (z-scored if resampled)"); plt.title(title); plt.tight_layout()

def plot_psd_pre_post(t, lfp, first_trigger_ms, pre_ms=1000.0, post_ms=1000.0, fs=1000.0):
    if first_trigger_ms is None:
        print("[warn] no trigger found; skip PSD pre/post.")
        return

        # 选窗（注意：t 是等间隔的话，用这种布尔掩码足够）
    mask_pre = (t > (first_trigger_ms - pre_ms)) & (t <= first_trigger_ms)
    mask_post = (t > first_trigger_ms) & (t <= (first_trigger_ms + post_ms))

    vpre = lfp[mask_pre]
    vpost = lfp[mask_post]

    # 样本量检查
    if vpre.size < 32 or vpost.size < 32:
        print(f"[warn] insufficient samples for PSD windows: pre={vpre.size}, post={vpost.size}")
        return

    # 关键：强制两段使用相同的 nperseg → 保证频率轴一致
    nperseg = int(min(len(vpre), len(vpost), 1024))
    noverlap = nperseg // 2

    f_pre, Ppre = welch(vpre, fs=fs, nperseg=nperseg, noverlap=noverlap)
    f_post, Ppost = welch(vpost, fs=fs, nperseg=nperseg, noverlap=noverlap)

    # 双重保险：如果仍然不同（不应该发生），就对齐到共同频率轴
    if f_pre.shape != f_post.shape or not np.allclose(f_pre, f_post):
        # 以较短的一方为共同频率轴
        n = min(len(f_pre), len(f_post))
        f = f_pre[:n]
        Ppre = Ppre[:n]
        Ppost = Ppost[:n]
    else:
        f = f_pre

    plt.figure(figsize=(6, 3))
    plt.semilogy(f, Ppre, label="pre")
    plt.semilogy(f, Ppost, label="post")
    plt.xlim(0, 30)
    try:
        lo, hi = CFG.get("sigma_band", (12.0, 15.0))
        plt.axvspan(lo, hi, alpha=0.2, label="sigma band")
    except Exception:
        pass
    plt.xlabel("Hz");
    plt.ylabel("PSD")
    plt.title("Pre vs Post (first stimulus)")
    plt.legend()
    plt.tight_layout()


def plot_tvb_eeg(time_ms, eeg_proxy, region_indices=None, triggers_ms=None, title="TVB EEG proxy"):
    if time_ms is None or eeg_proxy is None:
        return
    t = time_ms
    if region_indices is None:
        region_indices = [0]
    plt.figure(figsize=(8,3))
    for idx in region_indices:
        y = eeg_proxy[:, idx]
        y = (y - np.mean(y)) / (np.std(y)+1e-9)
        plt.plot(t, y, label=f"region {idx}")
    if triggers_ms:
        for x in triggers_ms:
            plt.axvline(x, linestyle="--", alpha=0.4)
    plt.xlabel("Time (ms)"); plt.ylabel("z EEG proxy"); plt.title(title); plt.legend(); plt.tight_layout()

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    # (A) phase-locked triggers
    t_ms, tvb_x, tvb_xf, tvb_phase, triggers = schedule_phase_locked_triggers(
        total_ms=CFG["sim_T_ms"], tvb_dt_ms=CFG["tvb_dt_ms"],
        f_hz=CFG["slowwave_f_hz"], noise_sigma=CFG["slowwave_noise"],
        center_deg=CFG["phase_center_deg"], half_deg=CFG["phase_half_deg"],
        refractory_ms=CFG["phase_refractory_ms"]
    )
    print(f"[info] triggers picked: {np.round(triggers,1).tolist()}")

    # (B) NEST
    H = build_nest_network()
    run_nest_with_triggers(H, triggers, sim_T=CFG["sim_T_ms"])

    # (C) LFP proxy from CxE
    t_lfp, lfp = make_lfp_proxy_from_multimeter(H["mm_cxe"])
    # 原始平均Vm
    plot_lfp(t_lfp, lfp, triggers_ms=triggers, title="CxE mean Vm (raw)")

    # 重采样到 1ms，并导出供 TVB 使用
    t_tvb, drive_tvb = resample_to_tvb_grid(t_lfp, lfp, tvb_dt_ms=CFG["tvb_dt_ms"], total_ms=CFG["sim_T_ms"])
    np.save(CFG["export_prefix"] + ".npy", drive_tvb)
    np.savetxt(CFG["export_prefix"] + ".csv", drive_tvb, delimiter=",")
    print(f"[info] exported NEST→TVB drive to: {CFG['export_prefix']}.npy / .csv")

    # 可视化：重采样后的 LFP（更平滑，且 z-score）
    plot_lfp(t_tvb, drive_tvb, triggers_ms=triggers, title="CxE mean Vm (resampled, z-scored)")

    # raster（抽样，以免太密）
    plot_raster(H["sd_all"], triggers_ms=triggers, max_points=CFG["plot_sample_spikes"])

    # PSD 前后对比（用第一刺激时刻为界）
    first_trig = triggers[0] if len(triggers) else None
    plot_psd_pre_post(t_tvb, drive_tvb, first_trig,
                      pre_ms=CFG["pre_ms"], post_ms=CFG["post_ms"], fs=1000.0/CFG["tvb_dt_ms"])

    # (D) TVB（若已安装）
    time_eeg, eeg_proxy = run_tvb_with_nest_drive(drive_tvb, tvb_dt_ms=CFG["tvb_dt_ms"],
                                                  target_nodes=("l_caudalanteriorcingulate", "r_caudalanteriorcingulate"),
                                                  G=0.0001, noise_sigma=0.005, conduction_vel=3.0)
    plot_tvb_eeg(time_eeg, eeg_proxy, region_indices=None, triggers_ms=triggers,
                 title="TVB EEG proxy (if available)")
    #
    plt.show()

