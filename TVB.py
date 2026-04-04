import inspect
import tvb.simulator.integrators as integrators

print([name for name, cls in inspect.getmembers(integrators, inspect.isclass)])

import numpy as np
import matplotlib.pyplot as plt
from tvb.simulator.lab import models, connectivity, coupling, integrators, monitors, simulator, noise
import inspect

# === 超参数（保守设置，保证稳定）===
DT_MS       = 0.1       # 步长
SIM_MS      = 300.0     # 总时长
G_COUP      = 1e-4       # 全脑耦合增益
NOISE_SIG   = 0.001      # 小噪声
MU_INPUT    = 0.2        # 背景输入
SEED        = 7
CONN_ZIP    = "/Users/alex-mhuang/opt/anaconda3/envs/nest310/lib/python3.10/site-packages/tvb_data/connectivity/connectivity_68.zip"

def run_jr():
    # === 1) 载入连接矩阵 ===
    conn = connectivity.Connectivity.from_file(CONN_ZIP)
    conn.configure()

    # === 2) JR 模型 ===
    jr = models.JansenRit(
        A=np.array([3.25]), a=np.array([100.0]),
        B=np.array([22.0]), b=np.array([50.0]),
        mu=np.array([MU_INPUT]),
        p_min=np.array([0.1]), p_max=np.array([0.2])
    )

    # === 3) 初始条件 ===
    rng = np.random.default_rng(SEED)
    ic = 1e-3 * rng.standard_normal((jr.nvar, 1, conn.number_of_regions))
    jr.initial_conditions = ic

    # === 4) 耦合 + 积分器 ===
    coup = coupling.Linear(a=np.array([G_COUP]))
    integ = integrators.HeunStochastic(dt=DT_MS, noise=noise.Additive(nsig=np.array([NOISE_SIG])))

    # === 5) 监视器 ===
    mon = (monitors.Raw(period=DT_MS),)

    # === 6) 配置模拟器 ===
    sim = simulator.Simulator(model=jr, connectivity=conn, coupling=coup,
                              integrator=integ, monitors=mon)
    sim.configure()
    (raw_time, raw_data), = sim.run(simulation_length=SIM_MS)

    # === 7) EEG proxy: y1 - y2 ===
    y1 = raw_data[:, 1, :, 0]
    y2 = raw_data[:, 2, :, 0]
    eeg_proxy = y1 - y2

    labels = [lb.decode() if isinstance(lb, bytes) else lb for lb in conn.region_labels]
    idx_plot = labels.index("r_superiorfrontal") if "r_superiorfrontal" in labels else 0

    t = raw_time.squeeze()
    y = eeg_proxy[:, idx_plot]
    y = (y - y.mean()) / (y.std() + 1e-9)

    plt.figure(figsize=(9,3))
    plt.plot(t, y, lw=1)
    plt.title(f"JR model EEG proxy ({labels[idx_plot]})")
    plt.xlabel("Time (ms)"); plt.ylabel("z-scored EEG proxy")
    plt.tight_layout(); plt.show()

    return t, y



def run_adex():
    # === 1) 载入连接矩阵 ===
    conn = connectivity.Connectivity.from_file(CONN_ZIP)
    conn.configure()

    # === 2) AdEx mean-field 模型 ===
    adex = models.AdExMeanfield(
        g=np.array([1e-4]),        # 耦合强度（全脑交互）
        T=np.array([0.02]),        # 膜时间常数
        tau_w=np.array([500.0]),   # 适应时间常数 (ms)
        a=np.array([0.0]),         # 次阈电导
        b=np.array([60.0]),        # spike-triggered 适应 (关键参数)
        Vthr=np.array([-50.0]),    # 阈值
        Iext=np.array([MU_INPUT])  # 外部输入
    )

    rng = np.random.default_rng(SEED)
    ic = 1e-3 * rng.standard_normal((adex.nvar, 1, conn.number_of_regions))
    adex.initial_conditions = ic

    coup = coupling.Linear(a=np.array([G_COUP]))
    integ = integrators.HeunStochastic(dt=DT_MS, noise=noise.Additive(nsig=np.array([NOISE_SIG])))
    mon = (monitors.Raw(period=DT_MS),)

    sim = simulator.Simulator(model=adex, connectivity=conn, coupling=coup,
                              integrator=integ, monitors=mon)
    sim.configure()
    (raw_time, raw_data), = sim.run(simulation_length=SIM_MS)

    # AdEx 输出的变量： firing rate / adaptation / voltage proxy
    fr = raw_data[:, 0, :, 0]   # firing rate 作为 EEG proxy

    labels = [lb.decode() if isinstance(lb, bytes) else lb for lb in conn.region_labels]
    idx_plot = labels.index("r_superiorfrontal") if "r_superiorfrontal" in labels else 0

    t = raw_time.squeeze()
    y = fr[:, idx_plot]
    y = (y - y.mean()) / (y.std() + 1e-9)

    plt.figure(figsize=(9,3))
    plt.plot(t, y, lw=1)
    plt.title(f"AdEx mean-field firing rate ({labels[idx_plot]})")
    plt.xlabel("Time (ms)"); plt.ylabel("z-scored firing rate")
    plt.tight_layout(); plt.show()

    return t, y

if __name__ == "__main__":
    # JR
    # t_jr, y_jr = run_jr()

    # AdEx
    t_adex, y_adex = run_adex()
