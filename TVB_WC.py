import numpy as np
import matplotlib.pyplot as plt
from tvb.simulator.lab import models, connectivity, coupling, noise, integrators, monitors, simulator

# ---------- 参数（保守稳定） ----------
DT_MS    = 0.05        # 小步长更稳
SIM_MS   = 3000.0
G_COUP   = 1e-4        # 很小的全脑耦合
NOISE_SD = 0.002       # 轻微噪声
SEED     = 7

# 68区连接矩阵（确保路径正确）
CONN_ZIP = "/Users/alex-mhuang/opt/anaconda3/envs/nest310/lib/python3.10/site-packages/tvb_data/connectivity/connectivity_68.zip"
conn = connectivity.Connectivity.from_file(CONN_ZIP)
conn.configure()

# ---------- Wilson–Cowan 模型 ----------
# 模型
wc = models.WilsonCowan(
    tau_e=np.array([0.02]), tau_i=np.array([0.01]),
    c_ee=np.array([12.0]), c_ei=np.array([10.0]),
    c_ie=np.array([10.0]), c_ii=np.array([ 2.0]),
    P=np.array([0.7]), Q=np.array([0.4]),
    a_e=np.array([1.2]), b_e=np.array([2.8]), theta_e=np.array([4.0]), c_e=np.array([1.0]),
    a_i=np.array([1.0]), b_i=np.array([4.0]), theta_i=np.array([3.7]), c_i=np.array([1.0]),
)
wc.variables_of_interest = ("E", "I")
# wc.state_variable_range = np.array([[0.0, 1.0],
#                                     [0.0, 1.0]])

# 初值
rng = np.random.default_rng(7)
ic = 1e-3 * rng.standard_normal((wc.nvar, 1, conn.number_of_regions))
wc.initial_conditions = ic

# 耦合 + 积分器 + 监视器
coup = coupling.Linear(a=np.array([1e-4]))
integ = integrators.HeunStochastic(dt=0.1, noise=noise.Additive(nsig=np.array([0.001])))
mon = (monitors.Raw(period=1000.0),)

# 仿真
sim = simulator.Simulator(model=wc, connectivity=conn, coupling=coup,
                          integrator=integ, monitors=mon)
sim.configure()
(raw_time, raw_data), = sim.run(simulation_length=3000.0)

# 保险：裁剪
raw_data = np.clip(raw_data, 0.0, 1.0)

# 取 E/I
E = raw_data[:, 0, :, 0]
I = raw_data[:, 1, :, 0]
eeg_proxy = E - I

print("raw_data shape:", raw_data.shape)  # 期望 (T, 2, 68, 1)

# 选一个区域画图（比如右侧 superior frontal；若不存在就画第0区）
labels = [lb.decode() if isinstance(lb, bytes) else lb for lb in conn.region_labels]
try:
    idx_plot = labels.index("r_superiorfrontal")
except ValueError:
    idx_plot = 0

t = raw_time.squeeze()
y = eeg_proxy[:, idx_plot]
y = (y - y.mean()) / (y.std() + 1e-9)

plt.figure(figsize=(9,3))
plt.plot(t, y, lw=1)
plt.title(f"Wilson–Cowan EEG proxy (region={labels[idx_plot]})")
plt.xlabel("Time (ms)"); plt.ylabel("z")
plt.tight_layout(); plt.show()
