import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# 1. 数据路径
# ============================================================

# ------------------------------------------------------------
# baseline
# ------------------------------------------------------------
baseline_dir = Path(
    "/Users/zhangyuanzhuo/PycharmProjects/hybrid_brain_mac1/runs_tvb_be_sigma/baseline_240s_cut_20/sid_0_be_60.00_s_0.500_gee_0.400_sd_1/be60.00_s0.50_gee0.40_sd1"
)

# ------------------------------------------------------------
# up-phase stimulation
# ------------------------------------------------------------
up_dir = Path(
    "/Users/zhangyuanzhuo/PycharmProjects/hybrid_brain_mac1/runs_tvb_be_sigma/weight_0.00039_in_tools_up_side/sid_0_be_60.00_s_0.500_gee_0.400_sd_1/be60.00_s0.50_gee0.40_sd1"
)

# ------------------------------------------------------------
# down-phase stimulation
# ------------------------------------------------------------
down_dir = Path(
    "/Users/zhangyuanzhuo/PycharmProjects/hybrid_brain_mac1/runs_tvb_be_sigma/weight_0.00039_in_tools_simulation_down_side/sid_0_be_60.00_s_0.500_gee_0.400_sd_1/be60.00_s0.50_gee0.40_sd1"
)

# ============================================================
# 2. duty cycle函数
# ============================================================

def compute_duty_cycle(data_dir):

    # --------------------------------------------------------
    # load
    # --------------------------------------------------------
    rateE = np.load(data_dir / "rateE.npy")
    time = np.load(data_dir / "time.npy") / 1000.0

    print("\n===================================")
    print("loading:", data_dir)

    # --------------------------------------------------------
    # 去掉transient
    # --------------------------------------------------------
    cut_start = 10

    mask = time >= cut_start

    time = time[mask]
    rateE = rateE[mask]

    # --------------------------------------------------------
    # 全脑平均
    # --------------------------------------------------------
    x = rateE.mean(axis=1)

    # --------------------------------------------------------
    # 去均值
    # --------------------------------------------------------
    x = x - np.mean(x)

    # --------------------------------------------------------
    # threshold
    # --------------------------------------------------------
    threshold = 0.0

    # --------------------------------------------------------
    # up-state
    # --------------------------------------------------------
    up_state = x > threshold

    # --------------------------------------------------------
    # duty cycle
    # --------------------------------------------------------
    duty_cycle = np.mean(up_state)

    # --------------------------------------------------------
    # up/down duration
    # --------------------------------------------------------
    dt = np.median(np.diff(time))

    up_duration = np.sum(up_state) * dt

    down_duration = np.sum(~up_state) * dt

    print("Duty cycle   =", duty_cycle)
    print("Up duration  =", up_duration, "s")
    print("Down duration=", down_duration, "s")

    return (
        duty_cycle,
        up_duration,
        down_duration,
        x,
        time,
        up_state
    )

# ============================================================
# 3. compute
# ============================================================

(
    duty_base,
    up_base,
    down_base,
    x_base,
    time_base,
    state_base
) = compute_duty_cycle(baseline_dir)

(
    duty_up,
    up_up,
    down_up,
    x_up,
    time_up,
    state_up
) = compute_duty_cycle(up_dir)

(
    duty_down,
    up_down,
    down_down,
    x_down,
    time_down,
    state_down
) = compute_duty_cycle(down_dir)

# ============================================================
# 4. print summary
# ============================================================

print("\n===================================")
print("SUMMARY")
print("===================================")

print("\nBaseline")
print("Duty cycle =", duty_base)

print("\nUp-phase stimulation")
print("Duty cycle =", duty_up)

print("\nDown-phase stimulation")
print("Duty cycle =", duty_down)

# ============================================================
# 5. bar plot
# ============================================================

labels = [
    "Baseline",
    "Up-phase",
    "Down-phase"
]

values = [
    duty_base,
    duty_up,
    duty_down
]

plt.figure(figsize=(8,6))

bars = plt.bar(
    labels,
    values
)

# ============================================================
# 数值显示
# ============================================================

for bar, value in zip(bars, values):

    plt.text(
        bar.get_x() + bar.get_width()/2,
        value + 0.01,
        f"{value:.3f}",
        ha='center',
        fontsize=12
    )

# ============================================================
# beauty
# ============================================================

plt.ylabel("Duty Cycle", fontsize=16)

plt.title(
    "Duty Cycle Comparison",
    fontsize=20
)

plt.ylim(0, 1)

plt.tight_layout()

# ============================================================
# 6. save
# ============================================================

save_dir = Path("analysis/figures")

save_dir.mkdir(parents=True, exist_ok=True)

save_path = save_dir / "duty_cycle_comparison.png"

plt.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight"
)

print("\nsaved to:")
print(save_path)

plt.show()

# ============================================================
# 7. representative trace（可选）
# ============================================================

plt.figure(figsize=(14,4))

# 只展示前20秒
mask_plot = time_base <= 30

plt.plot(
    time_base[mask_plot],
    x_base[mask_plot],
    linewidth=2,
    label="Global mean"
)

plt.axhline(
    0,
    linestyle="--",
    linewidth=2,
    label="Threshold"
)

# up-state区域
plt.fill_between(
    time_base[mask_plot],
    x_base[mask_plot],
    0,
    where=state_base[mask_plot],
    alpha=0.3,
    label="Up-state"
)

plt.xlabel("Time [s]", fontsize=14)

plt.ylabel("Activity", fontsize=14)

plt.title(
    "Representative Up/Down State",
    fontsize=18
)

plt.legend()

plt.tight_layout()

plt.show()