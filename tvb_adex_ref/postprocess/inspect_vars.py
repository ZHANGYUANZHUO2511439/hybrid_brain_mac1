import numpy as np

P = "/Users/zhangyuanzhuo/PycharmProjects/hybrid_brain_mac1/runs_tvb_be_sigma/scan/sid_201774_be_30.00_s_0.500_gee_0.300_sd_0/be30.00_s0.50_gee0.30_sd0/step_0.npy"
arr = np.load(P, allow_pickle=True)

# 取第一个采样点的 state（就是你看到的 (8,68,1)）
state0 = arr[0, 0, 1]   # shape (8,68,1)

print("state0 shape:", state0.shape)

for vidx in range(state0.shape[0]):
    x = state0[vidx, :, 0]  # (68,)
    print(f"var[{vidx}] min={x.min():.4f}, max={x.max():.4f}, mean={x.mean():.4f}")
