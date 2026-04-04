import numpy as np

P = "/Users/zhangyuanzhuo/PycharmProjects/hybrid_brain_mac1/runs_tvb_be_sigma/scan/sid_201774_be_30.00_s_0.500_gee_0.300_sd_0/be30.00_s0.50_gee0.30_sd0/step_0.npy"
arr = np.load(P, allow_pickle=True)

print("arr.shape =", arr.shape, "dtype =", arr.dtype)

# 遍历所有元素，找出里面是 ndarray 的那些
nd_arrays = []
for i, x in enumerate(arr.flat):
    if isinstance(x, np.ndarray):
        nd_arrays.append((i, x.shape, x.dtype))
    elif isinstance(x, float):
        pass

print("\n[ndarray elements found]:", len(nd_arrays))
for (i, shp, dt) in nd_arrays[:50]:
    print(f"  flat[{i}] -> shape={shp}, dtype={dt}")

# 进一步：取第一个 ndarray 看看内容范围（避免报错，先转 float）
if nd_arrays:
    i0 = nd_arrays[0][0]
    x0 = arr.flat[i0]
    print("\n[first ndarray preview]")
    print("flat index:", i0)
    print("shape:", x0.shape, "dtype:", x0.dtype)
    try:
        x0_float = np.asarray(x0, dtype=float)
        print("min/max:", float(np.min(x0_float)), float(np.max(x0_float)))
    except Exception as e:
        print("min/max failed:", e)
