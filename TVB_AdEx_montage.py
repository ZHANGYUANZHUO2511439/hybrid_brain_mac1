# === 1) 初始化 & 运行仿真（沿用你的 Parameter 配置） ===
import matplotlib.pyplot as plt
from tvb_adex_ref.tvb_model_reference.simulation_file.parameter.parameter_M_Berlin import Parameter
import tvb_adex_ref.tvb_model_reference.src.tools_simulation as tools
from tvb.simulator.lab import connectivity
import numpy as np
import os, sys
sys.path.append("/Users/alex-mhuang/Library/CloudStorage/GoogleDrive-alex.huang0604@gmail.com/My Drive/PythonProject/hybrid_brain_mac1/tvb_adex_ref")

# 用你的 Parameter()
parameters = Parameter()
parameters.parameter_connection_between_region['path'] = \
    "/Users/alex-mhuang/Library/CloudStorage/GoogleDrive-alex.huang0604@gmail.com/My Drive/PythonProject/hybrid_brain_mac1/tvb_adex_ref/tvb_model_reference/data/QL_20120814/"
# 同步/异步自己改；这里保持你的 b_e=60（同步倾向）
run_sim = 4000.0  # ms
cut_transient = 2000.0  # ms

# 4) 刺激
n_regions = 68
weight = [0.0] * n_regions
weight[18] = 0.0  # 改成非零可刺激
print(parameters.parameter_connection_between_region['path'])
parameters.parameter_stimulus['onset']    = 500.0
parameters.parameter_stimulus['tau']      = 20.0
parameters.parameter_stimulus['T']        = 1e9
parameters.parameter_stimulus['weights']  = weight
parameters.parameter_stimulus['variables']= [0]

simulator = tools.init(parameters.parameter_simulation,
                       parameters.parameter_model,
                       parameters.parameter_connection_between_region,
                       parameters.parameter_coupling,
                       parameters.parameter_integrator,
                       parameters.parameter_monitor,
                       parameter_stimulation=parameters.parameter_stimulus)

# 跑仿真（如果已在相同 path_result 跑过，可注释掉复用结果）
tools.run_simulation(simulator, run_sim,
                     parameters.parameter_simulation,
                     parameters.parameter_monitor)

# 取结果：result[0] 是 Raw 监视器
result = tools.get_result(parameters.parameter_simulation['path_result'],
                          cut_transient+0.1, run_sim)
times_ms = result[0][0]                  # 时间（ms）
raw = result[0][1]                       # 形状 (T, 8, R)
T, V, R = raw.shape
print("raw shape:", raw.shape)           # e.g. (80000, 8, 68)

# === 2) 取兴奋性放电率并做时间平均，得到每区一个值（Hz） ===
# 变量索引：0=E发放率, 1=I发放率, 2=E方差, 3=协方差, 4=I方差, 5=E适应, 6=I适应, 7=噪声
E_rate = raw[:, 0, :] * 1e3     # KHz -> Hz, shape (T, R)
E_mean = E_rate.mean(axis=0)    # shape (R,)
print("regions:", R, " E_mean range:", E_mean.min(), "->", E_mean.max())

# === 3) 载入表面、RegionMapping，并把区域值贴到皮层顶点 ===
from tvb_adex_ref.tvb_model_reference.view.plot import multiview
from tvb.simulator.lab import cortex as ct
from tvb.simulator.lab import region_mapping as rm
from tvb.simulator.lab import surfaces as surf


conn_path = parameters.parameter_connection_between_region['path']
# RegionMapping.txt 将“顶点 -> 区域索引”映射。注意 dtype=int（替换已弃用的 np.int）
region_mapping_data = np.loadtxt(os.path.join(conn_path, 'RegionMapping.txt'), dtype=int)

# 载入皮层表面网格
surface_cortex = surf.Surface().from_file(os.path.join(conn_path, 'Surface_Cortex.zip'))

# 官方示例里做了 X/Y 交换，这里保持一致，避免左右颠倒
tmp = surface_cortex.vertices[:, 0].copy()
surface_cortex.vertices[:, 0] = surface_cortex.vertices[:, 1]
surface_cortex.vertices[:, 1] = tmp

# 组装 RegionMapping 对象
region_mapping = rm.RegionMapping(array_data=region_mapping_data,
                                  connectivity=simulator.connectivity,
                                  surface=surface_cortex)

# 构造 Cortex 并塞入映射
cortex = ct.Cortex().from_file()
cortex.region_mapping_data = region_mapping

# 为 multiview 准备每个顶点的值：用区域均值在顶点上索引展开
vertex_values = E_mean[region_mapping.array_data]  # shape = 顶点数

# === 4) 画 multiview 大脑着色图 ===
# 半球索引：示例给的是奇偶交替；我们用 connectivity 自带的 labels 更稳妥（若有）
try:
    labels = list(simulator.connectivity.region_labels)
    # 依据标签后缀 _L/_R 自动拆分；若无该后缀，则回退到示例的奇偶方案
    left = np.array([i for i, lb in enumerate(labels) if str(lb).endswith(('_L','_Left','_LH'))])
    right = np.array([i for i, lb in enumerate(labels) if str(lb).endswith(('_R','_Right','_RH'))])
    if len(left)==0 or len(right)==0:
        raise ValueError
    hemispheres_left, hemispheres_right = left, right
except Exception:
    # 回退：与示例一致，偶数为左、奇数为右（按你的 RegionMapping 序列）
    mask = np.array([False,True]* (R//2) + ([False] if R%2 else []))
    hemispheres_right = np.where(mask)[0]
    hemispheres_left  = np.where(~mask)[0]

fig = plt.figure(figsize=(20,20))
multiview(cortex,
          hemispheres_left, hemispheres_right,
          vertex_values,
          fig,
          shaded=False,
          CB_position=[0.8, 0.25, 0.02, 0.5], CB_orientation='vertical')
plt.suptitle('Mean excitatory firing rate (Hz) projected on cortex', y=0.92)
plt.savefig('Plot_BrainSpace_mean_E.png', dpi=200)
plt.show()
