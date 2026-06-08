import os, sys
sys.path.append("/Users/alex-mhuang/Library/CloudStorage/GoogleDrive-alex.huang0604@gmail.com/My Drive/PythonProject/hybrid_brain_mac1/tvb_adex_ref")
from tvb_adex_ref.tvb_model_reference.simulation_file.parameter.parameter_M_Berlin import Parameter
import tvb_adex_ref.tvb_model_reference.src.tools_simulation as tools
from tvb.simulator.lab import connectivity
import numpy as np

# import os, zipfile
#
# DATA_DIR = "/Users/alex-mhuang/Library/CloudStorage/GoogleDrive-alex.huang0604@gmail.com/マイドライブ/PythonProject/hybird_brain/tvb_adex_ref/tvb_model_reference/data/QL_20120814"
# zip_path = os.path.join(DATA_DIR, "Connectivity.zip")
# print("Connectivity.zip exists:", os.path.exists(zip_path))
#
# with zipfile.ZipFile(zip_path, 'r') as zf:
#     print("Contents of Connectivity.zip:")
#     for n in zf.namelist():
#         print("  -", n)

# 1) 初始化参数
parameters = Parameter()
parameters.parameter_connection_between_region['path'] = \
    "/Users/alex-mhuang/Library/CloudStorage/GoogleDrive-alex.huang0604@gmail.com/My Drive/PythonProject/hybrid_brain_mac1/tvb_adex_ref/tvb_model_reference/data/QL_20120814/"

# 3) 选择状态
state = "synchronous"  # or "asynchronous"
if state == "synchronous":
    parameters.parameter_simulation['path_result'] = './result/DEMO_sync/'
    parameters.parameter_model['b_e'] = 0.0
else:
    parameters.parameter_simulation['path_result'] = './result/DEMO_async/'
    parameters.parameter_model['b_e'] = 0.0

os.makedirs(parameters.parameter_simulation['path_result'], exist_ok=True)

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

# 5) 初始化模拟器
simulator = tools.init(parameters.parameter_simulation,
                       parameters.parameter_model,
                       parameters.parameter_connection_between_region,
                       parameters.parameter_coupling,
                       parameters.parameter_integrator,
                       parameters.parameter_monitor,
                       parameter_stimulation=parameters.parameter_stimulus)

print("[check] N regions =", simulator.connectivity.number_of_regions)
print("[check] weights shape =", simulator.connectivity.weights.shape)
print("[check] tract_lengths shape =", simulator.connectivity.tract_lengths.shape)
print("[check] few labels =", simulator.connectivity.region_labels[:5])
print("\n=== FINAL CHECK AFTER tools.init ===")

print("===================================\n")
# 6) 运行
run_sim = 10000.0  # ms
tools.run_simulation(simulator,
                     run_sim,
                     parameters.parameter_simulation,
                     parameters.parameter_monitor)
print("✅ TVB–AdEx simulation finished.")
