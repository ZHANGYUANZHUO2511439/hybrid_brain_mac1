#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TVB-AdEx + RL 两阶段骨架

Stage 1（宏观层）:
  - 对每个 subject，用简单 1D 搜索标定 b_e*(sid)
  - 其他参数 (b, sigma, g_ee) 固定在一个 baseline 上
  - 目标是匹配个体的 SO 特征 (f_SO, slope_up)

Stage 2（微观层）:
  - 在 env 中固定每个 subject 的 b_e = b_e*(sid)
  - RL 的 action 只输出 (b, sigma, g_ee)
  - state = axis_joint（Age+AHI 联合轴）
  - reward = SO 特征误差（可结合物理约束项）
"""

import os, math, random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from gymnasium import spaces
import gymnasium as gym
from tvb_adex_ref.tvb_model_reference.simulation_file.parameter.parameter_M_Berlin import Parameter
#


# ============ 你要接入的底层 TVB 仿真接口 ============

"""
你需要在另一个文件中实现 tvb_sim_single，比如在 tvb_sim.py 里。

接口建议为：

def tvb_sim_single(
    b_e: float,
    b: float,
    sigma_ou: float,
    g_ee: float,
    sim_dur_s: float = 45.0,
    cut_transient_s: float = 15.0,
    dt_ms: float = 1.0,
    region_idx: int = 5,
    seed: int = 0,
    out_root: str = "tmp_rl_tvb",
) -> Tuple[float, float]:
    # 返回 (f_SO_Hz, slope_up)

这里为了骨架先 from tvb_sim import tvb_sim_single，
你只要保证函数签名一致即可。
"""
from tvb_stim import tvb_sim_single


# ============ 数据容器 ============

@dataclass
class Subject:
    sid: str
    age: float
    ahi: float
    bmi: float
    sex01: int
    axis_joint: float
    cv_period_target: float
    amp_so_target: float




def finite_diff_sensitivity(be, sigma, gee,
                            step={'b_e':1.0,'sigma':0.05,'g_ee':0.05},
                            repeats=1):
    """
    在当前点做一阶有限差分，输出每个参数对 f_SO 与 slope_up 的近似偏导（灵敏度）。
    """
    base = tvb_sim_single(be, sigma, gee)
    base = np.array(base)
    grads = {}
    for name, h in step.items():
        p = dict(b_e=be, sigma_ou=sigma, g_ee=gee)
        p[name if name!='sigma' else 'sigma_ou'] += h
        f1, s1 = tvb_sim_single(p['b_e'], p['sigma_ou'], p['g_ee'])
        df = (f1 - base[0]) / h
        ds = (s1 - base[1]) / h
        grads[name] = (df, ds)
    return base, grads


def load_subjects_from_csv(
    csv_path: str,
    id_col="nsrrid",
    cv_period_col="CV_period",
    amp_so_col="Amp_SO_uv",
    axis_col="Axis_joint_oriented",
    sex_col="sex",
    use_cols=("age", "ahi", "bmi"),
) -> Dict[str, Subject]:
    """
    从聚合特征表加载 subject 数据。
    必须包含 nsrrid, f_SO, slope_up_uvps, Axis_joint_oriented, sex, age, ahi, bmi 等。
    """
    df = pd.read_csv(csv_path)
    need = [id_col, cv_period_col, amp_so_col, axis_col, sex_col] + list(use_cols)
    for c in need:
        if c not in df.columns:
            raise ValueError(f"缺少列: {c}")
    df = df.dropna(subset=[cv_period_col, amp_so_col]).copy()

    def sex_to01(x):
        x = str(x).lower()
        if x in ["m", "male", "1", "男", "man"]:
            return 1
        if x in ["f", "female", "0", "女", "woman"]:
            return 0
        try:
            return int(float(x))
        except:
            return 0

    subj = {}
    for _, r in df.iterrows():
        sid = str(r[id_col])
        subj[sid] = Subject(
            sid=sid,
            age=float(r["age"]) if "age" in df.columns else 50.0,
            ahi=float(r["ahi"]) if "ahi" in df.columns else 5.0,
            bmi=float(r["bmi"]) if "bmi" in df.columns else 24.0,
            sex01=sex_to01(r[sex_col]),
            axis_joint=float(r[axis_col]),
            cv_period_target=float(r[cv_period_col]),
            amp_so_target=float(r[amp_so_col]),
        )
    return subj


def split_subjects(subj: Dict[str, Subject], train_ratio=0.8, seed=2025):
    ids = sorted(subj.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    ntr = int(len(ids) * train_ratio)
    return ids[:ntr], ids[ntr:]


# ============ Stage 1: b_e 标定（宏观层） ============

def calibrate_be_for_subject(
    subj: Subject,
    be_grid: np.ndarray,
    base_b: float,
    base_sigma: float,
    base_g_ee: float,
    feature_weights: Tuple[float, float] = (1.0, 1.5),
    sim_dur_s: float = 30.0,
    cut_transient_s: float = 10.0,
    region_idx: int = 5,
    out_root: str = "tmp_be_calib",
) -> float:
    """
    对单个 subject 做 1D 搜索:
      - 只调 b_e
      - 其他参数固定为 (base_b, base_sigma, base_g_ee)
      - 用 (f_SO, slope_up) 的加权平方差作为损失

    返回最优 b_e*（简单 argmin）
    """

    w_f, w_s = feature_weights
    best_be = None
    best_err = np.inf

    for be in be_grid:
        f_sim, s_sim = tvb_sim_single(
            b_e=float(be),
            sigma_ou=base_sigma,
            g_ee=base_g_ee,
            sim_dur_s=sim_dur_s,
            cut_transient_s=cut_transient_s,
            region_idx=region_idx,
            seed=0,
            out_root=os.path.join(out_root, f"sid_{subj.sid}_be_{be:.2f}"),
        )


        # 允许 tvb_sim_single 有 NaN 返回（仿真失败等）
        if not np.isfinite(f_sim) or not np.isfinite(s_sim):
            continue

        err = (
            w_f * (f_sim - subj.f_target) ** 2
            + w_s * (s_sim - subj.slope_target) ** 2
        )

        if err < best_err:
            best_err = err
            best_be = be

    # 如果全部失败，就退回一个默认值（例如 N3 prior）
    if best_be is None:
        best_be = float(be_grid[len(be_grid) // 2])

    return float(best_be)


def run_stage1_calibration(
    subjects: Dict[str, Subject],
    id_list: Optional[list] = None,
    be_range: Tuple[float, float] = (55.0, 65.0),
    be_num: int = 11,
    base_b: float = 1.0,
    base_sigma: float = 0.3,
    base_g_ee: float = 1.0,
    feature_weights: Tuple[float, float] = (1.0, 1.5),
    out_root: str = "runs_be_calib",
) -> Dict[str, float]:
    """
    对一组 subjects 跑 Stage 1:
      - 生成 b_e 网格
      - 调用 calibrate_be_for_subject
      - 返回 {sid: b_e*}
    """

    os.makedirs(out_root, exist_ok=True)
    if id_list is None:
        id_list = list(subjects.keys())

    be_grid = np.linspace(be_range[0], be_range[1], be_num)
    be_lookup: Dict[str, float] = {}

    logs = []
    for sid in id_list:
        subj = subjects[sid]
        be_star = calibrate_be_for_subject(
            subj,
            be_grid=be_grid,
            base_b=base_b,
            base_sigma=base_sigma,
            base_g_ee=base_g_ee,
            feature_weights=feature_weights,
            sim_dur_s=30.0,
            cut_transient_s=10.0,
            region_idx=5,
            out_root=os.path.join(out_root, "raw"),
        )
        be_lookup[sid] = be_star
        logs.append(
            {
                "sid": sid,
                "axis_joint": subj.axis_joint,
                "b_e_star": be_star,
            }
        )
        print(f"[Stage1] sid={sid} axis={subj.axis_joint:.3f} -> b_e*={be_star:.3f}")

    # 保存标定结果，后续 RL 直接 load 用
    pd.DataFrame(logs).to_csv(
        os.path.join(out_root, "be_calibration_summary.csv"), index=False
    )
    return be_lookup


# ============ Stage 2: RL 环境（b_e 固定） ============

class TVBSOEnv(gym.Env):
    """
    单步环境（contextual bandit）:
      - state = axis_joint (1 维)
      - action = (b, sigma, g_ee)（3 维，归一化为 [-1,1]）
      - 对于每个 subject，b_e 固定为 Stage 1 标定的 b_e*(sid)

    reward:
      r = - [ w_f (f_sim - f_target)^2 + w_s (s_sim - slope_target)^2 ] - 物理惩罚
    """

    metadata = {"render_modes": []}


    def __init__(
        self,
        subjects: Dict[str, Subject],
        id_list: list,
        be_lookup: Dict[str, float],
        action_bounds: Dict[str, Tuple[float, float]] = None,
        feature_weights: Tuple[float, float] = (1.0, 0.01),
        lam_phys: float = 1.0,
        seed: int = 42,
        verbose = False
    ):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.subjects = subjects
        self.id_list = id_list
        self.be_lookup = be_lookup
        self.feature_weights = feature_weights
        self.lam_phys = lam_phys
        self.verbose = verbose

        # 如果有 subject 没有标定结果，可以 fallback 到某个默认值（比如 60）
        self.default_be = 60.0

        # 动作空间：这里只让 RL 动 b, sigma, g_ee 三维
        if action_bounds is None:
            action_bounds = {
                # 这些 range 是示例，你可以根据经验 / 文献再调
                # "b": (0.5, 3.0),
                "sigma": (0.9, 1.1),
                "g_ee": (0.2, 0.5),
            }
        self.bounds = action_bounds
        # self.dim_order = ["b", "sigma", "g_ee"]
        self.dim_order = [ "sigma", "g_ee"]


        # 观测空间：axis_joint 一维
        self.obs_dim = 1
        # 假设 axis_joint 大概在 [-3, 3]，你可以根据实际数据再调
        high_obs = np.array([1.0], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-high_obs, high=high_obs, dtype=np.float32
        )

        # 动作空间：3 维 [-1,1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self.cur_sid = None
        self.cur_subj: Optional[Subject] = None

    # ---- 工具：动作缩放到真实参数空间 ----
    def action_to_params(self, a: np.ndarray) -> Dict[str, float]:
        a = np.clip(a, -1.0, 1.0)
        theta = {}
        for i, k in enumerate(self.dim_order):
            lo, hi = self.bounds[k]
            theta[k] = float((a[i] + 1.0) * 0.5 * (hi - lo) + lo)
        return theta

    # ---- 观测构造：axis only ----
    def _make_obs(self, subj: Subject) -> np.ndarray:
        x = np.array([subj.axis_joint], dtype=np.float32)
        # 如需缩放可以 / 常数，例如 x /= 3.0
        return x

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.cur_sid = random.choice(self.id_list)
        self.cur_subj = self.subjects[self.cur_sid]
        obs = self._make_obs(self.cur_subj)
        return obs, {"sid": self.cur_sid}

    def step(self, action: np.ndarray):
        subj = self.cur_subj

        # 1) 从标定表中取 b_e*
        be_star = self.be_lookup.get(subj.sid, self.default_be)

        # 2) 动作 -> (b, sigma, g_ee)
        theta_micro = self.action_to_params(action)
        # b = theta_micro["b"]
        sigma = theta_micro["sigma"]
        g_ee = theta_micro["g_ee"]

        # 3) 调用底层 TVB 仿真
        f_sim, s_sim = tvb_sim_single(
            b_e=be_star,
            sigma_ou=sigma,
            g_ee=g_ee,
            sim_dur_s=20.0,
            cut_transient_s=5.0,
            region_idx=5,
            seed=0,
            out_root="tmp_rl_tvb"   # 可按需拆目录
        )

        w_f, w_s = self.feature_weights
        err = (
            w_f * (f_sim - subj.f_target) ** 2
            + w_s * (s_sim - subj.slope_target) ** 2
        )
        if self.verbose:
            print("f_sim: ",f_sim,"subj.f_target: ", subj.f_target,
                  "s_sim: ",s_sim, "subj.slope_target: ", subj.slope_target)

        # 4) 物理边界惩罚（可选，轻微惩罚贴边解）
        phys_pen = 0.0
        for k, v in theta_micro.items():
            lo, hi = self.bounds[k]
            margin = (v - lo) / (hi - lo + 1e-9)
            phys_pen += 0.01 * ((margin < 0.02) or (margin > 0.98))

        reward = -float(err) - self.lam_phys * float(phys_pen)
        terminated = True
        truncated = False
        info = {
            "sid": subj.sid,
            "b_e_star": be_star,
            # "b": b,
            "sigma": sigma,
            "g_ee": g_ee,
            "f_sim": f_sim,
            "slope_sim": s_sim,
            "f_tgt": subj.f_target,
            "slope_tgt": subj.slope_target,
            "err": float(err),
        }
        obs = self._make_obs(subj)
        return obs, reward, terminated, truncated, info


# ============ Stage 2: 训练（TD3 / Tiny DDPG） ============

def train_with_sb3(env_train, env_val, total_steps=10000, logdir="runs_td3"):
    try:
        from stable_baselines3 import TD3
        from stable_baselines3.common.noise import NormalActionNoise
        from stable_baselines3.common.monitor import Monitor
    except Exception as e:
        print("⚠️ 未检测到 stable-baselines3，将改用轻量DDPG示例。安装: pip install stable-baselines3[extra]")
        return None

    os.makedirs(logdir, exist_ok=True)
    n_actions = env_train.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.2 * np.ones(n_actions),
    )
    mtrain = Monitor(env_train)
    model = TD3(
        "MlpPolicy",
        mtrain,
        action_noise=action_noise,
        verbose=1,
        learning_rate=1e-3,
        buffer_size=100000,
        batch_size=256,
        train_freq=1,
        gradient_steps=1,
        tensorboard_log=logdir,
    )
    model.learn(total_timesteps=total_steps)

    # 验证：确保 obs 与 sid 一致
    eval_logs = []
    for sid in env_val.id_list:
        env_val.cur_sid = sid
        env_val.cur_subj = env_val.subjects[sid]
        obs = env_val._make_obs(env_val.cur_subj)

        action, _ = model.predict(obs, deterministic=True)
        _, _, _, _, info = env_val.step(action)
        eval_logs.append(info)

    pd.DataFrame(eval_logs).to_csv(
        os.path.join(logdir, "eval_val.csv"), index=False
    )
    return model


# ==== 兜底：极简单步 DDPG（不含 target-net / gamma） ====

class TinyReplay:
    def __init__(self, cap=50000):
        self.S, self.A, self.R = [], [], []
        self.cap = cap

    def add(self, s, a, r):
        self.S.append(s)
        self.A.append(a)
        self.R.append([r])
        if len(self.S) > self.cap:
            self.S = self.S[-self.cap :]
            self.A = self.A[-self.cap :]
            self.R = self.R[-self.cap :]

    def sample(self, bs):
        idx = np.random.choice(len(self.S), size=bs)
        return (
            np.array([self.S[i] for i in idx]),
            np.array([self.A[i] for i in idx]),
            np.array([self.R[i] for i in idx]),
        )


def train_tiny_ddpg(env_train, env_val, steps=20000, logdir="runs_tiny_ddpg",BATCH_SIZE=64,warmup=1000):
    import torch
    import torch.nn as nn
    import torch.optim as optim

    os.makedirs(logdir, exist_ok=True)

    obs_dim = env_train.observation_space.shape[0]
    act_dim = env_train.action_space.shape[0]

    class MLP(nn.Module):
        def __init__(self, inp, outp, act_last=None):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(inp, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, outp),
            )
            self.act_last = act_last

        def forward(self, x):
            y = self.net(x)
            return self.act_last(y) if self.act_last else y

    actor = MLP(obs_dim, act_dim, act_last=nn.Tanh())
    critic = MLP(obs_dim + act_dim, 1)
    aopt = optim.Adam(actor.parameters(), lr=1e-3)
    copt = optim.Adam(critic.parameters(), lr=1e-3)
    mse = nn.MSELoss()

    buf = TinyReplay(10000)

    obs, _ = env_train.reset()
    for t in range(steps):
        # 探索
        with torch.no_grad():
            a = actor(torch.tensor(obs[None, :], dtype=torch.float32)).numpy()[0]
        a = np.clip(a + np.random.normal(0, 0.2, size=act_dim), -1, 1)

        nxt, r, done, trunc, info = env_train.step(a) #        obs, reward, terminated, truncated, info
        buf.add(obs, a, r)
        obs, _ = env_train.reset()

        # 更新
        if t > int(warmup):
            s, a_batch, rw = buf.sample(BATCH_SIZE)
            s = torch.tensor(s, dtype=torch.float32)
            a_batch = torch.tensor(a_batch, dtype=torch.float32)
            rw = torch.tensor(rw, dtype=torch.float32)

            q = critic(torch.cat([s, a_batch], dim=1))
            loss_c = mse(q, rw)
            copt.zero_grad()
            loss_c.backward()
            copt.step()

            a_pred = actor(s)
            q_pred = critic(torch.cat([s, a_pred], dim=1))
            loss_a = -q_pred.mean()
            aopt.zero_grad()
            loss_a.backward()
            aopt.step()

    # 验证：同样修 obs/sid 一致性
    logs = []
    for sid in env_val.id_list:
        env_val.cur_sid = sid
        env_val.cur_subj = env_val.subjects[sid]
        obs = env_val._make_obs(env_val.cur_subj)
        with torch.no_grad():
            a = actor(torch.tensor(obs[None, :], dtype=torch.float32)).numpy()[0]
        _, _, _, _, info = env_val.step(a)
        logs.append(info)

    pd.DataFrame(logs).to_csv(
        os.path.join(logdir, "eval_val.csv"), index=False
    )
    return {"actor": actor, "critic": critic}


# ============ 主入口：串 Stage 1 + Stage 2 ============

def main(fix_be = True):
    input_csv = "./data/shhs_so_features_axis.csv"
    out_root = "runs_rl_tvb"
    os.makedirs(out_root, exist_ok=True)

    subjects = load_subjects_from_csv(input_csv)
    train_ids, val_ids = split_subjects(subjects, train_ratio=0.8, seed=2025)
    # smoke test
    num_smoke_test_train, num_smoke_test_test, repeat,warmup = 100,2,3,10
    train_ids, val_ids = train_ids[:num_smoke_test_train], val_ids[:num_smoke_test_test]

    # ---------------------------------------------------------
    # Stage 1: b_e 标定 或 固定 b_e
    # ---------------------------------------------------------
    if fix_be:
        print("⚠️ 使用固定 b_e=57.0（跳过 Stage1 标定）")
        be_const = 57.0
        be_lookup = {sid: be_const for sid in (train_ids + val_ids)}

        # 保存一个 fake 的标定文件，保持目录一致性
        logs = []
        for sid in (train_ids + val_ids):
            logs.append({
                "sid": sid,
                "axis_joint": subjects[sid].axis_joint,
                "b_e_star": be_const,
            })
        pd.DataFrame(logs).to_csv(
            os.path.join(out_root, "be_calib", "be_calibration_summary.csv"), index=False
        )

    else:
        print("🔍 运行 Stage1：b_e 标定...")
        be_lookup = run_stage1_calibration(
            subjects,
            id_list=train_ids + val_ids,
            be_range=(55.0, 65.0),
            be_num=11,
            base_b=1.0,
            base_sigma=0.3,
            base_g_ee=1.0,
            feature_weights=(1.0, 1.5),
            out_root=os.path.join(out_root, "be_calib"),
        )
    # ------ Stage 2: RL 训练 ------
    env_train = TVBSOEnv(
        subjects, train_ids, be_lookup=be_lookup, seed=2025
    )
    env_val = TVBSOEnv(
        subjects, val_ids, be_lookup=be_lookup, seed=2026,verbose=True
    )

    model = train_with_sb3(
        env_train,
        env_val,
        total_steps=20000,
        logdir=os.path.join(out_root, "td3"),
    )
    if model is None:
        train_tiny_ddpg(
            env_train,
            env_val,
            steps=num_smoke_test_train*repeat,
            logdir=os.path.join(out_root, "tiny_ddpg"),
            warmup = 1
        )

    print("Done.")


if __name__ == "__main__":
    main(fix_be=True)
    #===== this part used as the preliminary test for the sensitvity
    # of the SO features to the tvb model's parameter =======#

    # base, grads = finite_diff_sensitivity(57.0, 1.0, 0.3)
    # print("baseline f, slope:", base)
    # for k, (df, ds) in grads.items():
    #     print(f"{k}: df/d{k}={df:.4f}, ds/d{k}={ds:.4f}")
    # =====#=====#=====#=====#=====#=====#=====#=====#=====
