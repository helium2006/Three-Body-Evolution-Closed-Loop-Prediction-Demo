"""
三体进化闭环预测系统
=====================
模拟三体人视角：从有误差的观测数据中预测三体轨道。当预测偏差过大时，
沿梯度方向修正预测模型（反馈），积累历史模型作为元数据，训练"反馈
预测模型"（元学习器）以进化出更优的修正策略。

核心依赖：
  - three_body.py → run_simulation()  作为模拟宇宙的物理规律本体
  - PyTorch       → 神经网络 + 自动微分
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy
import os
from three_body import run_simulation, equilateral_ic


# ==================== 观测噪声（模拟三体人的测量误差） ====================

def add_observation_noise(truth, noise_level, seed=None):
    """给真实轨迹叠加观测噪声。

    Args:
        truth:  (N, 3, 3) 位置或速度的 numpy 数组
        noise_level: <1 时为相对噪声比例，>=1 为绝对标准差
        seed:     随机种子

    Returns:
        noisy:   与 truth 同形的带噪观测数据
    """
    if seed is not None:
        np.random.seed(seed)
    if noise_level < 1.0:
        scale = np.std(truth, axis=0, keepdims=True).clip(min=1e-8)
        noise = np.random.randn(*truth.shape) * noise_level * scale
    else:
        noise = np.random.randn(*truth.shape) * noise_level
    return truth + noise


def generate_observation(gt_data, noise_level, seed=None):
    """从真实轨迹生成带噪观测。"""
    if seed is not None:
        np.random.seed(seed)
    return {
        't':      gt_data['t'].copy(),
        'r_obs':  add_observation_noise(gt_data['r'], noise_level),
        'v_obs':  add_observation_noise(gt_data['v'], noise_level),
        'r_true': gt_data['r'].copy(),
        'v_true': gt_data['v'].copy(),
    }


# ==================== 动力学神经网络 ====================

class DynamicsNetwork(nn.Module):
    """学习三体动力学的神经网络。

    输入:  (batch, 18)  = 3 天体 × 6 状态分量 (rx,ry,rz,vx,vy,vz)
    输出:  (batch, 9)   = 3 天体 × 3 加速度分量 (ax,ay,az)

    三体人对引力定律一无所知，网络从轨迹数据中隐式学习动力学。
    """

    def __init__(self, hidden_dims=(256, 256, 256), dropout=0.0):
        super().__init__()
        self.hidden_dims = hidden_dims
        layers = []
        in_dim = 18
        for h in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.SiLU(),  # Swish 激活函数，比 ReLU 更平滑
                nn.Dropout(dropout),
            ])
            in_dim = h
        layers.append(nn.Linear(in_dim, 9))  # 输出 9 维加速度

        self.net = nn.Sequential(*layers)

        # 残差连接用投影（如果 hidden dim ≠ 18）
        self.proj = nn.Linear(18, 9, bias=False) if hidden_dims else None

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, state):
        """state: (batch, 18) → acceleration: (batch, 9)"""
        # 标准化输入（稳定训练）
        pos = state[:, :9]    # 位置分量
        vel = state[:, 9:]    # 速度分量
        # 简单缩放：位置除以特征尺度
        pos_scale = torch.norm(pos, dim=1, keepdim=True).clamp(min=1.0)
        vel_scale = torch.norm(vel, dim=1, keepdim=True).clamp(min=0.01)
        scaled = torch.cat([pos / pos_scale, vel / vel_scale], dim=1)

        acc = self.net(scaled)
        # 还原尺度：a ∝ 1/r²，缩放 a * pos_scale（近似）
        acc = acc / (pos_scale + 1e-8)
        return acc

    def get_weight_stats(self):
        """提取模型参数的统计特征（供元学习器使用）。"""
        stats = []
        for name, param in self.named_parameters():
            if 'weight' in name:
                w = param.detach()
                stats.extend([
                    w.mean().item(), w.std().item(),
                    w.norm().item(),
                ])
        return np.array(stats, dtype=np.float32)


# ==================== 基于神经网络的预测器 ====================

class NeuralPredictor:
    """使用 DynamicsNetwork + RK4 积分进行轨迹预测。

    符合原 BasePredictor 接口，可直接接入 ClosedLoopEvolution。
    """

    def __init__(self, dynamics_net=None, hidden_dims=(256, 256, 256),
                 learning_rate=0.001, device='cpu', name='NeuralPredictor'):
        self.name = name
        self.device = torch.device(device)
        self.learning_rate = learning_rate

        if dynamics_net is None:
            self.dynamics = DynamicsNetwork(hidden_dims).to(self.device)
        else:
            self.dynamics = dynamics_net.to(self.device)

        self.optimizer = optim.AdamW(self.dynamics.parameters(), lr=learning_rate,
                                      weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=50, T_mult=2)
        self._metadata_history = []

    def _state_to_tensor(self, state):
        """将 numpy state (3,6) 转为 torch tensor (1,18)。"""
        if isinstance(state, torch.Tensor):
            return state.to(self.device)
        return torch.tensor(state.reshape(1, 18), dtype=torch.float32,
                            device=self.device)

    def _rk4_step(self, r, v, dt):
        """单步 RK4 积分，使用网络预测的加速度。"""
        state = torch.cat([r.flatten(1), v.flatten(1)], dim=1)
        batch = state.shape[0]

        # k1
        a1 = self.dynamics(state).view(batch, 3, 3)
        # k2
        r2 = r + 0.5 * dt * v
        v2 = v + 0.5 * dt * a1
        s2 = torch.cat([r2.flatten(1), v2.flatten(1)], dim=1)
        a2 = self.dynamics(s2).view(batch, 3, 3)
        # k3
        r3 = r + 0.5 * dt * v2
        v3 = v + 0.5 * dt * a2
        s3 = torch.cat([r3.flatten(1), v3.flatten(1)], dim=1)
        a3 = self.dynamics(s3).view(batch, 3, 3)
        # k4
        r4 = r + dt * v3
        v4 = v + dt * a3
        s4 = torch.cat([r4.flatten(1), v4.flatten(1)], dim=1)
        a4 = self.dynamics(s4).view(batch, 3, 3)

        r_new = r + dt / 6 * (v + 2 * v2 + 2 * v3 + v4)
        v_new = v + dt / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
        return r_new, v_new

    def predict(self, state, t_future, n_steps=100):
        """从当前状态用 RK4 积出未来轨迹。

        Args:
            state:     (3, 6) numpy 数组
            t_future:  预测时长（标量）
            n_steps:   积分步数

        Returns:
            dict: {'t': (N,), 'r': (N,3,3), 'v': (N,3,3)}
        """
        self.dynamics.eval()
        state_t = self._state_to_tensor(state)  # (1, 18)
        r = state_t[:, :9].view(1, 3, 3)
        v = state_t[:, 9:].view(1, 3, 3)

        if np.isscalar(t_future):
            t_arr = np.linspace(0, t_future, n_steps + 1)
        else:
            t_arr = np.asarray(t_future)
            n_steps = len(t_arr) - 1

        dt = (t_arr[-1] - t_arr[0]) / n_steps

        r_hist = [r.detach().cpu().numpy()[0].copy()]
        v_hist = [v.detach().cpu().numpy()[0].copy()]

        with torch.no_grad():
            for _ in range(n_steps):
                r, v = self._rk4_step(r, v, dt)
                r_hist.append(r.cpu().numpy()[0].copy())
                v_hist.append(v.cpu().numpy()[0].copy())

        return {
            't': t_arr,
            'r': np.array(r_hist),
            'v': np.array(v_hist),
        }

    def feedback(self, prediction, observation, learning_rate=None):
        """沿梯度方向修正动力学网络。

        loss = MSE(预测加速度, 真实加速度)，真实加速度由观测轨迹差分估计。
        """
        if learning_rate is None:
            learning_rate = self.learning_rate

        self.dynamics.train()
        # 从观测数据中提取训练样本
        samples = trajectory_to_samples(
            observation['t'], observation['r'], observation['v'],
            dt=prediction['t'][1] - prediction['t'][0] if len(prediction['t']) > 1 else 10.0
        )

        if len(samples) == 0:
            return {'total_error': 0.0, 'lr': learning_rate, 'n_samples': 0}

        X = torch.tensor(samples['state'], dtype=torch.float32, device=self.device)
        Y = torch.tensor(samples['accel'], dtype=torch.float32, device=self.device)

        dataset = TensorDataset(X, Y)
        loader = DataLoader(dataset, batch_size=256, shuffle=True)

        # 保存反馈前权重快照
        pre_weights = {n: p.detach().clone()
                       for n, p in self.dynamics.named_parameters()}

        total_loss = 0.0
        for group in self.optimizer.param_groups:
            group['lr'] = learning_rate

        for epoch in range(20):  # 反馈时做少量 epoch 微调
            epoch_loss = 0.0
            for xb, yb in loader:
                self.optimizer.zero_grad()
                pred = self.dynamics(xb)
                loss = nn.functional.mse_loss(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.dynamics.parameters(), 10.0)
                self.optimizer.step()
                epoch_loss += loss.item()
            total_loss += epoch_loss / len(loader)

        self.scheduler.step()

        # 计算权重变化量
        weight_delta = {}
        for n, p in self.dynamics.named_parameters():
            weight_delta[n] = (p.detach() - pre_weights[n]).cpu().numpy()

        meta = {
            'model_name': self.name,
            'total_error': float(total_loss / 20),
            'lr': learning_rate,
            'n_samples': len(samples),
            'weight_delta': weight_delta,
            'weight_stats_pre': self.dynamics.get_weight_stats(),
        }
        self._metadata_history.append(meta)
        return meta

    def clone(self):
        """深拷贝（含网络权重）。"""
        cloned = NeuralPredictor(
            dynamics_net=copy.deepcopy(self.dynamics),
            learning_rate=self.learning_rate,
            device=str(self.device),
            name=self.name,
        )
        cloned._metadata_history = list(self._metadata_history)
        return cloned

    def get_metadata(self):
        return {
            'name': self.name,
            'n_params': sum(p.numel() for p in self.dynamics.parameters()),
            'n_feedbacks': len(self._metadata_history),
        }

    def save(self, path, extras=None):
        """保存模型，可选附加额外元数据（如模拟参数）供独立评估脚本使用。"""
        data = {
            'dynamics': self.dynamics.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'metadata': self._metadata_history,
            'hidden_dims': self.dynamics.hidden_dims,
            'learning_rate': self.learning_rate,
        }
        if extras is not None:
            data['extras'] = extras
        torch.save(data, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        hidden_dims = ckpt.get('hidden_dims', (128, 128, 128))
        if hasattr(self.dynamics, 'hidden_dims'):
            pass  # 已初始化，直接加载权重
        self.dynamics.load_state_dict(ckpt['dynamics'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self._metadata_history = ckpt.get('metadata', [])
        return ckpt  # 返回完整内容供外部读取 extras


# ==================== 数据工具 ====================

def trajectory_to_samples(t, r, v, dt=10.0):
    """将轨迹转换为 (state, acceleration) 训练样本。

    加速度由速度的有限差分估计：a_t ≈ (v_{t+dt} - v_t) / dt

    Returns:
        dict: {'state': (M,18), 'accel': (M,9)}
    """
    dt_actual = t[1] - t[0] if len(t) > 1 else dt
    state_list = []
    accel_list = []

    for i in range(len(t) - 1):
        s = np.concatenate([r[i].flatten(), v[i].flatten()])  # (18,)
        a = (v[i + 1] - v[i]) / dt_actual                     # (3,3)
        state_list.append(s)
        accel_list.append(a.flatten())

    return {
        'state': np.array(state_list),
        'accel': np.array(accel_list),
    }


def prepare_training_data(gt_data, dt=10.0):
    """准备初始训练数据（从真实轨迹学动力学）。"""
    return trajectory_to_samples(gt_data['t'], gt_data['r'], gt_data['v'], dt)


# ==================== 元学习器（反馈预测模型） ====================

class MetaLearner(nn.Module):
    """学习预测"何时以及如何修正预测模型"的元模型。

    输入:
      - weight_stats:  网络权重统计特征（均值/标准差/范数）
      - error_stats:   预测误差的统计特征（均值/标准差/分位数）
      - context:       时间上下文（已运行时间 / 总时间，最近误差趋势）

    输出:
      - lr_multipliers: 每个参数组的学习率乘数
      - confidence:     本次预测的置信度 (0~1)
    """

    def __init__(self, weight_dim=9, error_dim=6, hidden_dim=64):
        """
        Args:
            weight_dim: DynamicsNetwork 权重统计向量维数
            error_dim:  误差统计向量维数
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(weight_dim + error_dim + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.lr_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),  # 保证输出 > 0
        )
        self.conf_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, weight_stats, error_stats, context):
        """
        Args:
            weight_stats: (batch, weight_dim) 网络权重统计
            error_stats:  (batch, error_dim)  误差统计
            context:      (batch, 3)  [t_frac, err_trend, feedback_count]

        Returns:
            lr_mult:     (batch, 1) 推荐学习率乘数
            confidence:  (batch, 1) 预测置信度
        """
        x = torch.cat([weight_stats, error_stats, context], dim=1)
        h = self.encoder(x)
        lr_mult = self.lr_head(h)
        confidence = self.conf_head(h)
        return lr_mult, confidence

    def train_on_history(self, history, epochs=100, lr=0.001):
        """用历史反馈数据训练元学习器。

        history: list of dict，每项包含：
          - weight_stats_pre: 反馈前权重统计
          - total_error:      反馈前误差
          - lr:               实际使用的学习率
          - n_samples:        训练样本数
          - context:          (t_frac, err_trend, feedback_idx)
        """
        if len(history) < 5:
            return 0.0  # 数据太少，跳过训练

        X_w = []
        X_e = []
        X_c = []
        Y_lr = []

        for h in history:
            ws = h.get('weight_stats_pre')
            if ws is None or len(ws) == 0:
                continue
            X_w.append(ws)

            # 误差统计：从 total_error 构造伪统计
            err = h['total_error']
            X_e.append([err, err * 0.5, err * 1.5,
                        np.log1p(err), np.sqrt(err), 1.0 / (err + 1e-8)])

            ctx = h.get('context', [0.0, 0.0, 0.0])
            X_c.append(ctx)

            # 目标：学习率乘数（实际 lr / 默认 lr）
            Y_lr.append([h['lr'] / 0.001])

        if len(X_w) < 5:
            return 0.0

        device = next(self.parameters()).device
        X_w = torch.tensor(np.array(X_w), dtype=torch.float32, device=device)
        X_e = torch.tensor(np.array(X_e), dtype=torch.float32, device=device)
        X_c = torch.tensor(np.array(X_c), dtype=torch.float32, device=device)
        Y_lr = torch.tensor(np.array(Y_lr), dtype=torch.float32, device=device)

        dataset = TensorDataset(X_w, X_e, X_c, Y_lr)
        loader = DataLoader(dataset, batch_size=min(32, len(X_w)), shuffle=True)

        opt = optim.Adam(self.parameters(), lr=lr)
        total_loss = 0.0

        for _ in range(epochs):
            for xw, xe, xc, yl in loader:
                opt.zero_grad()
                pred_lr, _ = self(xw, xe, xc)
                loss = nn.functional.mse_loss(pred_lr, yl)
                loss.backward()
                opt.step()
                total_loss += loss.item()

        return total_loss / epochs / max(1, len(loader))

    def suggest_lr(self, weight_stats, error_stats, context):
        """给定当前状态，建议学习率乘数。"""
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            ws = torch.tensor(weight_stats, dtype=torch.float32, device=device).unsqueeze(0)
            es = torch.tensor(error_stats, dtype=torch.float32, device=device).unsqueeze(0)
            ct = torch.tensor(context, dtype=torch.float32, device=device).unsqueeze(0)
            lr_mult, conf = self(ws, es, ct)
        return lr_mult.item(), conf.item()


# ==================== 闭环进化预测主流程 ====================

class ClosedLoopEvolution:
    """闭环进化三体预测系统（在线共进化版）。

    核心设计 —— 模拟宇宙与预测模型同时在线运行：
      1. 模拟宇宙持续向前推进（尊重 three_body.py 的停止时间 T）
      2. 预测模型跟随宇宙演进而实时进化
      3. 每个时间片段：
         a. 预测器从当前状态外推未来轨迹
         b. 宇宙继续运行该片段 → 获得"真实观测"（含噪声）
         c. 对比预测 vs 观测 → 偏差超过阈值则沿梯度修正模型
         d. 保存模型快照与修正元数据
      4. 元数据足够 → 在线训练元学习器，辅助后续反馈决策
      5. 最终评估：用完整模拟检验已进化预测器在全时间轴上的精度
    """

    def __init__(self, ground_truth_params,
                 noise_level=0.01,
                 feedback_threshold=1.0,
                 predict_horizon=500,
                 segment_length=None,
                 n_predict_steps=50,
                 init_frac=0.2,
                 device='cpu'):
        """
        Args:
            ground_truth_params: run_simulation 参数字典（T 为模拟宇宙总停止时间）
            noise_level:         观测噪声等级
            feedback_threshold:  触发反馈的 MSE 偏差阈值
            predict_horizon:     每次预测的时间跨度
            segment_length:      每次推进的时间窗口（默认 = predict_horizon）
            n_predict_steps:     RK4 积分步数
            init_frac:           初始观测窗口占总时间比例（用于预训练）
            device:              'cpu' / 'cuda'
        """
        self.gt_params = ground_truth_params
        self.noise_level = noise_level
        self.feedback_threshold = feedback_threshold
        self.predict_horizon = predict_horizon
        self.segment_length = segment_length if segment_length is not None else predict_horizon
        self.n_predict_steps = n_predict_steps
        self.init_frac = init_frac
        self.device = device

        # 运行时状态
        self.predictor = None
        self.meta_learner = None
        self.model_snapshots = []
        self.feedback_log = []
        self.error_history = []
        self.error_times = []       # 每个片段对应的宇宙时间
        self._init_data = None      # 初始窗口数据（供最终评估使用）

    # ---------- 工具 ----------

    def _params_from_state(self, r, v, T_val):
        """从 (3,3) numpy 数组构造 run_simulation 参数字典。"""
        return {
            'T': float(T_val),
            'm': list(self.gt_params['m']),
            'r': r.tolist(),
            'v': v.tolist(),
        }

    def _make_error_stats(self, err_total, err_r, err_v):
        """构造 6 维误差特征向量。"""
        return [err_total, err_r, err_v,
                np.log1p(err_total), np.sqrt(err_total),
                1.0 / (err_total + 1e-8)]

    def _compute_error_trend(self):
        """最近误差趋势（最近两次的变化量）。"""
        if len(self.error_history) >= 2:
            return self.error_history[-1] - self.error_history[-2]
        return 0.0

    @staticmethod
    def _interp_trajectory(t_src, data, t_tgt):
        """将轨迹数据线性插值到目标时间点。"""
        result = np.zeros((len(t_tgt), 3, 3))
        for body in range(3):
            for comp in range(3):
                result[:, body, comp] = np.interp(
                    t_tgt, t_src, data[:, body, comp],
                    left=data[0, body, comp], right=data[-1, body, comp])
        return result

    # ---------- 训练 ----------

    def _train_dynamics(self, train_data, epochs=100, batch_size=256, verbose=True):
        """用 (state, acceleration) 数据训练动力学网络。"""
        X = torch.tensor(train_data['state'], dtype=torch.float32,
                         device=self.device)
        Y = torch.tensor(train_data['accel'], dtype=torch.float32,
                         device=self.device)

        n = len(X)
        n_train = int(n * 0.8)
        indices = torch.randperm(n)
        X_train, Y_train = X[indices[:n_train]], Y[indices[:n_train]]
        X_val, Y_val = X[indices[n_train:]], Y[indices[n_train:]]

        train_loader = DataLoader(TensorDataset(X_train, Y_train),
                                   batch_size=batch_size, shuffle=True)

        self.predictor.dynamics.train()
        opt = self.predictor.optimizer

        for epoch in range(epochs):
            total_loss = 0.0
            for xb, yb in train_loader:
                opt.zero_grad()
                pred = self.predictor.dynamics(xb)
                loss = nn.functional.mse_loss(pred, yb)
                r_norm = torch.norm(xb[:, :9].view(-1, 3, 3), dim=2).mean()
                acc_norm = torch.norm(pred.view(-1, 3, 3), dim=2).mean()
                physics_penalty = 0.01 * torch.relu(acc_norm * r_norm - 10.0)
                total = loss + physics_penalty
                total.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.predictor.dynamics.parameters(), 10.0)
                opt.step()
                total_loss += loss.item()

            self.predictor.scheduler.step()

            if epoch % 20 == 0 and len(X_val) > 0:
                with torch.no_grad():
                    val_pred = self.predictor.dynamics(X_val[:500])
                    val_loss = nn.functional.mse_loss(val_pred, Y_val[:500])
                if verbose:
                    print(f'  epoch {epoch:4d}  train_loss={total_loss/len(train_loader):.4e}  '
                          f'val_loss={val_loss.item():.4e}')

    # ======================== 主流程 ========================

    def run(self, pretrain_epochs=100, meta_trigger=10, verbose=True,
            progress_callback=None, continue_state=None):
        """执行在线共进化闭环预测。

        模拟宇宙持续运行（不预知未来），预测器跟随宇宙演进而进化。

        Args:
            pretrain_epochs: 初始预训练 epoch 数
            meta_trigger:    元学习器训练的反馈次数阈值
            verbose:         是否打印详细信息
            progress_callback: 可选回调 callable(phase, data)
                phase='init_done':    data={t_current, T, gt_init, obs_init}
                phase='continue_start': data={t_current, T, T_segment}
                phase='segment_done': data={t_current, T, seg_gt, seg_obs, ...}
                phase='done':         data=最终 results 字典
            continue_state:   可选 dict，从上一轮终点继续运行：
                {'t_current', 'r_current', 'v_current', 'predictor', 'meta_learner',
                 'feedback_log', 'model_snapshots', 'error_history', 'error_times'}
        """
        T = self.gt_params['T']
        seg_len = self.segment_length
        t_base = 0.0  # 用于进度计算的基准时间

        if continue_state is not None:
            # ── 继续模式：跳过初始化，使用已有模型从终点继续 ──
            self.predictor = continue_state['predictor']
            self.meta_learner = continue_state['meta_learner']
            t_current = float(continue_state['t_current'])
            r_current = continue_state['r_current'].copy()
            v_current = continue_state['v_current'].copy()
            t_base = t_current
            n_cycles_start = continue_state.get('n_cycles', 1)  # 已完成的周期数

            self.model_snapshots = continue_state.get('model_snapshots', [])
            self.feedback_log = continue_state.get('feedback_log', [])
            self.error_history = continue_state.get('error_history', [])
            self.error_times = continue_state.get('error_times', [])
            self._init_data = continue_state.get('_init_data', None)

            feedback_count = len(self.feedback_log)
            last_meta_train_at = feedback_count  # 避免立即触发元学习器训练

            if verbose:
                print(f'\n=== 继续共进化（从 t={t_current:.0f} 再运行 T={T:.0f}）===')
                print(f'  目标时间={t_base + T:.0f}  预测视野={self.predict_horizon}  单步推进={seg_len}')

            if progress_callback:
                progress_callback('continue_start', {
                    't_current': float(t_current),
                    'T': t_base + T,
                    'T_segment': T,
                })

            segment_num = 0
            # 直接进入阶段 2 循环
        else:
            # ── 全新启动：初始化预测器 & 元学习器 ──
            n_cycles_start = 0  # 尚未完成任何宇宙周期

            self.predictor = NeuralPredictor(
                hidden_dims=(128, 128, 128),
                learning_rate=0.001,
                device=self.device,
            )
            self.meta_learner = MetaLearner(
                weight_dim=len(self.predictor.dynamics.get_weight_stats()),
                error_dim=6,
                hidden_dim=64,
            ).to(self.device)

            self.model_snapshots = []
            self.feedback_log = []
            self.error_history = []
            self.error_times = []

            if verbose:
                print(f'=== 在线共进化闭环预测 ===')
                print(f'  模拟宇宙 T={T}  噪声={self.noise_level}  反馈阈值={self.feedback_threshold}')
                print(f'  预测视野={self.predict_horizon}  单步推进={seg_len}')
                print(f'  初始观测比例={self.init_frac}  设备={self.device}')

            # ═══════════ 阶段 1：初始观测窗口 ═══════════
            T_init = min(T * self.init_frac, max(2000, T * 0.25))
            T_init = min(T_init, T * 0.5)

            if verbose:
                print(f'\n[阶段1] 初始观测窗口 T_init={T_init:.0f} ...')

            init_params = copy.deepcopy(self.gt_params)
            init_params['T'] = T_init
            gt_init = run_simulation(init_params, output_interval=T_init / 100.0)
            obs_init = generate_observation(gt_init, self.noise_level, seed=42)

            train_data = prepare_training_data(gt_init,
                                                dt=obs_init['t'][1] - obs_init['t'][0])
            if verbose:
                print(f'  预训练 {pretrain_epochs} epochs（样本数={len(train_data["state"])}）...')
            self._train_dynamics(train_data, epochs=pretrain_epochs, verbose=verbose)

            t_current = gt_init['t'][-1]
            r_current = gt_init['r'][-1].copy()
            v_current = gt_init['v'][-1].copy()

            self._init_data = {
                'gt': gt_init,
                'obs': obs_init,
                'T_init': T_init,
            }

            if verbose:
                print(f'  初始窗口完成  宇宙时间 t={t_current:.0f}/{T:.0f}')

            if progress_callback:
                progress_callback('init_done', {
                    't_current': float(t_current),
                    'T': T,
                    'gt_init': gt_init,
                    'obs_init': obs_init,
                })

            feedback_count = 0
            last_meta_train_at = 0
            segment_num = 0

        # ═══════════════════════════════════════════════════════
        #  阶段 2：在线共进化（首次运行 or 继续运行共用此循环）
        # ═══════════════════════════════════════════════════════
        T_target = t_base + T

        if verbose and continue_state is None:
            n_segments_est = int((T - t_current) / seg_len) + 1
            print(f'\n[阶段2] 在线共进化（预计 {n_segments_est} 段）...')

        while t_current < T_target - 1e-10:
            segment_num += 1
            remaining = T_target - t_current
            lookahead = min(seg_len, remaining)
            if lookahead < 10:
                break

            t_seg_start = float(t_current)  # 本段起始时刻，供 UI 偏移预测时间轴

            # 2a. 构造当前状态 → 预测器外推未来
            state = np.hstack([r_current.flatten(), v_current.flatten()]).reshape(3, 6)
            pred_horizon = min(self.predict_horizon, remaining)
            prediction = self.predictor.predict(
                state, pred_horizon, n_steps=self.n_predict_steps)

            # 2b. 宇宙继续运行 lookahead 时长
            seg_params = self._params_from_state(r_current, v_current, lookahead)
            seg_gt = run_simulation(seg_params, output_interval=lookahead / 50.0)

            # 2c. 生成带噪观测（每一次观测都有独立的新噪声）
            seg_obs = generate_observation(seg_gt, self.noise_level,
                                            seed=42 + segment_num)

            # 2d. 对比预测 vs 观测
            cmp_t = seg_obs['t']
            cmp_t = cmp_t[cmp_t <= pred_horizon + 1e-10]  # 只看预测视野内
            if len(cmp_t) < 3:
                # 片段太短，跳过比较，直接推进
                r_current = seg_gt['r'][-1].copy()
                v_current = seg_gt['v'][-1].copy()
                t_current += lookahead
                continue

            r_pred = self._interp_trajectory(prediction['t'], prediction['r'], cmp_t)
            v_pred = self._interp_trajectory(prediction['t'], prediction['v'], cmp_t)
            # 匹配观测数据
            cmp_indices = np.searchsorted(seg_obs['t'], cmp_t)
            cmp_indices = np.clip(cmp_indices, 0, len(seg_obs['t']) - 1)
            r_obs_cmp = seg_obs['r_obs'][cmp_indices]
            v_obs_cmp = seg_obs['v_obs'][cmp_indices]

            err_r = np.mean((r_pred - r_obs_cmp) ** 2)
            err_v = np.mean((v_pred - v_obs_cmp) ** 2)
            err_total = float(err_r + err_v)

            self.error_history.append(err_total)
            self.error_times.append(float(t_current))

            # 2e. 判断是否触发反馈
            triggered = err_total > self.feedback_threshold

            if triggered:
                # ── 元学习器辅助决策 ──
                suggested_lr = self.predictor.learning_rate
                meta_conf = 0.0
                if (feedback_count >= meta_trigger
                        and self.meta_learner is not None
                        and (feedback_count - last_meta_train_at) > 0):
                    ws = self.predictor.dynamics.get_weight_stats()
                    es = self._make_error_stats(err_total, err_r, err_v)
                    ctx = [t_current / T,
                           self._compute_error_trend(),
                           float(feedback_count)]
                    suggested_lr, meta_conf = self.meta_learner.suggest_lr(ws, es, ctx)
                    suggested_lr = max(1e-5, min(0.1,
                                                  suggested_lr * self.predictor.learning_rate))

                # 保存快照 + 捕获反馈前权重统计
                snapshot = self.predictor.clone()
                weight_stats_pre = snapshot.dynamics.get_weight_stats()

                # 执行反馈修正
                obs_segment_full = {
                    't': seg_obs['t'],
                    'r': seg_obs['r_obs'],
                    'v': seg_obs['v_obs'],
                }
                pred_interp = {
                    't': seg_obs['t'][:min(len(prediction['t']), len(seg_obs['t']))],
                    'r': self._interp_trajectory(
                        prediction['t'], prediction['r'],
                        seg_obs['t'][:min(len(prediction['t']), len(seg_obs['t']))]),
                    'v': self._interp_trajectory(
                        prediction['t'], prediction['v'],
                        seg_obs['t'][:min(len(prediction['t']), len(seg_obs['t']))]),
                }

                feedback_meta = self.predictor.feedback(
                    pred_interp, obs_segment_full,
                    learning_rate=suggested_lr,
                )
                feedback_meta['meta_conf'] = meta_conf
                feedback_meta['context'] = [t_current / T,
                                             self._compute_error_trend(),
                                             float(feedback_count)]
                feedback_meta['weight_stats_pre'] = weight_stats_pre

                self.model_snapshots.append((snapshot, feedback_meta))
                feedback_count += 1

                self.feedback_log.append({
                    'segment': segment_num,
                    't': float(t_current),
                    'error': err_total,
                    'total_error': err_total,  # MetaLearner 训练需要此字段
                    'threshold': self.feedback_threshold,
                    'meta_lr': suggested_lr,
                    'meta_conf': meta_conf,
                    'lr': suggested_lr,         # MetaLearner 训练需要此字段
                    'n_samples': feedback_meta.get('n_samples', 0),
                    'context': feedback_meta['context'],
                    'weight_stats_pre': feedback_meta['weight_stats_pre'],
                })

                if verbose:
                    tag = ' [元学习]' if meta_conf > 0.5 else ''
                    print(f'  段{segment_num:4d} | t={t_current:8.0f}/{T:.0f} | '
                          f'偏差={err_total:.3e} > 阈值 → 反馈 #{feedback_count}{tag}'
                          f'  lr={suggested_lr:.2e}')

                # ── 在线训练元学习器 ──
                if (feedback_count - last_meta_train_at >= meta_trigger
                        and len(self.feedback_log) >= meta_trigger):
                    meta_loss = self.meta_learner.train_on_history(
                        self.feedback_log, epochs=80, lr=0.001)
                    last_meta_train_at = feedback_count
                    if verbose:
                        print(f'  [元学习器在线训练 #{feedback_count}] loss={meta_loss:.4f}')
            else:
                if verbose and segment_num % 10 == 0:
                    print(f'  段{segment_num:4d} | t={t_current:8.0f}/{T:.0f} | '
                          f'偏差={err_total:.3e} | 正常')

            # 2f. 推进宇宙状态
            r_current = seg_gt['r'][-1].copy()
            v_current = seg_gt['v'][-1].copy()
            t_current += lookahead

            # 清理 GPU 缓存
            if 'cuda' in self.device:
                torch.cuda.empty_cache()

            # 回调：汇报本段结果
            if progress_callback:
                progress_callback('segment_done', {
                    't_current': float(t_current),
                    't_seg_start': t_seg_start,
                    'T': T_target,
                    'seg_gt': seg_gt,
                    'seg_obs': seg_obs,
                    'prediction': prediction,
                    'error': err_total,
                    'feedback': triggered,
                    'segment_num': segment_num,
                    'feedback_count': feedback_count,
                })

        if verbose:
            print(f'\n[阶段2 完] 总段数={segment_num}  反馈次数={feedback_count}'
                  f'  最终 t={t_current:.0f}/{T:.0f}')

        # ═══════════════════════════════════════════════════════
        #  阶段 3：最终评估 —— 重跑完整模拟，检验已进化预测器
        #  在全时间轴上的预测精度
        # ═══════════════════════════════════════════════════════
        if verbose:
            print('\n[阶段3] 最终评估（重跑完整模拟检验预测器）...')

        full_gt = run_simulation(self.gt_params, output_interval=T / 200.0)
        full_obs = generate_observation(full_gt, self.noise_level, seed=99)
        eval_results = self._evaluate(full_gt, full_obs)

        # 元学习器最终精炼
        meta_loss_final = 0.0
        if feedback_count >= meta_trigger:
            meta_loss_final = self.meta_learner.train_on_history(
                self.feedback_log, epochs=150, lr=0.0005)
            if verbose:
                print(f'  元学习器最终训练  loss={meta_loss_final:.4f}')

        if verbose:
            n_snap = len(self.model_snapshots)
            print(f'\n=== 共进化完成 ===（已训练 {n_cycles_start + 1} 个宇宙周期）')
            print(f'  反馈次数={feedback_count}  快照数={n_snap}')
            if len(self.error_history) >= 2:
                ratio = self.error_history[-1] / max(1e-10, self.error_history[0])
                print(f'  初始误差={self.error_history[0]:.3e}  '
                      f'最终误差={self.error_history[-1]:.3e}  '
                      f'变化={ratio:.2f}x')
            if eval_results['r_mse']:
                idx_last = -1 if eval_results['r_mse'][-1] < 1e10 else (
                    np.argmin([x for x in eval_results['r_mse'] if x < 1e5]))
                print(f'  最终 r_MSE={eval_results["r_mse"][idx_last]:.3e}  '
                      f'v_MSE={eval_results["v_mse"][idx_last]:.3e}')

        n_cycles_completed = n_cycles_start + 1

        results = {
            'gt': full_gt,
            'obs': full_obs,
            'init_data': self._init_data,
            'predictor': self.predictor,
            'meta_learner': self.meta_learner,
            'snapshots': self.model_snapshots,
            'feedback_log': self.feedback_log,
            'error_history': self.error_history,
            'error_times': self.error_times,
            'eval_results': eval_results,
            'feedback_threshold': self.feedback_threshold,
            'n_cycles': n_cycles_completed,  # 本模型已经历的宇宙周期数
            # 供"继续"按钮使用的终点状态
            'final_state': {
                't_current': float(t_current),
                'r_current': r_current.copy(),
                'v_current': v_current.copy(),
                'predictor': self.predictor,
                'meta_learner': self.meta_learner,
                'feedback_log': self.feedback_log,
                'model_snapshots': self.model_snapshots,
                'error_history': self.error_history,
                'error_times': self.error_times,
                '_init_data': self._init_data,
                'n_cycles': n_cycles_completed,  # 传递给下一段继续
            },
        }

        if progress_callback:
            progress_callback('done', results)

        return results

    def _evaluate(self, gt, obs):
        """对完整轨迹做多点 rollout 评估（使用带噪观测作为起点）。"""
        n_eval = min(20, len(obs['t']) - 2)
        eval_indices = np.linspace(len(obs['t']) // 10,
                                    len(obs['t']) - 2,
                                    n_eval, dtype=int)
        r_mse_list, v_mse_list = [], []

        for idx in eval_indices:
            state = np.hstack([
                obs['r_obs'][idx].flatten(),
                obs['v_obs'][idx].flatten(),
            ]).reshape(3, 6)
            pred = self.predictor.predict(state, self.predict_horizon,
                                           n_steps=30)
            pred_end_t = obs['t'][idx] + self.predict_horizon
            mask = (obs['t'] >= obs['t'][idx]) & (obs['t'] <= pred_end_t)
            obs_idx = np.where(mask)[0]
            if len(obs_idx) < 2:
                continue
            rp = self._interp_trajectory(pred['t'], pred['r'], obs['t'][obs_idx])
            vp = self._interp_trajectory(pred['t'], pred['v'], obs['t'][obs_idx])
            r_mse_list.append(np.mean((rp - obs['r_obs'][obs_idx]) ** 2))
            v_mse_list.append(np.mean((vp - obs['v_obs'][obs_idx]) ** 2))

        return {
            'r_mse': r_mse_list,
            'v_mse': v_mse_list,
            't_eval': obs['t'][eval_indices[:len(r_mse_list)]],
        }


# ==================== 可视化 ====================

def plot_evolution_results(results, save_path=None):
    """绘制在线共进化闭环预测结果。

    需要 matplotlib，仅在调用时导入。
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('在线共进化闭环预测结果', fontsize=16, fontweight='bold')

    gt = results['gt']
    obs = results['obs']
    T_total = gt['t'][-1]
    init_data = results.get('init_data', {})

    # (0,0): 天体 A 真实轨道 + 初始观测窗口分界线
    ax = axes[0, 0]
    ax.plot(gt['r'][:, 0, 0], gt['r'][:, 0, 1], 'k-', alpha=0.25, lw=0.5,
            label='真实轨迹')
    if init_data:
        T_init = init_data['T_init']
        mask_init = init_data['gt']['t'] <= T_init
        ax.plot(init_data['gt']['r'][mask_init, 0, 0],
                init_data['gt']['r'][mask_init, 0, 1],
                'b-', lw=1.2, label=f'初始窗口 (t≤{T_init:.0f})')
        # 标记初始窗口终点
        if mask_init.sum() > 0:
            idx_end = mask_init.sum() - 1
            ax.plot(init_data['gt']['r'][idx_end, 0, 0],
                    init_data['gt']['r'][idx_end, 0, 1],
                    'o', c='orange', ms=6, label='在线进化开始')
    ax.set_xlabel('X'); ax.set_ylabel('Y')
    ax.set_title('天体 A 轨道 (XY 投影)')
    ax.legend(fontsize=8); ax.set_aspect('equal')

    # (0,1): 预测误差随宇宙时间变化
    ax = axes[0, 1]
    err_times = results.get('error_times', [])
    errs = results['error_history']
    if len(err_times) == len(errs) and len(errs) > 0:
        ax.semilogy(err_times, errs, 'b-', lw=1, alpha=0.7)
    else:
        ax.semilogy(errs, 'b-', lw=1, alpha=0.7)
    # 标注反馈点
    fb_times = [e['t'] for e in results['feedback_log']]
    fb_errs = [e['error'] for e in results['feedback_log']]
    if fb_times:
        ax.scatter(fb_times, fb_errs, c='r', s=40, zorder=5, marker='x',
                   label='反馈触发')
    ax.axhline(results.get('feedback_threshold', 1.0), c='gray', ls='--',
               label='反馈阈值')
    ax.set_xlabel('宇宙时间 t'); ax.set_ylabel('预测 MSE')
    ax.set_title('在线预测误差变化')
    ax.legend(fontsize=8)

    # (0,2): 累计反馈次数随时间变化 + 元学习器参与
    ax = axes[0, 2]
    if fb_times:
        ax.fill_between(fb_times, range(1, len(fb_times) + 1),
                         alpha=0.3, color='orange', step='post')
        ax.step(fb_times, range(1, len(fb_times) + 1), 'r-', where='post',
                lw=1.5, label='累计反馈')
        # 标记元学习器开始参与
        meta_start = results.get('meta_trigger_for_plot',
                                  results.get('feedback_threshold', 0))
        if len(fb_times) > 5:
            ax.axvline(x=fb_times[min(5, len(fb_times) - 1)],
                       c='purple', ls=':', lw=1.2, label='元学习器参与')
    ax.set_xlabel('宇宙时间 t'); ax.set_ylabel('累计反馈次数')
    ax.set_title('反馈累计曲线')
    ax.legend(fontsize=8)

    # (1,0): 元学习器学习率建议 vs 默认值
    ax = axes[1, 0]
    fl = results['feedback_log']
    if fl:
        lrs = [e.get('meta_lr', 0.001) for e in fl]
        confs = [e.get('meta_conf', 0) for e in fl]
        fb_idx = range(1, len(fl) + 1)
        ax.bar(fb_idx, lrs, color='steelblue', alpha=0.7, label='建议学习率')
        ax.axhline(y=0.001, c='gray', ls='--', lw=1, label='默认 lr=0.001')
        ax2 = ax.twinx()
        ax2.plot(fb_idx, confs, 'm-o', ms=5, lw=1.5, label='元学习置信度')
        ax2.set_ylim(-0.05, 1.15)
        ax.set_xlabel('反馈序号'); ax.set_ylabel('学习率')
        ax2.set_ylabel('置信度')
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7)
        ax.set_title('元学习器辅助决策（每次反馈）')

    # (1,1): 最终评估 — 各时间点 rollout 误差
    ax = axes[1, 1]
    if results.get('eval_results'):
        ev = results['eval_results']
        if ev.get('t_eval') is not None and len(ev['t_eval']) > 0:
            ax.semilogy(ev['t_eval'], ev['r_mse'], 'b.-', ms=4, alpha=0.8,
                        label='位置 MSE')
            ax.semilogy(ev['t_eval'], ev['v_mse'], 'r.-', ms=4, alpha=0.8,
                        label='速度 MSE')
    ax.set_xlabel('宇宙时间 t'); ax.set_ylabel('MSE')
    ax.set_title('最终评估（多点 rollout）')
    ax.legend(fontsize=8)

    # (1,2): 进化统计概览
    ax = axes[1, 2]
    n_snaps = len(results['snapshots'])
    n_feedbacks = len(results['feedback_log'])
    n_segments = len(results['error_history'])
    ax.barh(['反馈次数', '快照保存', '总推进段', '总时间 T'],
            [n_feedbacks, n_snaps, n_segments, T_total / 1000],
            color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'])
    ax.set_title('进化统计 (T 单位: 千)')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'图片已保存: {save_path}')
    else:
        plt.show()
    plt.close()


# ==================== 演示入口 ====================

def demo_quick():
    """快速演示：短时间在线共进化。"""
    pos, vel = equilateral_ic(R=10000.0)
    params = {
        'T': 10000.0,
        'm': [1.0, 1.0, 1.0],
        'r': [list(p) for p in pos],
        'v': [list(v) for v in vel],
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    system = ClosedLoopEvolution(
        ground_truth_params=params,
        noise_level=0.02,
        feedback_threshold=0.5,
        predict_horizon=500,
        segment_length=500,
        n_predict_steps=40,
        init_frac=0.2,
        device=device,
    )

    results = system.run(pretrain_epochs=50, meta_trigger=5, verbose=True)

    os.makedirs('checkpoints', exist_ok=True)
    extras = {'gt_params': params, 'noise_level': 0.02, 'feedback_threshold': 0.5,
              'predict_horizon': 500, 'segment_length': 500, 'n_predict_steps': 40,
              'init_frac': 0.2}
    results['predictor'].save('checkpoints/predictor_online.pt', extras)
    torch.save(results['meta_learner'].state_dict(),
               'checkpoints/meta_learner_online.pt')
    print('\n模型已保存到 checkpoints/')

    try:
        plot_evolution_results(results, save_path='evolution_online.png')
    except Exception as e:
        print(f'绘图跳过: {e}')

    return results


def demo_full():
    """完整演示：较长时间在线共进化，充分验证元学习能力。"""
    pos, vel = equilateral_ic(R=10000.0)
    params = {
        'T': 50000.0,
        'm': [1.0, 1.0, 1.0],
        'r': [list(p) for p in pos],
        'v': [list(v) for v in vel],
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'使用设备: {device}')
    if device == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(0)}')

    system = ClosedLoopEvolution(
        ground_truth_params=params,
        noise_level=0.02,
        feedback_threshold=0.5,
        predict_horizon=500,
        segment_length=500,
        n_predict_steps=50,
        init_frac=0.15,
        device=device,
    )

    results = system.run(pretrain_epochs=100, meta_trigger=5, verbose=True)

    os.makedirs('checkpoints', exist_ok=True)
    extras = {'gt_params': params, 'noise_level': 0.02, 'feedback_threshold': 0.5,
              'predict_horizon': 500, 'segment_length': 500, 'n_predict_steps': 50,
              'init_frac': 0.15}
    results['predictor'].save('checkpoints/predictor_online_full.pt', extras)
    torch.save(results['meta_learner'].state_dict(),
               'checkpoints/meta_learner_online_full.pt')

    try:
        plot_evolution_results(results, save_path='evolution_online_full.png')
    except Exception as e:
        print(f'绘图跳过: {e}')

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['quick', 'full'], default='quick')
    args = parser.parse_args()

    if args.mode == 'full':
        demo_full()
    else:
        demo_quick()
