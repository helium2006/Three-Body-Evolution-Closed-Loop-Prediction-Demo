"""
三体问题物理引擎与数值模拟
===========================
提供等边三角形特解、辛积分器、自适应步长等核心物理计算。
不包含任何 UI 代码 —— UI 面板请见 ui_panels.py。
"""

import sys
import numpy as np
import torch


# ==================== 等边三角形特解计算 ====================

def equilateral_ic(R):
    """等边三角形特解：返回 (positions, velocities)
    positions: ((xA,yA,zA), (xB,yB,zB), (xC,yC,zC))
    velocities: ((vxA,vyA,vzA), (vxB,vyB,vzB), (vxC,vyC,vzC))
    """
    rA = (0.0, 0.0, 0.0)
    rB = (R, 0.0, 0.0)
    rC = (R / 2.0, R * np.sqrt(3) / 2.0, 0.0)

    inv_sqrt_R = 1.0 / np.sqrt(R)
    vA = (0.5 * inv_sqrt_R, -np.sqrt(3) / 2 * inv_sqrt_R, 0.0)
    vB = (0.5 * inv_sqrt_R,  np.sqrt(3) / 2 * inv_sqrt_R, 0.0)
    vC = (-inv_sqrt_R, 0.0, 0.0)

    return (rA, rB, rC), (vA, vB, vC)


def figure8_ic(scale=1.0, m=1.0):
    """8字形特解（Chenciner-Montgomery 周期轨道）。

    参考: Chenciner & Montgomery (2000), Simó (2002)
    标准归一化初值（G=1, m₁=m₂=m₃=1, 周期≈6.3259）。

    Args:
        scale: 空间尺度缩放因子。位置按 scale 线性缩放，
               速度按 sqrt(m/scale) 缩放以保持轨道形状。
        m:     单天体质量（三天体必须等质量）。

    Returns:
        (positions, velocities)
        positions: ((xA,yA,zA), (xB,yB,zB), (xC,yC,zC))
        velocities: ((vxA,vyA,vzA), (vxB,vyB,vzB), (vxC,vyC,vzC))
    """
    # 标准归一化初值（平面运动，z=0）
    x1, y1 = -0.97000436,  0.24308753
    x2, y2 =  0.97000436, -0.24308753
    x3, y3 =  0.0,         0.0

    vx1, vy1 =  0.4662036850,  0.4323657300
    vx2, vy2 =  0.4662036850,  0.4323657300
    vx3, vy3 = -0.9324073700, -0.8647314600

    pos_s = scale
    vel_s = np.sqrt(m / max(scale, 1e-10))

    positions = (
        (x1 * pos_s, y1 * pos_s, 0.0),
        (x2 * pos_s, y2 * pos_s, 0.0),
        (x3 * pos_s, y3 * pos_s, 0.0),
    )
    velocities = (
        (vx1 * vel_s, vy1 * vel_s, 0.0),
        (vx2 * vel_s, vy2 * vel_s, 0.0),
        (vx3 * vel_s, vy3 * vel_s, 0.0),
    )
    return positions, velocities


# ==================== 物理引擎 ====================

def compute_accelerations(r, G, m):
    """向量化计算三个天体的加速度（PyTorch + CUDA）
    r: (3, 3)  G: scalar  m: (3,)  返回 a: (3, 3)
    """
    diff = r.unsqueeze(0) - r.unsqueeze(1)
    dist_sq = (diff * diff).sum(dim=2)
    dist = dist_sq.sqrt()
    mask = dist > 0
    force = torch.zeros((3, 3, 3), dtype=r.dtype, device=r.device)
    force[mask] = G * (m.unsqueeze(1).unsqueeze(2) * m.unsqueeze(0).unsqueeze(2) *
                       diff / dist.unsqueeze(2).pow(3))[mask]
    a = force.sum(dim=1) / m.unsqueeze(1)
    return a


def yoshida4_step(r, v, G, m, dt):
    """4 阶 Yoshida 辛积分器（O(dt⁵) 精度，保能量）"""
    x1 = 1.3512071919596578
    x0 = -1.7024143839193156

    r = r + x1 * dt / 2 * v
    v = v + x1 * dt * compute_accelerations(r, G, m)
    r = r + x1 * dt / 2 * v

    r = r + x0 * dt / 2 * v
    v = v + x0 * dt * compute_accelerations(r, G, m)
    r = r + x0 * dt / 2 * v

    r = r + x1 * dt / 2 * v
    v = v + x1 * dt * compute_accelerations(r, G, m)
    r = r + x1 * dt / 2 * v

    return r, v


def compute_adaptive_dt(r, v, dt_min, dt_max, eta=0.02):
    """基于天体间最小穿越时间自动调整积分步长。"""
    diff = r.unsqueeze(0) - r.unsqueeze(1)
    dist = torch.norm(diff, dim=2)
    v_diff = v.unsqueeze(0) - v.unsqueeze(1)
    v_rel = torch.norm(v_diff, dim=2)

    mask = dist > 0
    if mask.sum() == 0:
        return dt_max
    crossing_time = dist[mask] / (v_rel[mask] + 1e-10)
    dt = eta * crossing_time.min().item()
    return max(dt_min, min(dt, dt_max))


# ==================== 独立模拟接口 ====================

def run_simulation(params, output_interval=None, dt_min=None, dt_max=None,
                   eta=0.02, progress_callback=None, step_callback=None,
                   stop_flag=None):
    """运行三体数值模拟，返回轨迹数据。

    Args:
        params: dict with 'T', 'm'[3], 'r'[3][3], 'v'[3][3]
        output_interval: 输出采样间隔，默认 T/100
        dt_min: 最小步长，默认 T/100000
        dt_max: 最大步长，默认 T/200
        eta: 自适应步长系数，默认 0.02
        progress_callback: callable(pct, t_current, dt) 或 None
        step_callback: callable(t_hist, r_hist, v_hist) 或 None，每次采样时调用
        stop_flag: 可变的 bool 容器（如 list [False]），设为 [True] 可中断模拟

    Returns:
        dict: {
            't':     np.ndarray (N,)      时间序列
            'r':     np.ndarray (N,3,3)   位置历史
            'v':     np.ndarray (N,3,3)   速度历史
            'steps': int                  总积分步数
        }
    """
    T = params['T']
    if output_interval is None:
        output_interval = T / 100.0
    if dt_min is None:
        dt_min = T / 100000.0
    if dt_max is None:
        dt_max = T / 200.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64

    G = torch.tensor(1.0, dtype=dtype, device=device)
    m = torch.tensor(params['m'], dtype=dtype, device=device)
    r = torch.tensor(params['r'], dtype=dtype, device=device)
    v = torch.tensor(params['v'], dtype=dtype, device=device)

    t_hist = [0.0]
    r_hist = [r.cpu().numpy().copy()]
    v_hist = [v.cpu().numpy().copy()]

    t_current = 0.0
    next_out = output_interval
    total_steps = 0
    last_pct = -1

    while t_current < T:
        if stop_flag and stop_flag[0]:
            break

        dt = compute_adaptive_dt(r, v, dt_min, dt_max, eta)
        dt = min(dt, T - t_current)
        r, v = yoshida4_step(r, v, G, m, dt)
        t_current += dt
        total_steps += 1

        pct = int(t_current / T * 100)
        if pct != last_pct and progress_callback:
            progress_callback(pct, t_current, dt)
            last_pct = pct

        if t_current >= next_out - 1e-10:
            t_hist.append(t_current)
            r_hist.append(r.cpu().numpy().copy())
            v_hist.append(v.cpu().numpy().copy())
            next_out += output_interval
            if step_callback:
                step_callback(t_hist, r_hist, v_hist)

    return {
        't':     np.array(t_hist),
        'r':     np.array(r_hist),
        'v':     np.array(v_hist),
        'steps': total_steps,
    }


# ==================== 入口 ====================

def _compute_auto_solve_c(mA, mB, mC, rA, rB, vA, vB):
    """自动求解天体 C：质心归零 + 总动量归零。"""
    rC = -(mA * np.asarray(rA) + mB * np.asarray(rB)) / mC
    vC = -(mA * np.asarray(vA) + mB * np.asarray(vB)) / mC
    return tuple(rC.tolist()), tuple(vC.tolist())


def _parse_3vec(s):
    """解析 'x,y,z' 字符串为三元组。"""
    parts = [float(x.strip()) for x in s.split(',')]
    if len(parts) != 3:
        raise ValueError(f'需要 3 个值，用了逗号分隔，如: 1,-2,0；实际收到: {s}')
    return tuple(parts)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    print('=' * 50)
    print('  三体问题 GT 轨迹模拟')
    print('=' * 50)
    print()
    print('请选择初值模式:')
    print('  1. 等边三角形特解（绕质心圆周运动）')
    print('  2. 8字形特解（Chenciner-Montgomery 周期轨道）')
    print('  3. 自动求解天体 C（自由输入 A/B，质心+动量归零解 C）')
    print()
    choice = input('输入编号 (1/2/3) [1]: ').strip() or '1'

    T_sim = float(input('总时长 T [100000]: ').strip() or '100000')

    if choice == '1':
        R = float(input('R 边长 [10000]: ').strip() or '10000')
        mA = float(input('mA [1.0]: ').strip() or '1.0')
        mB = mA
        mC = mA
        pos, vel = equilateral_ic(R)
        label_extra = f'R={R:.0f}  m₁=m₂=m₃={mA:.3g}'

    elif choice == '2':
        scale = float(input('Scale 尺度 [1.0]: ').strip() or '1.0')
        m_val = float(input('单天体质量 [1.0]: ').strip() or '1.0')
        mA = mB = mC = m_val
        pos, vel = figure8_ic(scale=scale, m=m_val)
        label_extra = f'scale={scale:.4g}  m₁=m₂=m₃={m_val:.3g}'

    elif choice == '3':
        mA = float(input('mA [1.0]: ').strip() or '1.0')
        mB = float(input('mB [2.0]: ').strip() or '2.0')
        mC = float(input('mC [3.0]: ').strip() or '3.0')
        print()
        print('天体 A 初值:')
        rA = _parse_3vec(input('  rA (x,y,z) [0,0,0]: ').strip() or '0,0,0')
        vA = _parse_3vec(input('  vA (vx,vy,vz) [0.5,-0.866,0]: ').strip() or '0.5,-0.866,0')
        print('天体 B 初值:')
        rB = _parse_3vec(input('  rB (x,y,z) [10000,0,0]: ').strip() or '10000,0,0')
        vB = _parse_3vec(input('  vB (vx,vy,vz) [0.5,0.866,0]: ').strip() or '0.5,0.866,0')

        rC, vC = _compute_auto_solve_c(mA, mB, mC, rA, rB, vA, vB)
        pos = (rA, rB, rC)
        vel = (vA, vB, vC)
        label_extra = f'mA={mA:.3g} mB={mB:.3g} mC={mC:.3g}  (C 自动求解)'

    else:
        print(f'无效选择: {choice}')
        sys.exit(1)

    params = {
        'T': T_sim,
        'm': [mA, mB, mC],
        'r': [list(p) for p in pos],
        'v': [list(v) for v in vel],
    }

    print(f'\n运行三体模拟 (T={T_sim:.0f})…')
    result = run_simulation(params, output_interval=T_sim / 200.0)
    print(f'完成。积分步数={result["steps"]}, 输出点数={len(result["t"])}')

    # ── 绘制 ABC 三星合并总星图 ──
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#fafafa')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(f'三体真实轨迹  ABC 同框\nT={T_sim:.0f}  {label_extra}',
                 fontsize=14, fontweight='bold')

    colors = {'A': '#c0392b', 'B': '#27ae60', 'C': '#2980b9'}
    all_xyz = []

    for idx, name in enumerate(['A', 'B', 'C']):
        r = result['r'][:, idx, :]
        all_xyz.append(r)
        ax.plot(r[:, 0], r[:, 1], r[:, 2],
                '-', color=colors[name], lw=1.2, alpha=0.85, label=f'天体 {name}')

    xyz = np.concatenate(all_xyz)
    margin = max(np.ptp(xyz[:, 0]), np.ptp(xyz[:, 1]), np.ptp(xyz[:, 2])) * 0.08 + 1.0
    ax.set_xlim(xyz[:, 0].min() - margin, xyz[:, 0].max() + margin)
    ax.set_ylim(xyz[:, 1].min() - margin, xyz[:, 1].max() + margin)
    ax.set_zlim(xyz[:, 2].min() - margin, xyz[:, 2].max() + margin)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.show()
