"""
独立推理评估脚本 —— 加载已进化的预测模型，纯推理检验预测能力。

不对模型做任何更新（不训练、不反馈），仅评估：
  1. 沿整条轨迹分段的预测误差
  2. 模型能够准确预测的时间步长上限（可靠预测视界）
  3. 可视化：误差随时长衰减图 / 预测视界沿轨迹变化 / 预测 vs 真实轨迹对比

用法:
    python evaluate_predictor.py [checkpoint_path]
    默认加载 checkpoints/predictor_online.pt
"""

import os, sys
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch

from three_body import run_simulation
from closed_loop_evolution import (
    NeuralPredictor, ClosedLoopEvolution,
)

_interp_trajectory = ClosedLoopEvolution._interp_trajectory

# ── matplotlib 中文字体 ──
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

BODY_COLORS = ['#c0392b', '#27ae60', '#2980b9']
BODY_NAMES = ['A', 'B', 'C']
ERROR_THRESHOLD = 0.5  # r-MSE 阈值，超过此值认为预测失效


def load_checkpoint(path):
    """加载 checkpoint，返回 (predictor, extras_dict)。"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(path, map_location=device, weights_only=False)

    hidden_dims = ckpt.get('hidden_dims', (128, 128, 128))
    lr = ckpt.get('learning_rate', 0.001)

    predictor = NeuralPredictor(
        hidden_dims=hidden_dims,
        learning_rate=lr,
        device=device,
    )
    predictor.dynamics.load_state_dict(ckpt['dynamics'])
    predictor.optimizer.load_state_dict(ckpt['optimizer'])
    predictor._metadata_history = ckpt.get('metadata', [])
    extras = ckpt.get('extras', {})
    return predictor, extras


def run_pure_inference(predictor, gt_params, predict_horizon,
                       n_predict_steps, num_samples=40, n_cycles=1,
                       r_final=None, v_final=None):
    """纯推理：[nT, (n+1)T] 均匀采样，从每个采样点做预测外推。

    这是对"时间外泛化"的评估：模型在 [0, nT] 内进化，
    在 [nT, (n+1)T] 上取样，检验它能否外推到从未见过的轨道构型。

    若提供了 r_final/v_final（checkpoint 中保存的 nT 时刻状态），
    则直接以此作为 GT 初始条件，只模拟 [nT, (n+1)T] 一个周期。

    Args:
        n_cycles: 模型已完成训练的宇宙周期数（默认 1，即训练了 [0, T]）
        r_final:  可选的 (3,3) 数组，nT 时刻的真实位置
        v_final:  可选的 (3,3) 数组，nT 时刻的真实速度

    Returns:
        segments: list of {t_start, t_pred, t_cmp, r_pred, err_body}
        full_gt:  GT 轨迹（若从 nT 开始则只含 [nT, (n+1)T]，否则 [0, (n+1)T]）
    """
    T = gt_params['T']
    nT = n_cycles * T

    if r_final is not None and v_final is not None:
        # ── 直接从 nT 状态开始，只模拟一个周期 ──
        r_init = np.asarray(r_final)
        v_init = np.asarray(v_final)
        seg_params = {
            'T': T,
            'm': gt_params['m'],
            'r': r_init.tolist() if hasattr(r_init, 'tolist') else list(r_init),
            'v': v_init.tolist() if hasattr(v_init, 'tolist') else list(v_init),
        }
        full_gt = run_simulation(seg_params, output_interval=T / 200.0)
        # 将相对时间偏移到绝对时间
        full_gt['t'] = full_gt['t'] + nT
    else:
        # ── 回退：完整模拟 [0, (n+1)T] ──
        total_T = (n_cycles + 1) * T
        params_total = {**gt_params, 'T': total_T}
        full_gt = run_simulation(params_total, output_interval=total_T / 200.0)

    sample_times = np.linspace(nT, (n_cycles + 1) * T - predict_horizon * 0.5,
                               num_samples)

    segments = []
    for si, t_start in enumerate(sample_times):
        gt_idx = np.searchsorted(full_gt['t'], t_start)
        gt_idx = min(gt_idx, len(full_gt['t']) - 1)
        r0 = full_gt['r'][gt_idx].copy()
        v0 = full_gt['v'][gt_idx].copy()
        t_start = float(full_gt['t'][gt_idx])

        state = np.hstack([r0.flatten(), v0.flatten()]).reshape(3, 6)
        prediction = predictor.predict(state, predict_horizon, n_steps=n_predict_steps)

        pred_t = prediction['t']
        pred_t = pred_t[pred_t <= predict_horizon + 1e-10]
        cmp_t = np.linspace(pred_t[0], pred_t[-1],
                            min(n_predict_steps, len(pred_t)))

        r_pred_interp = _interp_trajectory(prediction['t'], prediction['r'], cmp_t)

        # GT 在 [t_start, t_start + predict_horizon] 的数据
        gt_mask = (full_gt['t'] >= t_start - 1e-10) & \
                  (full_gt['t'] <= t_start + predict_horizon + 1e-10)
        gt_t = full_gt['t'][gt_mask]
        gt_r = full_gt['r'][gt_mask]

        if len(gt_t) >= 3 and len(cmp_t) >= 2:
            r_gt_interp = _interp_trajectory(gt_t - t_start, gt_r, cmp_t)
            err_body = np.array([
                np.linalg.norm(r_pred_interp[:, b] - r_gt_interp[:, b], axis=1)
                for b in range(3)
            ])
        else:
            err_body = np.full((3, len(cmp_t)), np.nan)

        segments.append({
            't_start': t_start,
            't_pred': pred_t, 't_cmp': cmp_t,
            'r_pred': r_pred_interp,
            'err_body': err_body,
        })

        if (si + 1) % 10 == 1 or si <= 3:
            print(f'  样本 {si+1:4d}/{num_samples}  t₀={t_start:.0f}  预测视界={predict_horizon:.0f}')

    return segments, full_gt


def _params_from_state(gt_params, r0, v0, T_seg):
    """构造以当前状态为初始条件的模拟参数。"""
    return {
        'T': T_seg,
        'm': gt_params['m'],
        'r': r0.copy(),
        'v': v0.copy(),
    }


def compute_reliable_horizon(segments, threshold=ERROR_THRESHOLD):
    """计算每段的"可靠预测视界"——预测位置误差首次超过阈值的时间。

    Returns:
        horizons: list of (t_start, max_reliable_t, last_err)
    """
    horizons = []
    for seg in segments:
        err = seg['err_body']
        if err.shape[1] < 2 or np.all(np.isnan(err)):
            horizons.append((seg['t_start'], 0.0, np.nan))
            continue

        # 取三个天体的平均位置误差
        err_mean = np.nanmean(err, axis=0)  # (n_times,)
        t_pred = seg['t_cmp'] if len(seg['t_cmp']) == len(err_mean) else seg['t_pred']

        # 找到误差首次超过阈值的时间点
        exceed_idx = np.where(err_mean > threshold)[0]
        if len(exceed_idx) == 0:
            max_t = t_pred[-1]
            last_err = err_mean[-1]
        else:
            idx = exceed_idx[0]
            if idx == 0:
                max_t = 0.0
            else:
                # 线性插值估计准确超过阈值的时间
                w = (threshold - err_mean[idx - 1]) / max(1e-10, err_mean[idx] - err_mean[idx - 1])
                max_t = t_pred[idx - 1] + w * (t_pred[idx] - t_pred[idx - 1])
            last_err = err_mean[idx]

        horizons.append((seg['t_start'], max_t, last_err))

    return horizons


def plot_evaluation(segments, horizons, full_gt, extras, save_path='evaluation_result.png'):
    """绘制评估结果。"""
    T = extras['gt_params']['T']
    n_cycles = extras.get('n_cycles', 1)
    nT = n_cycles * T
    pred_horizon = extras.get('predict_horizon', 500)
    threshold = ERROR_THRESHOLD

    fig = plt.figure(figsize=(18, 9))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)
    fig.suptitle(f'预测器时间外泛化评估  [{nT:.0f}, {(n_cycles+1)*T:.0f}]（{n_cycles}周期训练后）  阈值={threshold}',
                 fontsize=16, fontweight='bold')

    # ═══════════════════════════════════════════════
    #  子图 1：误差随预测时长衰减（取中间段做典型分析）
    # ═══════════════════════════════════════════════
    ax1 = fig.add_subplot(gs[0, 0])
    mid_idx = len(segments) // 2
    seg = segments[mid_idx]
    err = seg['err_body']
    t_pred = seg['t_cmp'] if len(seg['t_cmp']) == err.shape[1] else seg['t_pred']
    for b in range(3):
        ax1.plot(t_pred, err[b], color=BODY_COLORS[b], alpha=0.7,
                 label=f'天体 {BODY_NAMES[b]}', linewidth=2)
    err_mean = np.nanmean(err, axis=0)
    ax1.plot(t_pred, err_mean, 'k-', linewidth=2.5, label='平均')
    ax1.axhline(threshold, color='#e74c3c', linestyle='--', linewidth=1.5,
                label=f'阈值={threshold}')
    ax1.set_xlabel('预测时长 Δt', fontsize=11)
    ax1.set_ylabel('位置误差 |Δr|', fontsize=11)
    ax1.set_title(f'典型预测误差（段 #{mid_idx}, t≈{seg["t_start"]:.0f}）', fontsize=12)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # ═══════════════════════════════════════════════
    #  子图 2：可靠预测视界沿轨迹的变化
    # ═══════════════════════════════════════════════
    ax2 = fig.add_subplot(gs[0, 1])
    t_starts = [h[0] for h in horizons]
    max_ts = [h[1] for h in horizons]
    ax2.fill_between(t_starts, 0, max_ts, alpha=0.25, color='#3498db')
    ax2.plot(t_starts, max_ts, 'o-', color='#2980b9', markersize=4, linewidth=1.5,
             label='可靠预测视界')
    ax2.axhline(pred_horizon, color='#95a5a6', linestyle=':', linewidth=1,
                label=f'最大预测视界={pred_horizon:.0f}')
    ax2.set_xlabel('宇宙时间 t', fontsize=11)
    ax2.set_ylabel('可靠预测时长', fontsize=11)
    ax2.set_title('可靠预测视界沿轨迹演化', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ═══════════════════════════════════════════════
    #  子图 3：可靠视界直方图
    # ═══════════════════════════════════════════════
    ax3 = fig.add_subplot(gs[0, 2])
    valid_ts = [h for h in max_ts if h > 0 and not np.isnan(h)]
    counts, bins = np.histogram(valid_ts, bins=20)
    ax3.bar(bins[:-1], counts, width=np.diff(bins), edgecolor='white',
            color='#8e44ad', alpha=0.7)
    ax3.axvline(np.median(valid_ts) if valid_ts else 0, color='#e74c3c',
                linestyle='--', linewidth=2,
                label=f'中位数={np.median(valid_ts):.0f}' if valid_ts else '')
    ax3.set_xlabel('可靠预测时长', fontsize=11)
    ax3.set_ylabel('段数', fontsize=11)
    ax3.set_title('可靠预测视界分布', fontsize=12)
    if valid_ts:
        ax3.legend(fontsize=9)
    ax3.grid(axis='y', alpha=0.3)

    # ═══════════════════════════════════════════════
    #  子图 4-6：ABC 三维轨迹 GT [nT,(n+1)T] + 末段预测
    # ═══════════════════════════════════════════════
    last_seg = segments[-1]
    # 只显示 [nT, (n+1)T] 评估区的 GT
    mask_eval = full_gt['t'] >= nT - 1e-10
    for b in range(3):
        ax = fig.add_subplot(gs[1, b], projection='3d')
        r_eval = full_gt['r'][mask_eval]
        ax.plot(r_eval[:, b, 0], r_eval[:, b, 1], r_eval[:, b, 2],
                '-', color='#7f8c8d', alpha=0.5, linewidth=1,
                label=f'GT [{nT:.0f},{(n_cycles+1)*T:.0f}]')
        # 末段预测
        if last_seg.get('r_pred') is not None and last_seg.get('t_start') is not None:
            r_p = last_seg['r_pred']
            ax.plot(r_p[:, b, 0], r_p[:, b, 1], r_p[:, b, 2],
                    'o-', color=BODY_COLORS[b], markersize=3, linewidth=2,
                    label='预测')

        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title(f'天体 {BODY_NAMES[b]}  末段预测 vs GT', fontsize=12,
                     fontweight='bold', color=BODY_COLORS[b])
        ax.legend(fontsize=8)
        ax.set_facecolor('#fafafa')

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'\n图表已保存到: {save_path}')
    plt.show()


def print_summary(segments, horizons, extras):
    """打印评估摘要。"""
    T = extras['gt_params']['T']
    n_cycles = extras.get('n_cycles', 1)
    nT = n_cycles * T
    valid_ts = [h[1] for h in horizons if h[1] > 0 and not np.isnan(h[1])]
    all_errs = []
    for seg in segments:
        e = seg['err_body']
        if not np.all(np.isnan(e)):
            all_errs.append(np.nanmean(e))

    print('\n' + '=' * 60)
    print('  评  估  摘  要')
    print('=' * 60)
    print(f'  评估范围: [{nT:.0f}, {(n_cycles+1)*T:.0f}]（{n_cycles} 周期训练后）')
    print(f'  训练总时长            = {nT:.0f}（{n_cycles} × T，T={T:.0f}）')
    print(f'  采样点数              = {len(segments)}')
    print(f'  可靠视界中位数      = {np.median(valid_ts):.1f}' if valid_ts else '  无有效数据')
    print(f'  可靠视界平均值      = {np.mean(valid_ts):.1f}' if valid_ts else '')
    print(f'  可靠视界最小值      = {np.min(valid_ts):.1f}' if valid_ts else '')
    print(f'  可靠视界最大值      = {np.max(valid_ts):.1f}' if valid_ts else '')
    if all_errs:
        print(f'  平均位置误差        = {np.mean(all_errs):.4e} (中位数={np.median(all_errs):.4e})')
        print(f'  误差范围            = {np.min(all_errs):.4e} ~ {np.max(all_errs):.4e}')
    if valid_ts:
        pct_pred = np.mean(valid_ts) / extras.get('predict_horizon', 500) * 100
        print(f'  可靠视界/预测视界    = {pct_pred:.1f}%')
    print('=' * 60)


def main():
    # 默认加载上次闭环进化保存的模型
    path = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints/predictor_online.pt'
    if not os.path.exists(path):
        print(f'错误: 找不到 checkpoint 文件: {path}')
        print('请先运行闭环进化生成模型，或指定文件路径作为参数')
        sys.exit(1)

    print(f'加载模型: {path}')
    predictor, extras = load_checkpoint(path)

    gt_params = extras.get('gt_params')
    if gt_params is None:
        print('错误: checkpoint 中缺少 gt_params，无法运行评估')
        print('请重新运行闭环进化保存含参数的模型')
        sys.exit(1)

    predict_horizon = extras.get('predict_horizon', 500)
    n_predict_steps = extras.get('n_predict_steps', 50)
    n_cycles = extras.get('n_cycles', 1)
    r_final = extras.get('r_final', None)
    v_final = extras.get('v_final', None)
    T = gt_params['T']
    nT = n_cycles * T

    print(f'  训练总时长={nT:.0f}（{n_cycles} 个宇宙周期，T={T:.0f}）  预测视界={predict_horizon}')
    if r_final is not None:
        print(f'  GT 模拟: 从 nT 状态直接运行 [{nT:.0f}, {(n_cycles+1)*T:.0f}]')
    else:
        print(f'  GT 模拟: 完整 [0, {(n_cycles+1)*T:.0f}]（r_final 不可用，回退）')
    print(f'  评估范围: [{nT:.0f}, {(n_cycles+1)*T:.0f}]')
    print(f'  预测步数={n_predict_steps}  设备={predictor.device}')
    n_fb = len(predictor._metadata_history)
    print(f'  已记录反馈次数={n_fb}')

    if n_fb == 0:
        print('\n⚠ 该模型尚未经过任何闭环反馈训练，预测能力可能有限')

    print(f'\n开始时间外泛化评估（[{nT:.0f}, {(n_cycles+1)*T:.0f}]上均匀采样）…')
    segments, full_gt = run_pure_inference(
        predictor, gt_params, predict_horizon,
        n_predict_steps, num_samples=40, n_cycles=n_cycles,
        r_final=r_final, v_final=v_final,
    )

    horizons = compute_reliable_horizon(segments, threshold=ERROR_THRESHOLD)

    print_summary(segments, horizons, extras)

    plot_evaluation(segments, horizons, full_gt, extras,
                    save_path='evaluation_result.png')


if __name__ == '__main__':
    main()
