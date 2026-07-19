"""
预测器纯推理评估 — UI 版
=========================
加载已进化的 checkpoint，沿整条 GT 轨迹分段做纯推理，
实时在主星图上展示预测拼接轨迹的累积过程。

用法:
    python evaluate_ui.py [checkpoint_path]
"""

import sys, os
import numpy as np
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QProgressBar, QLabel, QFileDialog, QGroupBox, QGridLayout,
    QDoubleSpinBox, QSpinBox, QTextEdit,
)
from PySide6.QtCore import QThread, Signal
import torch

from three_body import run_simulation
from closed_loop_evolution import NeuralPredictor, ClosedLoopEvolution

_interp_trajectory = ClosedLoopEvolution._interp_trajectory

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

BODY_COLORS = ['#c0392b', '#27ae60', '#2980b9']
BODY_NAMES = ['A', 'B', 'C']
ERROR_THRESHOLD = 0.5 # r-MSE 阈值，超过此值认为预测失效



# ═══════════════════════════════════════════════════════════════
#  辅助函数
# ═══════════════════════════════════════════════════════════════

def load_checkpoint(path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(path, map_location=device, weights_only=False)
    hidden_dims = ckpt.get('hidden_dims', (128, 128, 128))
    lr = ckpt.get('learning_rate', 0.001)
    predictor = NeuralPredictor(hidden_dims=hidden_dims, learning_rate=lr, device=device)
    predictor.dynamics.load_state_dict(ckpt['dynamics'])
    predictor.optimizer.load_state_dict(ckpt['optimizer'])
    predictor._metadata_history = ckpt.get('metadata', [])
    extras = ckpt.get('extras', {})
    return predictor, extras


def _params_from_state(gt_params, r0, v0, T_seg):
    return {'T': T_seg, 'm': gt_params['m'], 'r': r0.copy(), 'v': v0.copy()}


def compute_reliable_horizon(segments, threshold=ERROR_THRESHOLD):
    horizons = []
    for seg in segments:
        err = seg['err_body']
        if err.shape[1] < 2 or np.all(np.isnan(err)):
            horizons.append((seg['t_start'], 0.0, np.nan))
            continue
        err_mean = np.nanmean(err, axis=0)
        t_pred = seg['t_cmp'] if len(seg['t_cmp']) == err.shape[1] else seg['t_pred']
        exceed_idx = np.where(err_mean > threshold)[0]
        if len(exceed_idx) == 0:
            max_t = t_pred[-1]; last_err = err_mean[-1]
        else:
            idx = exceed_idx[0]
            if idx == 0:
                max_t = 0.0; last_err = err_mean[0]
            else:
                w = (threshold - err_mean[idx-1]) / max(1e-10, err_mean[idx] - err_mean[idx-1])
                max_t = t_pred[idx-1] + w * (t_pred[idx] - t_pred[idx-1])
                last_err = err_mean[idx]
        horizons.append((seg['t_start'], max_t, last_err))
    return horizons


# ═══════════════════════════════════════════════════════════════
#  评估工作线程
# ═══════════════════════════════════════════════════════════════

class EvaluateWorker(QThread):
    progress = Signal(int, str)           # pct, status text
    gt_ready = Signal(object)             # full GT trajectory dict
    segment_done = Signal(object)         # segment result dict
    finished = Signal(object, object, object)  # segments, horizons, extras
    error = Signal(str)

    def __init__(self, ckpt_path, predict_horizon, n_predict_steps, num_samples):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.predict_horizon = predict_horizon
        self.n_predict_steps = n_predict_steps
        self.num_samples = num_samples

    def run(self):
        try:
            self.progress.emit(0, '加载模型…')
            predictor, extras = load_checkpoint(self.ckpt_path)

            gt_params = extras.get('gt_params')
            if gt_params is None:
                self.error.emit('checkpoint 中缺少 gt_params')
                return

            T = gt_params['T']
            n_cycles = extras.get('n_cycles', 1)  # 默认 1（向后兼容旧 checkpoint）
            r_final = extras.get('r_final', None)
            v_final = extras.get('v_final', None)
            nT = n_cycles * T

            if r_final is not None and v_final is not None:
                # ── 直接从 nT 状态开始，只模拟一个周期 ──
                self.progress.emit(0, f'运行真实模拟 [{nT:.0f} → {(n_cycles+1)*T:.0f}]…')
                r_init = np.asarray(r_final)
                v_init = np.asarray(v_final)
                seg_params = {
                    'T': T,
                    'm': gt_params['m'],
                    'r': r_init.tolist() if hasattr(r_init, 'tolist') else list(r_init),
                    'v': v_init.tolist() if hasattr(v_init, 'tolist') else list(v_init),
                }
                # 自适应 GT 采样密度：保证每个预测窗口内至少有 MIN_GT_POINTS 个采样点
                gt_output_interval = self.predict_horizon / 20.0
                full_gt = run_simulation(seg_params, output_interval=gt_output_interval)
                full_gt['t'] = full_gt['t'] + nT  # 偏移到绝对时间
            else:
                # ── 回退：完整模拟 [0, (n+1)T] ──
                self.progress.emit(0, f'运行真实模拟 (0 → {(n_cycles+1)}T)…')
                total_T = (n_cycles + 1) * T
                params_total = {**gt_params, 'T': total_T}
                gt_output_interval = self.predict_horizon / 20.0
                full_gt = run_simulation(params_total, output_interval=gt_output_interval)

            # 发射 GT 供主星图显示
            gt_data = {
                't': full_gt['t'], 'r': full_gt['r'], 'v': full_gt['v'],
                'T_mark': nT,       # 训练区边界（若只模拟 [nT,(n+1)T] 则无意义但保留兼容）
                'n_cycles': n_cycles,
                'from_final_state': r_final is not None,
            }
            self.gt_ready.emit(gt_data)

            # ── 在 [nT, (n+1)T] 中均匀采样评估点 ──
            sample_times = np.linspace(nT, (n_cycles + 1) * T - self.predict_horizon * 0.5,
                                       self.num_samples)

            segments = []
            stitch_t_list, stitch_r_list = [], []

            for si, t_start in enumerate(sample_times):
                # 从 GT 中取该时刻的 (r, v)
                gt_idx = np.searchsorted(full_gt['t'], t_start)
                gt_idx = min(gt_idx, len(full_gt['t']) - 1)
                r0 = full_gt['r'][gt_idx].copy()
                v0 = full_gt['v'][gt_idx].copy()
                t_start = float(full_gt['t'][gt_idx])

                state = np.hstack([r0.flatten(), v0.flatten()]).reshape(3, 6)
                prediction = predictor.predict(state, self.predict_horizon,
                                                n_steps=self.n_predict_steps)

                pred_t = prediction['t']
                pred_t = pred_t[pred_t <= self.predict_horizon + 1e-10]
                cmp_t = np.linspace(pred_t[0], pred_t[-1],
                                    min(self.n_predict_steps, len(pred_t)))

                r_pred_interp = _interp_trajectory(prediction['t'], prediction['r'], cmp_t)

                # GT 在对应时段的数据
                gt_mask = (full_gt['t'] >= t_start - 1e-10) & \
                          (full_gt['t'] <= t_start + self.predict_horizon + 1e-10)
                gt_t = full_gt['t'][gt_mask]
                gt_r = full_gt['r'][gt_mask]

                if len(gt_t) >= 2 and len(cmp_t) >= 2:
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

                # 累积拼接预测（绝对时间，画在右图上）
                mask = prediction['t'] <= self.predict_horizon + 1e-10
                if mask.sum() > 1:
                    stitch_t_list.append(t_start + prediction['t'][mask])
                    stitch_r_list.append(prediction['r'][mask])

                # 实时更新
                acc_pred = {}
                if stitch_t_list:
                    acc_pred['t'] = np.concatenate(stitch_t_list)
                    acc_pred['r'] = np.concatenate(stitch_r_list)
                else:
                    acc_pred['t'] = np.array([])
                    acc_pred['r'] = np.zeros((0, 3, 3))

                self.segment_done.emit({
                    't_start': t_start,
                    'seg_num': si + 1,
                    'accumulated_pred': acc_pred,
                    'segments_so_far': segments,
                })

                pct = int((si + 1) / self.num_samples * 100)
                self.progress.emit(pct, f'样本 {si + 1}/{self.num_samples}  t₀={t_start:.0f}')

            horizons = compute_reliable_horizon(segments)
            self.finished.emit(segments, horizons, extras)

        except Exception as e:
            import traceback
            self.error.emit(f'{e}\n{traceback.format_exc()}')


# ═══════════════════════════════════════════════════════════════
#  主星图画布（评估专用，实时更新）
# ═══════════════════════════════════════════════════════════════

class EvalOverviewCanvas(QWidget):
    """评估主星图：左 = GT 轨迹，右 = 累积预测轨迹，实时更新。"""

    BODY_COLORS = {'A': '#c0392b', 'B': '#27ae60', 'C': '#2980b9'}

    def __init__(self):
        super().__init__()
        layout = QHBoxLayout(self)
        self.fig = Figure(figsize=(14, 6.5), dpi=120)
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self.ax_gt   = self.fig.add_subplot(1, 2, 1, projection='3d')
        self.ax_pred = self.fig.add_subplot(1, 2, 2, projection='3d')

        for ax in [self.ax_gt, self.ax_pred]:
            ax.set_facecolor('#fafafa')
            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
            ax.text2D(0.5, 0.5, '加载模型后开始评估…', ha='center', va='center',
                      transform=ax.transAxes, fontsize=14, color='#ccc')

        self.ax_gt.set_title('真实轨迹（ABC 同框）', fontsize=13, fontweight='bold')
        self.ax_pred.set_title('预测外推（ABC 同框）', fontsize=13, fontweight='bold')
        self.fig.tight_layout()
        self.canvas.draw_idle()

        self._gt_data = None
        self._n_cycles = 1
        self._T = 0

    def set_gt(self, gt_data):
        """设置 GT 数据并绘制左图。
        若 GT 只涵盖 [nT, (n+1)T]（from_final_state），则绘制为单一实线。
        否则实线 = [0, nT] 训练区，虚线 = [nT, (n+1)T] 评估区。
        """
        self._gt_data = gt_data
        T_mark = gt_data.get('T_mark', gt_data['t'][-1] / 2)
        n_cycles = gt_data.get('n_cycles', 1)
        T = T_mark / n_cycles if n_cycles > 0 else T_mark
        self._n_cycles = n_cycles
        self._T = T
        from_final = gt_data.get('from_final_state', False)

        self.ax_gt.cla()
        self.ax_gt.set_facecolor('#fafafa')
        self.ax_gt.set_xlabel('X'); self.ax_gt.set_ylabel('Y'); self.ax_gt.set_zlabel('Z')

        t = gt_data['t']
        r = gt_data['r']
        all_xyz = []

        if from_final:
            # 只有 [nT, (n+1)T] — 单一实线
            eval_end = T_mark + T
            self.ax_gt.set_title(f'真实轨迹 [{T_mark:.0f}, {eval_end:.0f}]（评估区）',
                                 fontsize=12, fontweight='bold')
            for idx, name in enumerate(['A', 'B', 'C']):
                rt = r[:, idx, :]
                all_xyz.append(rt)
                self.ax_gt.plot(rt[:, 0], rt[:, 1], rt[:, 2],
                                '-', color=self.BODY_COLORS[name], lw=1.5,
                                alpha=0.9, label=f'天体 {name}')
        else:
            # 完整 [0, (n+1)T] — 分训练区和评估区
            eval_end = T_mark + T
            self.ax_gt.set_title(f'真实轨迹（{n_cycles}周期训练）  [0,{T_mark:.0f}]=训练  [{T_mark:.0f},{eval_end:.0f}]=评估',
                                 fontsize=12, fontweight='bold')
            mask_train = t <= T_mark + 1e-10
            mask_eval  = t >= T_mark - 1e-10

            for idx, name in enumerate(['A', 'B', 'C']):
                rt = r[:, idx, :]
                if mask_train.any():
                    rt_train = rt[mask_train]
                    all_xyz.append(rt_train)
                    self.ax_gt.plot(rt_train[:, 0], rt_train[:, 1], rt_train[:, 2],
                                    '-', color=self.BODY_COLORS[name], lw=1.5,
                                    alpha=0.9, label=f'天体 {name}')
                if mask_eval.any():
                    rt_eval = rt[mask_eval]
                    all_xyz.append(rt_eval)
                    self.ax_gt.plot(rt_eval[:, 0], rt_eval[:, 1], rt_eval[:, 2],
                                    '--', color=self.BODY_COLORS[name], lw=1.2,
                                    alpha=0.6)
        if all_xyz:
            xyz = np.concatenate(all_xyz)
            margin = max(np.ptp(xyz[:, 0]), np.ptp(xyz[:, 1]), np.ptp(xyz[:, 2])) * 0.08 + 1.0
            self.ax_gt.set_xlim(xyz[:, 0].min() - margin, xyz[:, 0].max() + margin)
            self.ax_gt.set_ylim(xyz[:, 1].min() - margin, xyz[:, 1].max() + margin)
            self.ax_gt.set_zlim(xyz[:, 2].min() - margin, xyz[:, 2].max() + margin)

        self.ax_gt.legend(fontsize=8)
        self.canvas.draw_idle()

    def update_pred(self, acc_pred):
        """更新右图：拼接预测轨迹。"""
        self.ax_pred.cla()
        self.ax_pred.set_facecolor('#fafafa')
        self.ax_pred.set_xlabel('X'); self.ax_pred.set_ylabel('Y'); self.ax_pred.set_zlabel('Z')
        nT = self._n_cycles * self._T
        eval_end = (self._n_cycles + 1) * self._T
        self.ax_pred.set_title(f'预测外推 [{nT:.0f}, {eval_end:.0f}]（ABC 同框）', fontsize=13, fontweight='bold')

        t_arr = acc_pred.get('t', np.array([]))
        r_arr = acc_pred.get('r', np.zeros((0, 3, 3)))

        if len(t_arr) > 0 and r_arr.shape[0] > 0:
            all_xyz = []
            for idx, name in enumerate(['A', 'B', 'C']):
                r = r_arr[:, idx, :]
                all_xyz.append(r)
                self.ax_pred.plot(r[:, 0], r[:, 1], r[:, 2],
                                  '--', color=self.BODY_COLORS[name], lw=1.2,
                                  alpha=0.85, label=f'天体 {name}')
            if all_xyz:
                xyz = np.concatenate(all_xyz)
                margin = max(np.ptp(xyz[:, 0]), np.ptp(xyz[:, 1]), np.ptp(xyz[:, 2])) * 0.08 + 1.0
                self.ax_pred.set_xlim(xyz[:, 0].min() - margin, xyz[:, 0].max() + margin)
                self.ax_pred.set_ylim(xyz[:, 1].min() - margin, xyz[:, 1].max() + margin)
                self.ax_pred.set_zlim(xyz[:, 2].min() - margin, xyz[:, 2].max() + margin)
            self.ax_pred.legend(fontsize=8)
        else:
            self.ax_pred.text2D(0.5, 0.5, '运行中…', ha='center', va='center',
                                transform=self.ax_pred.transAxes, fontsize=14, color='#ccc')

        self.canvas.draw_idle()

    def clear(self):
        self._gt_data = None
        for ax in [self.ax_gt, self.ax_pred]:
            ax.cla()
            ax.set_facecolor('#fafafa')
            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
            ax.text2D(0.5, 0.5, '加载模型后开始评估…', ha='center', va='center',
                      transform=ax.transAxes, fontsize=14, color='#ccc')
        self.ax_gt.set_title('真实轨迹（ABC 同框）', fontsize=13, fontweight='bold')
        self.ax_pred.set_title('预测外推（ABC 同框）', fontsize=13, fontweight='bold')
        self.fig.tight_layout()
        self.canvas.draw_idle()


# ═══════════════════════════════════════════════════════════════
#  评估主窗口
# ═══════════════════════════════════════════════════════════════

class EvaluateWindow(QMainWindow):
    def __init__(self, ckpt_path=None):
        super().__init__()
        self.setWindowTitle('预测器纯推理评估')
        self.resize(1300, 850)
        self._ckpt_path = ckpt_path or 'checkpoints/predictor_online.pt'
        self._worker = None

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(8)

        # ── 控制栏 ──
        ctrl = QHBoxLayout()

        self.btn_load = QPushButton('📂 加载模型')
        self.btn_load.clicked.connect(self._choose_checkpoint)
        ctrl.addWidget(self.btn_load)

        self.label_ckpt = QLabel(self._ckpt_path)
        self.label_ckpt.setStyleSheet('color: #666;')
        ctrl.addWidget(self.label_ckpt, 1)

        ctrl.addWidget(QLabel('预测视界:'))
        self.spin_horizon = self._make_spin(500, 50, 10000, 0)
        ctrl.addWidget(self.spin_horizon)

        ctrl.addWidget(QLabel('采样点数:'))
        self.spin_samples = self._make_spin(40, 5, 200, 0)
        ctrl.addWidget(self.spin_samples)

        ctrl.addWidget(QLabel('预测步数:'))
        self.spin_n_steps = self._make_spin(40, 5, 500, 0)
        ctrl.addWidget(self.spin_n_steps)

        self.btn_run = QPushButton('▶ 开始评估')
        self.btn_run.clicked.connect(self._start)
        ctrl.addWidget(self.btn_run)

        self.btn_restart = QPushButton('↺ 重置')
        self.btn_restart.clicked.connect(self._restart)
        ctrl.addWidget(self.btn_restart)

        main_layout.addLayout(ctrl)

        # ── 第二行：进度 ──
        ctrl2 = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        ctrl2.addWidget(self.progress_bar, 1)
        self.label_status = QLabel('就绪')
        ctrl2.addWidget(self.label_status)
        main_layout.addLayout(ctrl2)

        # ── 主画布（全星图） ──
        self.overview = EvalOverviewCanvas()
        main_layout.addWidget(self.overview, 1)

        # ── 摘要区 ──
        summary_grp = QGroupBox('评估摘要')
        summary_layout = QVBoxLayout(summary_grp)
        self.text_summary = QTextEdit()
        self.text_summary.setReadOnly(True)
        self.text_summary.setMinimumHeight(160)
        self.text_summary.setStyleSheet('font-family: Consolas; font-size: 12px;')
        self.text_summary.setPlainText('等待评估完成…')
        summary_layout.addWidget(self.text_summary)
        main_layout.addWidget(summary_grp)

    @staticmethod
    def _make_spin(default, minv, maxv, decimals):
        if decimals:
            w = QDoubleSpinBox()
            w.setDecimals(decimals)
        else:
            w = QSpinBox()
        w.setRange(minv, maxv)
        w.setValue(default)
        return w

    def _choose_checkpoint(self):
        path, _ = QFileDialog.getOpenFileName(
            self, '选择 checkpoint 文件',
            os.path.join(os.getcwd(), 'checkpoints'),
            'PyTorch files (*.pt *.pth);;All files (*)')
        if path:
            self._ckpt_path = path
            self.label_ckpt.setText(path)

    def _start(self):
        self.btn_run.setEnabled(False)
        self.btn_load.setEnabled(False)
        self.progress_bar.setValue(0)
        self.label_status.setText('加载模型…')

        self._worker = EvaluateWorker(
            ckpt_path=self._ckpt_path,
            predict_horizon=self.spin_horizon.value(),
            n_predict_steps=self.spin_n_steps.value(),
            num_samples=self.spin_samples.value(),
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.gt_ready.connect(self._on_gt_ready)
        self._worker.segment_done.connect(self._on_segment_done)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(lambda *_: setattr(self, '_worker', None))
        self._worker.start()

    def _on_progress(self, pct, status):
        self.progress_bar.setValue(pct)
        self.label_status.setText(status)

    def _on_gt_ready(self, gt_data):
        self.overview.set_gt(gt_data)

    def _on_segment_done(self, data):
        acc_pred = data.get('accumulated_pred', {})
        self.overview.update_pred(acc_pred)

    def _on_finished(self, segments, horizons, extras):
        valid_ts = [h[1] for h in horizons if h[1] > 0 and not np.isnan(h[1])]
        all_errs = []
        for seg in segments:
            e = seg['err_body']
            if not np.all(np.isnan(e)):
                all_errs.append(np.nanmean(e))

        T = extras['gt_params']['T']
        n_cycles = extras.get('n_cycles', 1)
        nT = n_cycles * T
        lines = [
            f'评估范围: [{nT:.0f}, {(n_cycles+1)*T:.0f}]（{n_cycles} 周期训练后）',
            f'训练总时长          = {nT:.0f}（{n_cycles} × T，T={T:.0f}）',
            f'采样点数            = {len(segments)}',
            f'有效误差段数        = {len(all_errs)}',
        ]
        if valid_ts:
            lines += [
                f'可靠视界中位数      = {np.median(valid_ts):.1f}',
                f'可靠视界平均值      = {np.mean(valid_ts):.1f}',
                f'可靠视界最小值      = {np.min(valid_ts):.1f}',
                f'可靠视界最大值      = {np.max(valid_ts):.1f}',
                f'可靠视界/预测视界    = {np.mean(valid_ts) / extras.get("predict_horizon", 500) * 100:.1f}%',
            ]
        if all_errs:
            lines += [
                f'平均位置误差        = {np.mean(all_errs):.4e} (中位数={np.median(all_errs):.4e})',
                f'误差范围            = {np.min(all_errs):.4e} ~ {np.max(all_errs):.4e}',
            ]
        # 逐天体误差细分
        for b, name in enumerate(['A', 'B', 'C']):
            body_errs = []
            for seg in segments:
                e = seg['err_body']
                if e.shape[0] > b and not np.all(np.isnan(e[b])):
                    body_errs.append(np.nanmean(e[b]))
            if body_errs:
                lines.append(f'天体 {name} 平均位置误差 = {np.mean(body_errs):.4e} '
                             f'(中位数={np.median(body_errs):.4e})')

        summary_text = '\n'.join(lines)
        self.text_summary.setPlainText(summary_text)
        # 同时输出到控制台
        print('\n' + '=' * 50)
        print('  评  估  统  计')
        print('=' * 50)
        print(summary_text)
        print('=' * 50)

        self.label_status.setText('评估完成')
        self.btn_run.setEnabled(True)
        self.btn_load.setEnabled(True)

    def _on_error(self, msg):
        self.label_status.setText('出错')
        self.text_summary.setPlainText(f'[错误] {msg}')
        self.btn_run.setEnabled(True)
        self.btn_load.setEnabled(True)

    def _restart(self):
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait(2000)
        self._worker = None
        self.btn_run.setEnabled(True)
        self.btn_load.setEnabled(True)
        self.progress_bar.setValue(0)
        self.label_status.setText('就绪')
        self.text_summary.setPlainText('等待评估完成…')
        self.overview.clear()

    def closeEvent(self, event):
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait(2000)
        event.accept()


# ═══════════════════════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════════════════════

def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints/predictor_online.pt'
    app = QApplication(sys.argv)
    win = EvaluateWindow(ckpt_path)
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
