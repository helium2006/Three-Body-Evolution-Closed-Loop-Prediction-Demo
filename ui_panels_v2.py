"""
三体进化闭环预测 — UI 面板 v2（轨迹图版）
==========================================
标签页结构：
  参数设置 | 实时-A | 实时-B | 实时-C
  预测-A | 预测-B | 预测-C
  对比-A | 对比-B | 对比-C
  全星图（真实 + 预测拼接） | 模型统计

与 v1 的区别：实时 / 预测 / 对比全部使用 3D 轨迹图，
不再绘制 |r|、|v|、θ、φ 时间序列。
"""

import sys
import os
import io
import numpy as np
import torch

# Qt / Matplotlib 后端
import matplotlib
matplotlib.use('QtAgg')

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QTabWidget,
                                QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                                QProgressBar, QTextEdit, QGridLayout,
                                QGroupBox, QDoubleSpinBox, QCheckBox, QSpinBox,
                                QFileDialog)
from PySide6.QtCore import QThread, Signal, Qt

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from three_body import run_simulation, equilateral_ic, figure8_ic
from closed_loop_evolution import NeuralPredictor, MetaLearner


# ==================== 参数设置面板（复用 v1） ====================

class SettingsPanel(QGroupBox):
    """模拟参数设置面板。

    三种工作模式（互斥）：
      等边三角形  — 只需 R，自动填位置/速度
      8 字形      — 只需 Scale，三天体等质量，自动填位置/速度
      自动求解 C  — 用户自由填 A、B 的参数，C 由质心归零 + 动量归零解出
    """

    RED_BORDER = 'QDoubleSpinBox { border: 2px solid #e74c3c; }'
    NORMAL_BORDER = ''

    def __init__(self):
        super().__init__('模拟参数设置')
        self._updating = False

        grid = QGridLayout(self)
        grid.setVerticalSpacing(6)
        grid.setHorizontalSpacing(10)

        # ── Row 0: 等边三角形特解 ──
        self.chk_equilateral = QCheckBox('等边三角形特解（绕质心圆周运动）')
        self.chk_equilateral.setChecked(True)
        self.chk_equilateral.toggled.connect(self._on_equilateral_toggled)
        grid.addWidget(self.chk_equilateral, 0, 0, 1, 2)

        grid.addWidget(QLabel('R（边长）:'), 0, 2)
        self.spin_R = self._make_spin(10000.0, 1.0, 1e6, 1)
        self.spin_R.valueChanged.connect(self._on_R_changed)
        grid.addWidget(self.spin_R, 0, 3)

        grid.addWidget(QLabel('总时长 T:'), 0, 4)
        self.spin_T = self._make_spin(100000.0, 1.0, 1e9, 1)
        grid.addWidget(self.spin_T, 0, 5)

        # ── Row 1: 8 字形特解 ──
        self.chk_figure8 = QCheckBox('8字形特解（Chenciner-Montgomery 周期轨道）')
        self.chk_figure8.setChecked(False)
        self.chk_figure8.toggled.connect(self._on_figure8_toggled)
        grid.addWidget(self.chk_figure8, 1, 0, 1, 2)

        self.lbl_scale = QLabel('Scale（尺度）:')
        grid.addWidget(self.lbl_scale, 1, 2)
        self.spin_scale = self._make_spin(1.0, 0.01, 1e6, 2)
        self.spin_scale.valueChanged.connect(self._on_scale_changed)
        grid.addWidget(self.spin_scale, 1, 3)
        self.lbl_scale.setVisible(False)
        self.spin_scale.setVisible(False)

        # ── Row 2: 自动求解天体 C ──
        self.chk_auto_solve = QCheckBox('自动求解天体 C（质心归零 + 总动量归零）')
        self.chk_auto_solve.setChecked(False)
        self.chk_auto_solve.toggled.connect(self._on_auto_solve_toggled)
        grid.addWidget(self.chk_auto_solve, 2, 0, 1, 4)

        # ── Row 3: 质量 ──
        grid.addWidget(QLabel('mA:'), 3, 0)
        self.spin_mA = self._make_spin(1.0, 0.01, 1e6, 4)
        grid.addWidget(self.spin_mA, 3, 1)
        self.spin_mA.valueChanged.connect(self._on_mass_changed)

        grid.addWidget(QLabel('mB:'), 3, 2)
        self.spin_mB = self._make_spin(1.0, 0.01, 1e6, 4)
        grid.addWidget(self.spin_mB, 3, 3)

        grid.addWidget(QLabel('mC:'), 3, 4)
        self.spin_mC = self._make_spin(1.0, 0.01, 1e6, 4)
        grid.addWidget(self.spin_mC, 3, 5)

        # ── Row 4: 初始位置 ──
        lbl_pos = QLabel('── 初始位置 ──')
        lbl_pos.setStyleSheet('color: #888;')
        grid.addWidget(lbl_pos, 4, 0, 1, 6)

        self.pos_spins = []
        for bname, row in [('A', 5), ('B', 6), ('C', 7)]:
            grid.addWidget(QLabel(f'{bname}:'), row, 0)
            spins = []
            for ci, comp in enumerate(['x', 'y', 'z']):
                col_base = ci * 2 + 1
                grid.addWidget(QLabel(comp), row, col_base)
                sb = self._make_spin(0.0, -1e8, 1e8, 2)
                grid.addWidget(sb, row, col_base + 1)
                spins.append(sb)
            self.pos_spins.append(spins)

        # ── Row 8: 初始速度 ──
        lbl_vel = QLabel('── 初始速度 ──')
        lbl_vel.setStyleSheet('color: #888;')
        grid.addWidget(lbl_vel, 8, 0, 1, 6)

        self.vel_spins = []
        for bname, row in [('A', 9), ('B', 10), ('C', 11)]:
            grid.addWidget(QLabel(f'{bname}:'), row, 0)
            spins = []
            for ci, comp in enumerate(['vx', 'vy', 'vz']):
                col_base = ci * 2 + 1
                grid.addWidget(QLabel(comp), row, col_base)
                sb = self._make_spin(0.0, -1e8, 1e8, 6)
                grid.addWidget(sb, row, col_base + 1)
                spins.append(sb)
            self.vel_spins.append(spins)

        # 连接 A/B 参数变化信号（供自动求解 C 使用）
        self._auto_solve_signals_connected = False
        self._connect_auto_solve_signals()

        self._apply_equilateral(10000.0)

    # ═══════════════════════════════════════════════════
    #  基础工具
    # ═══════════════════════════════════════════════════

    def _make_spin(self, default, vmin, vmax, decimals):
        sb = QDoubleSpinBox()
        sb.setRange(vmin, vmax)
        sb.setDecimals(decimals)
        sb.setValue(default)
        sb.setSingleStep(10 ** (-decimals + 1) if decimals > 0 else 1)
        return sb

    def _set_pos_vel_spins(self, positions, velocities):
        self._updating = True
        for i in range(3):
            for j in range(3):
                self.pos_spins[i][j].setValue(positions[i][j])
                self.vel_spins[i][j].setValue(velocities[i][j])
        self._updating = False

    def _lock_pos_vel(self, locked, bodies=(0, 1, 2)):
        """锁定/解锁指定天体的位置和速度输入框。"""
        for bi in bodies:
            for sb in self.pos_spins[bi] + self.vel_spins[bi]:
                sb.setReadOnly(locked)
                sb.setStyleSheet(self.NORMAL_BORDER)

    def _sync_masses(self, m_val):
        self._updating = True
        self.spin_mA.setValue(m_val)
        self.spin_mB.setValue(m_val)
        self.spin_mC.setValue(m_val)
        self._updating = False

    def _highlight_missing(self):
        """标红缺失/不合理的参数。"""
        for sb in [self.spin_mA, self.spin_mB, self.spin_mC, self.spin_scale]:
            sb.setStyleSheet(self.NORMAL_BORDER)

        if self.chk_auto_solve.isChecked():
            if self.spin_mC.value() <= 0:
                self.spin_mC.setStyleSheet(self.RED_BORDER)
            return

        if not self.chk_figure8.isChecked():
            return

        mA, mB, mC = self.spin_mA.value(), self.spin_mB.value(), self.spin_mC.value()
        scale = self.spin_scale.value()

        if abs(mA - mB) > 1e-10 or abs(mA - mC) > 1e-10:
            for sb in [self.spin_mA, self.spin_mB, self.spin_mC]:
                sb.setStyleSheet(self.RED_BORDER)
        elif mA <= 0:
            self.spin_mA.setStyleSheet(self.RED_BORDER)

        if scale <= 0:
            self.spin_scale.setStyleSheet(self.RED_BORDER)

    # ═══════════════════════════════════════════════════
    #  等边三角形模式
    # ═══════════════════════════════════════════════════

    def _apply_equilateral(self, R):
        pos, vel = equilateral_ic(R)
        self._set_pos_vel_spins(pos, vel)

    def _on_equilateral_toggled(self, checked):
        if checked:
            self.chk_figure8.setChecked(False)
            self.chk_auto_solve.setChecked(False)
            self._lock_pos_vel(True)
            self.spin_R.setEnabled(True)
            self._apply_equilateral(self.spin_R.value())
        else:
            self._lock_pos_vel(False)
            self.spin_R.setEnabled(False)

    def _on_R_changed(self, val):
        if self._updating:
            return
        if self.chk_equilateral.isChecked():
            self._apply_equilateral(val)

    # ═══════════════════════════════════════════════════
    #  8 字形模式
    # ═══════════════════════════════════════════════════

    def _apply_figure8(self):
        scale = self.spin_scale.value()
        m_val = self.spin_mA.value()
        if scale > 0 and m_val > 0:
            pos, vel = figure8_ic(scale=scale, m=m_val)
            self._set_pos_vel_spins(pos, vel)
        self._highlight_missing()

    def _on_figure8_toggled(self, checked):
        self.lbl_scale.setVisible(checked)
        self.spin_scale.setVisible(checked)

        if checked:
            self.chk_equilateral.setChecked(False)
            self.chk_auto_solve.setChecked(False)
            self._lock_pos_vel(True)
            self.spin_R.setEnabled(False)
            m_val = self.spin_mA.value()
            if m_val <= 0:
                m_val = 1.0
            self._sync_masses(m_val)
            self._apply_figure8()
        else:
            self._lock_pos_vel(False)
            for sb in [self.spin_mA, self.spin_mB, self.spin_mC]:
                sb.setStyleSheet(self.NORMAL_BORDER)

    def _on_scale_changed(self, val):
        if self._updating:
            return
        if self.chk_figure8.isChecked():
            self._apply_figure8()

    def _on_mass_changed(self, val):
        if self._updating:
            return
        if self.chk_figure8.isChecked():
            self._sync_masses(self.spin_mA.value())
            self._apply_figure8()
        elif self.chk_auto_solve.isChecked():
            self._apply_auto_solve()

    # ═══════════════════════════════════════════════════
    #  自动求解天体 C 模式
    # ═══════════════════════════════════════════════════

    def _connect_auto_solve_signals(self):
        """连接 A/B/C 参数变化信号（始终连接，通过 _updating 防止循环）。
        
        mA 已有 _on_mass_changed 处理，此处只额外连接 mB、mC 以及 A/B 的位置/速度。
        """
        for sb in [self.spin_mB, self.spin_mC]:
            try:
                sb.valueChanged.disconnect(self._on_auto_solve_param_changed)
            except (TypeError, RuntimeError):
                pass
            sb.valueChanged.connect(self._on_auto_solve_param_changed)
        # A 和 B 的位置 + 速度
        for bi in (0, 1):
            for sb in self.pos_spins[bi] + self.vel_spins[bi]:
                sb.valueChanged.connect(self._on_auto_solve_param_changed)

    def _on_auto_solve_param_changed(self, _val=None):
        """A 或 B 的任意参数变化时，重新求解 C。"""
        if self._updating:
            return
        if self.chk_auto_solve.isChecked():
            self._apply_auto_solve()

    def _apply_auto_solve(self):
        """根据质心归零和总动量归零约束求解天体 C 的位置和速度。

        r_C = -(mA*r_A + mB*r_B) / mC
        v_C = -(mA*v_A + mB*v_B) / mC
        """
        mA = self.spin_mA.value()
        mB = self.spin_mB.value()
        mC = self.spin_mC.value()

        self._highlight_missing()
        if mC <= 0:
            return

        rA = np.array([sb.value() for sb in self.pos_spins[0]])
        rB = np.array([sb.value() for sb in self.pos_spins[1]])
        vA = np.array([sb.value() for sb in self.vel_spins[0]])
        vB = np.array([sb.value() for sb in self.vel_spins[1]])

        rC = -(mA * rA + mB * rB) / mC
        vC = -(mA * vA + mB * vB) / mC

        self._updating = True
        for j in range(3):
            self.pos_spins[2][j].setValue(rC[j])
            self.vel_spins[2][j].setValue(vC[j])
        self._updating = False

    def _on_auto_solve_toggled(self, checked):
        if checked:
            self.chk_equilateral.setChecked(False)
            self.chk_figure8.setChecked(False)
            # 解锁 A、B，锁定 C
            self._lock_pos_vel(False, bodies=(0, 1))
            self._lock_pos_vel(True, bodies=(2,))
            self.spin_R.setEnabled(False)
            self._apply_auto_solve()
        else:
            self._lock_pos_vel(False)
            self.spin_mC.setStyleSheet(self.NORMAL_BORDER)

    # ═══════════════════════════════════════════════════
    #  导出参数
    # ═══════════════════════════════════════════════════

    def get_params(self):
        return {
            'T': self.spin_T.value(),
            'm': [self.spin_mA.value(), self.spin_mB.value(), self.spin_mC.value()],
            'r': [[sb.value() for sb in row] for row in self.pos_spins],
            'v': [[sb.value() for sb in row] for row in self.vel_spins],
        }


# ==================== 实时监测 — 3D 轨迹 ====================

class BodyMonitorTrajectoryCanvas(QWidget):
    """单个天体实时 3D 轨迹图。"""

    BODY_COLORS = {'A': '#c0392b', 'B': '#27ae60', 'C': '#2980b9'}

    def __init__(self, body_name, body_idx):
        super().__init__()
        self.body_idx = body_idx
        self.body_name = body_name
        self.body_color = self.BODY_COLORS[body_name]
        layout = QVBoxLayout(self)
        self.fig = Figure(figsize=(7, 6), dpi=120)
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('#fafafa')
        self.ax.set_xlabel('X'); self.ax.set_ylabel('Y'); self.ax.set_zlabel('Z')
        self.ax.set_title(f'天体 {body_name}  实时轨迹', fontsize=13, fontweight='bold')
        (self.line,) = self.ax.plot([], [], [], '-', color=self.body_color, lw=1.3)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def update_data(self, t_hist, r_all, v_all):
        r_body = np.array(r_all)[:, self.body_idx, :]
        self.line.set_data(r_body[:, 0], r_body[:, 1])
        self.line.set_3d_properties(r_body[:, 2])
        self._auto_limits(r_body)
        self.canvas.draw_idle()

    def _auto_limits(self, xyz):
        x_min, x_max = xyz[:, 0].min(), xyz[:, 0].max()
        y_min, y_max = xyz[:, 1].min(), xyz[:, 1].max()
        z_min, z_max = xyz[:, 2].min(), xyz[:, 2].max()
        xy_range = max(x_max - x_min, y_max - y_min, z_max - z_min) * 0.08 + 1.0
        self.ax.set_xlim(x_min - xy_range, x_max + xy_range)
        self.ax.set_ylim(y_min - xy_range, y_max + xy_range)
        self.ax.set_zlim(z_min - xy_range, z_max + xy_range)


# ==================== 预测 — 3D 轨迹（实线=观测 / 虚线=预测外推） ====================

class BodyPredictionTrajectoryCanvas(QWidget):
    """单个天体预测 3D 轨迹：实线=观测，虚线=预测外推。"""

    BODY_COLORS = {'A': '#c0392b', 'B': '#27ae60', 'C': '#2980b9'}

    def __init__(self, body_name, body_idx):
        super().__init__()
        self.body_idx = body_idx
        self.body_name = body_name
        self.body_color = self.BODY_COLORS[body_name]
        layout = QVBoxLayout(self)
        self.fig = Figure(figsize=(7, 6), dpi=120)
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('#fafafa')
        self.ax.set_xlabel('X'); self.ax.set_ylabel('Y'); self.ax.set_zlabel('Z')
        self.ax.set_title(f'天体 {body_name}  观测 + 预测', fontsize=13, fontweight='bold')

        (self.line_obs,)  = self.ax.plot([], [], [], '-',
                                         color=self.body_color, lw=1.4, label='观测')
        (self.line_pred,) = self.ax.plot([], [], [], '-',
                                         color='#e67e22', lw=1.2, alpha=0.85, label='预测')
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def update_data(self, t_obs, r_obs_all, v_obs_all, t_pred, r_pred_all, v_pred_all):
        # 观测
        if len(t_obs) > 0:
            r_obs = np.array(r_obs_all)[:, self.body_idx, :]
            self.line_obs.set_data(r_obs[:, 0], r_obs[:, 1])
            self.line_obs.set_3d_properties(r_obs[:, 2])

        # 预测（跳过第 0 点，避免与观测终点重叠）
        if len(t_pred) > 1:
            r_pred = np.array(r_pred_all)[1:, self.body_idx, :]
            self.line_pred.set_data(r_pred[:, 0], r_pred[:, 1])
            self.line_pred.set_3d_properties(r_pred[:, 2])
        else:
            self.line_pred.set_data([], [])
            self.line_pred.set_3d_properties([])

        self._auto_limits(t_obs, r_obs_all, t_pred, r_pred_all)
        self.canvas.draw_idle()

    def _auto_limits(self, t_obs, r_obs_all, t_pred, r_pred_all):
        """合并观测与预测范围，统一坐标轴。"""
        all_xyz = []
        if len(t_obs) > 0:
            all_xyz.append(np.array(r_obs_all)[:, self.body_idx, :])
        if len(t_pred) > 1:
            all_xyz.append(np.array(r_pred_all)[1:, self.body_idx, :])
        if not all_xyz:
            return
        xyz = np.concatenate(all_xyz)
        x_min, x_max = xyz[:, 0].min(), xyz[:, 0].max()
        y_min, y_max = xyz[:, 1].min(), xyz[:, 1].max()
        z_min, z_max = xyz[:, 2].min(), xyz[:, 2].max()
        margin = max(x_max - x_min, y_max - y_min, z_max - z_min) * 0.08 + 1.0
        self.ax.set_xlim(x_min - margin, x_max + margin)
        self.ax.set_ylim(y_min - margin, y_max + margin)
        self.ax.set_zlim(z_min - margin, z_max + margin)


# ==================== 轨迹对比 — 观测 vs 真实叠加 ====================

class BodyTrajectoryCompareCanvas(QWidget):
    """单个天体观测轨迹 vs 真实轨迹叠加 3D 图。"""

    BODY_COLORS = {'A': '#c0392b', 'B': '#27ae60', 'C': '#2980b9'}

    def __init__(self, body_name, body_idx):
        super().__init__()
        self.body_idx = body_idx
        self.body_name = body_name
        self.body_color = self.BODY_COLORS[body_name]
        layout = QVBoxLayout(self)
        self.fig = Figure(figsize=(7, 6), dpi=120)
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('#fafafa')
        self.ax.set_xlabel('X'); self.ax.set_ylabel('Y'); self.ax.set_zlabel('Z')
        self.ax.set_title(f'天体 {body_name}  观测 vs 真实', fontsize=13, fontweight='bold')

        (self.line_obs,)  = self.ax.plot([], [], [], '-',
                                         color=self.body_color, lw=1.2, alpha=0.7, label='观测')
        (self.line_true,) = self.ax.plot([], [], [], '--',
                                         color='#e67e22', lw=1.2, alpha=0.85, label='真实')
        self.ax.legend(fontsize=9)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def plot_results(self, results):
        self.ax.clear()
        self.ax.set_facecolor('#fafafa')
        self.ax.set_xlabel('X'); self.ax.set_ylabel('Y'); self.ax.set_zlabel('Z')
        self.ax.set_title(f'天体 {self.body_name}  观测 vs 真实',
                          fontsize=13, fontweight='bold')

        obs = results['obs']
        gt  = results['gt']

        r_obs = obs['r_obs'][:, self.body_idx, :]
        r_gt  = gt['r'][:, self.body_idx, :]

        self.ax.plot(r_obs[:, 0], r_obs[:, 1], r_obs[:, 2],
                     '-', color=self.body_color, lw=1.2, alpha=0.7, label='观测')
        self.ax.plot(r_gt[:, 0], r_gt[:, 1], r_gt[:, 2],
                     '--', color='#e67e22', lw=1.2, alpha=0.85, label='真实')
        self.ax.legend(fontsize=9)

        all_xyz = np.concatenate([r_obs, r_gt])
        x_min, x_max = all_xyz[:, 0].min(), all_xyz[:, 0].max()
        y_min, y_max = all_xyz[:, 1].min(), all_xyz[:, 1].max()
        z_min, z_max = all_xyz[:, 2].min(), all_xyz[:, 2].max()
        margin = max(x_max - x_min, y_max - y_min, z_max - z_min) * 0.08 + 1.0
        self.ax.set_xlim(x_min - margin, x_max + margin)
        self.ax.set_ylim(y_min - margin, y_max + margin)
        self.ax.set_zlim(z_min - margin, z_max + margin)

        self.fig.tight_layout()
        self.canvas.draw_idle()


# ==================== 全星图 — 真实全星图 + 拼接预测全星图 ====================

class AllBodiesOverviewCanvas(QWidget):
    """左右两张 3D 图：
    左：ABC 三条真实轨迹同画
    右：ABC 三条拼接预测轨迹同画
    """

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
            ax.text2D(0.5, 0.5, '等待闭环完成…', ha='center', va='center',
                      transform=ax.transAxes, fontsize=14, color='#ccc')

        self.ax_gt.set_title('真实轨迹（ABC 同框）', fontsize=13, fontweight='bold')
        self.ax_pred.set_title('拼接预测（ABC 同框）', fontsize=13, fontweight='bold')

        self.fig.tight_layout()
        self.canvas.draw_idle()

    def plot_results(self, results):
        self.fig.clear()

        self.ax_gt   = self.fig.add_subplot(1, 2, 1, projection='3d')
        self.ax_pred = self.fig.add_subplot(1, 2, 2, projection='3d')

        for ax in [self.ax_gt, self.ax_pred]:
            ax.set_facecolor('#fafafa')
            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

        self.ax_gt.set_title('真实轨迹（ABC 同框）', fontsize=13, fontweight='bold')
        self.ax_pred.set_title('拼接预测（ABC 同框）', fontsize=13, fontweight='bold')

        gt = results['gt']
        self._draw_gt_trajectories(gt['r'])

        sp = results.get('stitched_pred', {})
        has_pred = len(sp.get('t', [])) > 0 and sp['r'].shape[0] > 0
        if has_pred:
            self._draw_pred_trajectories(sp['r'])
        else:
            self.ax_pred.text2D(0.5, 0.5, '无拼接预测数据', ha='center', va='center',
                                transform=self.ax_pred.transAxes, fontsize=14, color='#aaa')

        self.fig.tight_layout()
        self.canvas.draw_idle()

    def update_overview_realtime(self, t_obs, r_obs_list, v_obs_list,
                                  t_stitch, r_stitch_list, v_stitch_list):
        """实时更新全星图。左 = 已累积的观测轨迹，右 = 已累积的拼接预测。"""
        self.fig.clear()

        self.ax_gt   = self.fig.add_subplot(1, 2, 1, projection='3d')
        self.ax_pred = self.fig.add_subplot(1, 2, 2, projection='3d')

        for ax in [self.ax_gt, self.ax_pred]:
            ax.set_facecolor('#fafafa')
            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

        self.ax_gt.set_title('已观测轨迹（ABC 同框）', fontsize=13, fontweight='bold')
        self.ax_pred.set_title('拼接预测（ABC 同框）', fontsize=13, fontweight='bold')

        # ── 左图：已观测轨迹 ──
        if r_obs_list and len(r_obs_list) > 0:
            r_obs_arr = np.array(r_obs_list)
            self._draw_gt_trajectories(r_obs_arr)
        else:
            self.ax_gt.text2D(0.5, 0.5, '等待数据…', ha='center', va='center',
                              transform=self.ax_gt.transAxes, fontsize=14, color='#ccc')

        # ── 右图：拼接预测 ──
        if r_stitch_list and len(r_stitch_list) > 0:
            r_stitch_arr = np.array(r_stitch_list)
            self._draw_pred_trajectories(r_stitch_arr)
        else:
            self.ax_pred.text2D(0.5, 0.5, '等待数据…', ha='center', va='center',
                                transform=self.ax_pred.transAxes, fontsize=14, color='#ccc')

        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _draw_gt_trajectories(self, r_arr):
        """左图：以实线绘制 ABC 真实/观测轨迹。"""
        all_xyz = []
        for idx, name in enumerate(['A', 'B', 'C']):
            r = r_arr[:, idx, :]
            all_xyz.append(r)
            self.ax_gt.plot(r[:, 0], r[:, 1], r[:, 2],
                            '-', color=self.BODY_COLORS[name], lw=1.2,
                            alpha=0.9, label=f'天体 {name}')
        if all_xyz:
            xyz = np.concatenate(all_xyz)
            margin = max(np.ptp(xyz[:, 0]), np.ptp(xyz[:, 1]), np.ptp(xyz[:, 2])) * 0.08 + 1.0
            self.ax_gt.set_xlim(xyz[:, 0].min() - margin, xyz[:, 0].max() + margin)
            self.ax_gt.set_ylim(xyz[:, 1].min() - margin, xyz[:, 1].max() + margin)
            self.ax_gt.set_zlim(xyz[:, 2].min() - margin, xyz[:, 2].max() + margin)
        self.ax_gt.legend(fontsize=8)

    def _draw_pred_trajectories(self, r_arr):
        """右图：以虚线绘制 ABC 预测轨迹。"""
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


# ==================== 模型统计 Canvas（复用 v1） ====================

class ModelStatsCanvas(QWidget):
    """模型变化统计：误差曲线、反馈累计、元学习器、进化概览。"""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self.fig = Figure(figsize=(14, 9), dpi=120)
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        axes = self.fig.subplots(2, 2)
        self.fig.suptitle('模型进化统计', fontsize=15, fontweight='bold')
        for ax_row in axes:
            for ax in ax_row:
                ax.set_facecolor('#f8f8f8')
                ax.text(0.5, 0.5, '等待闭环完成…', ha='center', va='center',
                        transform=ax.transAxes, fontsize=14, color='#aaa')
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def plot_results(self, results):
        self.fig.clear()
        axes = self.fig.subplots(2, 2)
        self.fig.suptitle('模型进化统计', fontsize=15, fontweight='bold')

        T_total = results['gt']['t'][-1]
        fb_times = [e['t'] for e in results['feedback_log']]
        fb_errs  = [e['error'] for e in results['feedback_log']]

        # (0,0): 预测误差随时间变化
        ax = axes[0, 0]
        errs = results['error_history']
        err_times = results.get('error_times', [])
        if len(err_times) == len(errs) and len(errs) > 0:
            ax.semilogy(err_times, errs, 'b-', lw=1.2, alpha=0.8)
        if fb_times:
            ax.scatter(fb_times, fb_errs, c='r', s=50, zorder=5, marker='x',
                       linewidths=1.5, label='反馈触发')
        ax.axhline(results.get('feedback_threshold', 1.0), c='gray', ls='--',
                   lw=1, label='反馈阈值')
        ax.set_xlabel('宇宙时间 t'); ax.set_ylabel('预测 MSE')
        ax.set_title('在线预测误差变化')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # (0,1): 累计反馈次数
        ax = axes[0, 1]
        if fb_times:
            ax.fill_between(fb_times, range(1, len(fb_times) + 1), alpha=0.3,
                            color='orange', step='post')
            ax.step(fb_times, range(1, len(fb_times) + 1), 'r-', where='post',
                    lw=1.8, label=f'累计 ({len(fb_times)}次)')
            if len(fb_times) > 5:
                ax.axvline(x=fb_times[min(5, len(fb_times) - 1)],
                           c='purple', ls=':', lw=1.5, label='元学习器参与')
        ax.set_xlabel('宇宙时间 t'); ax.set_ylabel('累计反馈次数')
        ax.set_title('反馈累计曲线')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # (1,0): 元学习器学习率建议
        ax = axes[1, 0]
        fl = results['feedback_log']
        if fl:
            lrs   = [e.get('meta_lr', 0.001) for e in fl]
            confs = [e.get('meta_conf', 0) for e in fl]
            fb_idx = range(1, len(fl) + 1)
            bars = ax.bar(fb_idx, lrs, color='steelblue', alpha=0.7, label='建议学习率')
            ax.axhline(y=0.001, c='gray', ls='--', lw=1, label='默认 lr=0.001')
            ax2 = ax.twinx()
            ax2.plot(fb_idx, confs, 'm-o', ms=6, lw=1.5, label='元学习置信度')
            ax2.set_ylim(-0.05, 1.15)
            ax.set_xlabel('反馈序号'); ax.set_ylabel('学习率')
            ax2.set_ylabel('置信度')
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=7, loc='upper left')
            ax.set_title('元学习器辅助决策')
            ax.grid(True, alpha=0.3, axis='y')

        # (1,1): 进化统计概览
        ax = axes[1, 1]
        n_snaps    = len(results['snapshots'])
        n_feedbacks = len(results['feedback_log'])
        n_segments  = len(results['error_history'])
        stats_labels = ['反馈次数', '快照保存', '总段数', 'T / 千']
        stats_vals   = [n_feedbacks, n_snaps, n_segments, T_total / 1000]
        colors_bar   = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
        bars = ax.barh(stats_labels, stats_vals, color=colors_bar, alpha=0.85)
        for bar, val in zip(bars, stats_vals):
            ax.text(bar.get_width() + max(stats_vals) * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:.1f}' if isinstance(val, float) and abs(val - int(val)) > 1e-6 else str(int(val)),
                    va='center', fontsize=10, fontweight='bold')
        ax.set_title('进化统计概览')

        self.fig.tight_layout()
        self.canvas.draw_idle()


# ==================== 闭环工作线程（复用 v1 逻辑） ====================

class ClosedLoopWorker(QThread):
    """后台闭环进化（仿真+预测同步），实时汇报观测和预测数据。"""
    progress = Signal(int, str)
    monitor_data = Signal(list, list, list)
    prediction_data = Signal(list, list, list)
    overview_update = Signal(object, object, object, object, object, object)
    log_msg = Signal(str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, sim_params, noise_level, feedback_threshold,
                 predict_horizon, segment_length, n_predict_steps,
                 init_frac, pretrain_epochs, meta_trigger, device,
                 continue_state=None):
        super().__init__()
        self.sim_params = sim_params
        self.noise_level = noise_level
        self.feedback_threshold = feedback_threshold
        self.predict_horizon = predict_horizon
        self.segment_length = segment_length
        self.n_predict_steps = n_predict_steps
        self.init_frac = init_frac
        self.pretrain_epochs = pretrain_epochs
        self.meta_trigger = meta_trigger
        self.device = device
        self.continue_state = continue_state

        self._t_obs = []
        self._r_obs = []
        self._v_obs = []
        self._t_offset = 0.0
        self._t_seg_start = 0.0  # 当前段起始时间，供预测绘图截断观测用

        self._stitch_t = []
        self._stitch_r = []
        self._stitch_v = []

    def _on_progress(self, phase, data):
        if phase == 'init_done':
            gt = data['gt_init']
            self._t_obs = list(gt['t'])
            self._r_obs = list(gt['r'])
            self._v_obs = list(gt['v'])
            self._t_offset = data['t_current']
            self.monitor_data.emit(self._t_obs, self._r_obs, self._v_obs)
            pct = int(data['t_current'] / data['T'] * 100)
            self.progress.emit(pct, f'初始观测完成 t={data["t_current"]:.0f}')

        elif phase == 'continue_start':
            # 继续模式：从上一轮终点接续，_offset 设为当前时间
            self._t_offset = data['t_current']
            self._t_obs = [self._t_offset]
            r0 = self.continue_state['r_current'] if self.continue_state else np.zeros((3, 3))
            v0 = self.continue_state['v_current'] if self.continue_state else np.zeros((3, 3))
            self._r_obs = [r0]
            self._v_obs = [v0]
            self.monitor_data.emit(self._t_obs, self._r_obs, self._v_obs)
            pct = 0
            T_seg = data.get('T_segment', data['T'] - data['t_current'])
            self.progress.emit(pct, f'继续运行 t={data["t_current"]:.0f} T+={T_seg:.0f}')

        elif phase == 'segment_done':
            seg = data['seg_gt']
            for i in range(1, len(seg['t'])):
                self._t_obs.append(self._t_offset + seg['t'][i])
                self._r_obs.append(seg['r'][i])
                self._v_obs.append(seg['v'][i])
            self._t_offset += seg['t'][-1]
            self.monitor_data.emit(self._t_obs, self._r_obs, self._v_obs)

            # 预测数据 — 跳过第 0 点；记录段起始用于观测截断
            pred = data['prediction']
            t_seg_start = data['t_seg_start']
            lookahead = data['seg_gt']['t'][-1]
            self._t_seg_start = t_seg_start
            t_pred = [t_seg_start + pt for pt in pred['t'][1:]]
            self.prediction_data.emit(t_pred, list(pred['r'][1:]), list(pred['v'][1:]))

            # 累积拼接预测轨迹
            mask = pred['t'] <= lookahead + 1e-10
            n_pts = int(mask.sum())
            if n_pts > 1:
                self._stitch_t.append(t_seg_start + pred['t'][mask])
                self._stitch_r.append(pred['r'][mask])
                self._stitch_v.append(pred['v'][mask])

            # 发射全星图实时更新数据
            self.overview_update.emit(
                self._t_obs, list(np.array(self._r_obs)), list(np.array(self._v_obs)),
                list(np.concatenate(self._stitch_t)) if self._stitch_t else [],
                list(np.concatenate(self._stitch_r)) if self._stitch_r else [],
                list(np.concatenate(self._stitch_v)) if self._stitch_v else [],
            )

            pct = int(data['t_current'] / data['T'] * 100)
            status = f'段{data["segment_num"]} t={data["t_current"]:.0f}/{data["T"]:.0f}'
            if data['feedback']:
                status += f' ← 反馈#{data["feedback_count"]}'
            self.progress.emit(pct, status)

        elif phase == 'done':
            stitched_pred = {}
            if self._stitch_t:
                stitched_pred['t'] = np.concatenate(self._stitch_t)
                stitched_pred['r'] = np.concatenate(self._stitch_r)
                stitched_pred['v'] = np.concatenate(self._stitch_v)
            else:
                stitched_pred['t'] = np.array([])
                stitched_pred['r'] = np.empty((0, 3, 3))
                stitched_pred['v'] = np.empty((0, 3, 3))
            data['stitched_pred'] = stitched_pred

            self.progress.emit(100, '闭环完成，绘制结果…')
            self.finished.emit(data)

    def run(self):
        try:
            from closed_loop_evolution import ClosedLoopEvolution

            self.log_msg.emit(f'设备: {self.device}')
            self.progress.emit(0, '初始化闭环系统…')

            system = ClosedLoopEvolution(
                ground_truth_params=self.sim_params,
                noise_level=self.noise_level,
                feedback_threshold=self.feedback_threshold,
                predict_horizon=self.predict_horizon,
                segment_length=self.segment_length,
                n_predict_steps=self.n_predict_steps,
                init_frac=self.init_frac,
                device=self.device,
            )

            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                system.run(
                    pretrain_epochs=self.pretrain_epochs,
                    meta_trigger=self.meta_trigger,
                    verbose=True,
                    progress_callback=self._on_progress,
                    continue_state=self.continue_state,
                )
            finally:
                captured = sys.stdout.getvalue()
                sys.stdout = old_stdout
                for line in captured.split('\n'):
                    if line.strip():
                        self.log_msg.emit(line.strip())

        except Exception as e:
            import traceback
            self.error.emit(f'{e}\n{traceback.format_exc()}')


# ==================== 主窗口 v2 ====================

class MainWindowV2(QMainWindow):
    """三体进化闭环预测主窗口 v2（轨迹图版）
    标签页：参数设置 | 实时-ABC | 预测-ABC | 对比-ABC | 全星图 | 模型统计。"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle('三体进化闭环预测 — v2 轨迹图版')
        self.resize(1500, 950)
        self._pred_results = None
        self._worker = None
        self._final_state = None  # 供"继续"按钮使用的上一轮终点状态

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(8)

        # ---- 控制栏 ----
        ctrl = QHBoxLayout()
        self.btn_start = QPushButton('▶ 开始闭环进化')
        self.btn_start.clicked.connect(self._start_closed_loop)
        ctrl.addWidget(self.btn_start)

        self.btn_continue = QPushButton('⏩ 继续下一段 T')
        self.btn_continue.clicked.connect(self._continue_closed_loop)
        self.btn_continue.setEnabled(False)
        ctrl.addWidget(self.btn_continue)

        self.btn_load = QPushButton('📂 加载继续')
        self.btn_load.clicked.connect(self._load_checkpoint_and_continue)
        ctrl.addWidget(self.btn_load)

        self.btn_restart = QPushButton('↺ 重启')
        self.btn_restart.clicked.connect(self._restart)
        ctrl.addWidget(self.btn_restart)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        ctrl.addWidget(self.progress_bar, 1)

        self.label_status = QLabel('就绪')
        ctrl.addWidget(self.label_status)
        main_layout.addLayout(ctrl)

        # ---- 主标签页 ----
        self.tabs = QTabWidget()

        # 0: 参数设置
        self.settings = SettingsPanel()
        self.tabs.addTab(self.settings, '参数设置')

        # 1-3: 实时监测 ABC（3D 轨迹）
        names = ['A', 'B', 'C']
        self.monitor_tabs = []
        for idx, name in enumerate(names):
            tab = BodyMonitorTrajectoryCanvas(name, idx)
            self.monitor_tabs.append(tab)
            self.tabs.addTab(tab, f'实时-{name}')

        # 4-6: 预测 ABC（3D 轨迹）
        self.pred_tabs = []
        for idx, name in enumerate(names):
            tab = BodyPredictionTrajectoryCanvas(name, idx)
            self.pred_tabs.append(tab)
            self.tabs.addTab(tab, f'预测-{name}')

        # 7-9: 轨迹对比 ABC（观测 vs 真实叠加）
        self.compare_tabs = []
        for idx, name in enumerate(names):
            tab = BodyTrajectoryCompareCanvas(name, idx)
            self.compare_tabs.append(tab)
            self.tabs.addTab(tab, f'对比-{name}')

        # 10: 全星图（ABC 真实 + ABC 预测拼接）
        self.overview_canvas = AllBodiesOverviewCanvas()
        self.tabs.addTab(self.overview_canvas, '全星图')

        # 11: 模型统计
        self.model_stats_canvas = ModelStatsCanvas()
        self.tabs.addTab(self.model_stats_canvas, '模型统计')

        main_layout.addWidget(self.tabs, 1)

        # ---- 控制与日志区 ----
        bottom = QHBoxLayout()

        pred_ctrl = QGroupBox('闭环预测参数')
        grid = QGridLayout(pred_ctrl)
        grid.setVerticalSpacing(4)
        grid.setHorizontalSpacing(6)

        grid.addWidget(QLabel('噪声:'), 0, 0)
        self.pred_noise = self._make_spin(0.02, 0.0, 1.0, 3)
        grid.addWidget(self.pred_noise, 0, 1)

        grid.addWidget(QLabel('反馈阈值:'), 0, 2)
        self.pred_fb_thresh = self._make_spin(0.5, 0.01, 100.0, 2)
        grid.addWidget(self.pred_fb_thresh, 0, 3)

        grid.addWidget(QLabel('预测视野:'), 0, 4)
        self.pred_horizon_spin = QSpinBox()
        self.pred_horizon_spin.setRange(50, 10000)
        self.pred_horizon_spin.setValue(500)
        self.pred_horizon_spin.setSingleStep(50)
        grid.addWidget(self.pred_horizon_spin, 0, 5)

        grid.addWidget(QLabel('步长:'), 1, 0)
        self.pred_seg_len_spin = QSpinBox()
        self.pred_seg_len_spin.setRange(50, 10000)
        self.pred_seg_len_spin.setValue(500)
        self.pred_seg_len_spin.setSingleStep(50)
        grid.addWidget(self.pred_seg_len_spin, 1, 1)

        grid.addWidget(QLabel('预测步数:'), 1, 2)
        self.pred_n_steps_spin = QSpinBox()
        self.pred_n_steps_spin.setRange(5, 200)
        self.pred_n_steps_spin.setValue(40)
        grid.addWidget(self.pred_n_steps_spin, 1, 3)

        grid.addWidget(QLabel('初始观测:'), 1, 4)
        self.pred_init_frac = self._make_spin(0.2, 0.05, 0.5, 2)
        grid.addWidget(self.pred_init_frac, 1, 5)

        grid.addWidget(QLabel('预训练:'), 2, 0)
        self.pred_pretrain_spin = QSpinBox()
        self.pred_pretrain_spin.setRange(10, 500)
        self.pred_pretrain_spin.setValue(50)
        self.pred_pretrain_spin.setSingleStep(10)
        grid.addWidget(self.pred_pretrain_spin, 2, 1)

        grid.addWidget(QLabel('元学习:'), 2, 2)
        self.pred_meta_spin = QSpinBox()
        self.pred_meta_spin.setRange(1, 50)
        self.pred_meta_spin.setValue(5)
        grid.addWidget(self.pred_meta_spin, 2, 3)

        bottom.addWidget(pred_ctrl)

        self.pred_log = QTextEdit()
        self.pred_log.setReadOnly(True)
        self.pred_log.setMaximumHeight(80)
        bottom.addWidget(self.pred_log, 1)

        main_layout.addLayout(bottom)

    def _make_spin(self, default, vmin, vmax, decimals):
        sb = QDoubleSpinBox()
        sb.setRange(vmin, vmax)
        sb.setDecimals(decimals)
        sb.setValue(default)
        sb.setSingleStep(10 ** (-decimals + 1) if decimals > 0 else 1)
        return sb

    # ---------- 闭环控制 ----------

    def _start_closed_loop(self):
        self.btn_start.setEnabled(False)
        self.btn_continue.setEnabled(False)
        self._final_state = None
        self.progress_bar.setValue(0)
        self.label_status.setText('初始化…')
        self.pred_log.clear()

        sim_params = self.settings.get_params()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.pred_log.append(f'=== 闭环进化 ===  T={sim_params["T"]:.0f}  设备={device}')
        if device == 'cuda':
            self.pred_log.append(f'  GPU: {torch.cuda.get_device_name(0)}')

        self._worker = ClosedLoopWorker(
            sim_params=sim_params,
            noise_level=self.pred_noise.value(),
            feedback_threshold=self.pred_fb_thresh.value(),
            predict_horizon=self.pred_horizon_spin.value(),
            segment_length=self.pred_seg_len_spin.value(),
            n_predict_steps=self.pred_n_steps_spin.value(),
            init_frac=self.pred_init_frac.value(),
            pretrain_epochs=self.pred_pretrain_spin.value(),
            meta_trigger=self.pred_meta_spin.value(),
            device=device,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.monitor_data.connect(self._on_monitor)
        self._worker.prediction_data.connect(self._on_prediction)
        self._worker.overview_update.connect(self._on_overview_update)
        self._worker.log_msg.connect(self._on_log)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _continue_closed_loop(self):
        """基于上一轮终点状态，再运行一段总时长 T。"""
        if self._final_state is None:
            self.pred_log.append('[警告] 没有可用的终点状态，请先完成一次运行')
            return

        self.btn_start.setEnabled(False)
        self.btn_continue.setEnabled(False)
        self.progress_bar.setValue(0)
        self.label_status.setText('继续运行…')
        self.pred_log.append(f'=== 继续下一段 T（从 t={self._final_state["t_current"]:.0f} 开始） ===')

        sim_params = self.settings.get_params()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self._worker = ClosedLoopWorker(
            sim_params=sim_params,
            noise_level=self.pred_noise.value(),
            feedback_threshold=self.pred_fb_thresh.value(),
            predict_horizon=self.pred_horizon_spin.value(),
            segment_length=self.pred_seg_len_spin.value(),
            n_predict_steps=self.pred_n_steps_spin.value(),
            init_frac=self.pred_init_frac.value(),
            pretrain_epochs=self.pred_pretrain_spin.value(),
            meta_trigger=self.pred_meta_spin.value(),
            device=device,
            continue_state=self._final_state,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.monitor_data.connect(self._on_monitor)
        self._worker.prediction_data.connect(self._on_prediction)
        self._worker.overview_update.connect(self._on_overview_update)
        self._worker.log_msg.connect(self._on_log)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _load_checkpoint_and_continue(self):
        """从磁盘加载已保存的 checkpoint，重建 continue_state 以便继续训练。"""
        path, _ = QFileDialog.getOpenFileName(
            self, '选择 checkpoint 文件',
            os.path.join(os.getcwd(), 'checkpoints'),
            'PyTorch files (*.pt *.pth);;All files (*)')
        if not path:
            return

        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            ckpt = torch.load(path, map_location=device, weights_only=False)

            # ── 重建 predictor ──
            hidden_dims = ckpt.get('hidden_dims', (128, 128, 128))
            lr = ckpt.get('learning_rate', 0.001)
            predictor = NeuralPredictor(hidden_dims=hidden_dims, learning_rate=lr, device=device)
            predictor.dynamics.load_state_dict(ckpt['dynamics'])
            predictor.optimizer.load_state_dict(ckpt['optimizer'])
            predictor._metadata_history = ckpt.get('metadata', [])

            extras = ckpt.get('extras', {})
            if not extras:
                self.pred_log.append('[错误] checkpoint 中缺少 extras（旧格式），无法继续训练')
                return

            # ── 重建 meta_learner ──
            # weight_dim 必须匹配实际 dynamics 网络的权重层数
            weight_dim = len(predictor.dynamics.get_weight_stats())
            meta_state = extras.get('meta_learner_state', None)

            if meta_state is None:
                # 回退：尝试同目录下的 meta_learner_online.pt
                ckpt_dir = os.path.dirname(os.path.abspath(path))
                meta_path = os.path.join(ckpt_dir, 'meta_learner_online.pt')
                if os.path.exists(meta_path):
                    meta_state = torch.load(meta_path, map_location=device, weights_only=False)
                    self.pred_log.append(f'[信息] 从 {meta_path} 加载 meta_learner')
                else:
                    self.pred_log.append('[警告] 未找到 meta_learner 状态，将使用新的 meta_learner')

            meta_learner = MetaLearner(weight_dim=weight_dim)
            if meta_state is not None:
                meta_learner.load_state_dict(meta_state)
            meta_learner.to(device)

            # ── 回填模拟参数到设置面板 ──
            gt_params = extras.get('gt_params', {})
            if gt_params:
                self.settings.spin_T.setValue(gt_params.get('T', 100000))
                masses = gt_params.get('m', [1.0, 1.0, 1.0])
                self.settings.spin_mA.setValue(masses[0])
                self.settings.spin_mB.setValue(masses[1])
                self.settings.spin_mC.setValue(masses[2])
                r_arr = gt_params.get('r', [[0, 0, 0]] * 3)
                v_arr = gt_params.get('v', [[0, 0, 0]] * 3)
                self.settings._set_pos_vel_spins(r_arr, v_arr)

            # ── 回填超参数 ──
            if 'noise_level' in extras:
                self.pred_noise.setValue(extras['noise_level'])
            if 'feedback_threshold' in extras:
                self.pred_fb_thresh.setValue(extras['feedback_threshold'])
            if 'predict_horizon' in extras:
                self.pred_horizon_spin.setValue(int(extras['predict_horizon']))
            if 'segment_length' in extras:
                self.pred_seg_len_spin.setValue(int(extras['segment_length']))
            if 'n_predict_steps' in extras:
                self.pred_n_steps_spin.setValue(int(extras['n_predict_steps']))
            if 'init_frac' in extras:
                self.pred_init_frac.setValue(extras['init_frac'])

            # ── 重建 continue_state ──
            r_final = extras.get('r_final', None)
            v_final = extras.get('v_final', None)
            t_current = extras.get('t_trained', 0)

            if r_final is None or v_final is None:
                self.pred_log.append('[错误] checkpoint 中没有 r_final/v_final，无法继续训练')
                return

            self._final_state = {
                't_current': float(t_current),
                'r_current': np.asarray(r_final),
                'v_current': np.asarray(v_final),
                'predictor': predictor,
                'meta_learner': meta_learner,
                'feedback_log': extras.get('feedback_log', []),
                'model_snapshots': extras.get('model_snapshots', []),
                'error_history': extras.get('error_history', []),
                'error_times': extras.get('error_times', []),
                '_init_data': extras.get('_init_data', None),
                'n_cycles': extras.get('n_cycles', 1),
            }

            n_cycles = extras.get('n_cycles', 1)
            T = extras.get('gt_params', {}).get('T', 100000)
            self.pred_log.append(
                f'=== 已加载 checkpoint ===  {os.path.basename(path)}')
            self.pred_log.append(
                f'  已训练 {n_cycles} 周期  当前 t={t_current:.0f}  可继续 t=[{t_current:.0f} → {t_current + T:.0f}]')
            self.btn_continue.setEnabled(True)
            self.label_status.setText(f'已加载: t={t_current:.0f}, {n_cycles}周期 — 可继续训练')

        except Exception as e:
            import traceback
            self.pred_log.append(f'[错误] 加载失败: {e}\n{traceback.format_exc()}')

    def _restart(self):
        """终止当前运行并重置到初始状态。"""
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait(2000)
        self._worker = None
        self._pred_results = None
        self._final_state = None
        self.btn_start.setEnabled(True)
        self.btn_continue.setEnabled(False)
        self.progress_bar.setValue(0)
        self.label_status.setText('就绪')
        self.pred_log.clear()
        # 清空图表
        for tab in self.monitor_tabs:
            tab.line.set_data([], [])
            tab.line.set_3d_properties([])
            tab.canvas.draw_idle()
        for tab in self.pred_tabs:
            tab.line_obs.set_data([], [])
            tab.line_obs.set_3d_properties([])
            tab.line_pred.set_data([], [])
            tab.line_pred.set_3d_properties([])
            tab.canvas.draw_idle()
        for tab in self.compare_tabs:
            tab.ax.clear()
            tab.ax.set_facecolor('#fafafa')
            tab.ax.set_xlabel('X'); tab.ax.set_ylabel('Y'); tab.ax.set_zlabel('Z')
            tab.ax.set_title(f'天体 {tab.body_name}  观测 vs 真实',
                             fontsize=13, fontweight='bold')
            tab.canvas.draw_idle()
        self.overview_canvas.ax_gt.clear()
        self.overview_canvas.ax_pred.clear()
        for ax in [self.overview_canvas.ax_gt, self.overview_canvas.ax_pred]:
            ax.set_facecolor('#fafafa')
            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
            ax.text2D(0.5, 0.5, '等待闭环完成…', ha='center', va='center',
                      transform=ax.transAxes, fontsize=14, color='#ccc')
        self.overview_canvas.ax_gt.set_title('真实轨迹（ABC 同框）', fontsize=13, fontweight='bold')
        self.overview_canvas.ax_pred.set_title('拼接预测（ABC 同框）', fontsize=13, fontweight='bold')
        self.overview_canvas.fig.tight_layout()
        self.overview_canvas.canvas.draw_idle()

    def _on_progress(self, pct, status):
        self.progress_bar.setValue(pct)
        self.label_status.setText(status)

    def _on_monitor(self, t_hist, r_hist, v_hist):
        for tab in self.monitor_tabs:
            tab.update_data(t_hist, r_hist, v_hist)

    def _on_prediction(self, t_pred, r_pred, v_pred):
        # 观测数据只截取到当前段起始 t（预测起点之前），
        # 避免大 segment_length 时预测线完全被观测线覆盖
        t_obs_all = np.array(self._worker._t_obs)
        r_obs_all = np.array(self._worker._r_obs)
        v_obs_all = np.array(self._worker._v_obs)
        cutoff = self._worker._t_seg_start
        mask = t_obs_all <= cutoff + 1e-10
        t_obs = t_obs_all[mask].tolist()
        r_obs = r_obs_all[mask].tolist()
        v_obs = v_obs_all[mask].tolist()
        for tab in self.pred_tabs:
            tab.update_data(t_obs, r_obs, v_obs, t_pred, r_pred, v_pred)

    def _on_overview_update(self, t_obs, r_obs, v_obs, t_stitch, r_stitch, v_stitch):
        self.overview_canvas.update_overview_realtime(
            t_obs, r_obs, v_obs, t_stitch, r_stitch, v_stitch)

    def _on_log(self, msg):
        self.pred_log.append(msg)

    def _on_finished(self, results):
        self._pred_results = results
        self.label_status.setText('绘制结果中…')

        os.makedirs('checkpoints', exist_ok=True)
        final_state = results.get('final_state', {})
        extras = {
            'gt_params': self.settings.get_params(),
            'noise_level': self.pred_noise.value(),
            'feedback_threshold': self.pred_fb_thresh.value(),
            'predict_horizon': self.pred_horizon_spin.value(),
            'segment_length': self.pred_seg_len_spin.value(),
            'n_predict_steps': self.pred_n_steps_spin.value(),
            'init_frac': self.pred_init_frac.value(),
            'n_cycles': results.get('n_cycles', 1),
            't_trained': final_state.get('t_current', results.get('gt', {}).get('t', [0])[-1] if results.get('gt') is not None else 0),
            'r_final': final_state.get('r_current', None),
            'v_final': final_state.get('v_current', None),
            # 供「加载继续」使用的完整历史状态
            'meta_learner_state': results['meta_learner'].state_dict(),
            'feedback_log': results.get('feedback_log', []),
            'error_history': results.get('error_history', []),
            'error_times': results.get('error_times', []),
            'model_snapshots': results.get('snapshots', []),
            '_init_data': final_state.get('_init_data', None),
        }
        results['predictor'].save('checkpoints/predictor_online.pt', extras)
        torch.save(results['meta_learner'].state_dict(),
                   'checkpoints/meta_learner_online.pt')

        # 轨迹对比 ABC
        for tab in self.compare_tabs:
            tab.plot_results(results)

        # 全星图
        self.overview_canvas.plot_results(results)

        # 模型统计
        self.model_stats_canvas.plot_results(results)

        n_fb = len(results['feedback_log'])
        n_seg = len(results['error_history'])
        n_cycles = results.get('n_cycles', 1)
        self.pred_log.append(f'=== 闭环完成 ===  周期数={n_cycles}  段数={n_seg}  反馈={n_fb}')
        if results['error_history']:
            self.pred_log.append(
                f'  误差范围: {min(results["error_history"]):.3e} ~ {max(results["error_history"]):.3e}')
        self.label_status.setText(f'完成！段数={n_seg}  反馈={n_fb}')
        self.btn_start.setEnabled(True)
        self.btn_continue.setEnabled(True)
        self._final_state = results.get('final_state', None)

    def _on_error(self, msg):
        self.pred_log.append(f'[错误] {msg}')
        self.label_status.setText('出错')
        self.btn_start.setEnabled(True)

    def closeEvent(self, event):
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait(2000)
        event.accept()


# ==================== 入口 ====================

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindowV2()
    window.show()
    sys.exit(app.exec())
