# 三体进化闭环预测系统

以《三体》世界观为背景的深度学习模拟与可视化项目。从"三体人"的视角出发，构建一套**闭环进化预测系统**——神经网络预测器从噪声观测中隐式学习三体动力学，在预测偏差超过阈值时自我修正，并与元学习器协同进化。

## 项目架构

```
three_body.py              ← 物理引擎层（模拟宇宙本体，高精度数值积分）
       ↓
closed_loop_evolution.py   ← 核心算法层（神经网络 + 元学习 + 闭环控制）
       ↓
ui_panels_v2.py            ← UI 层（PySide6 + Matplotlib 实时可视化）
evaluate_ui.py             ← 独立评估（加载模型纯推理，评估预测精度）
```

### 数据流

1. `three_body.py` — 使用 **Yoshida 4 阶辛积分器** + 自适应步长进行高精度数值积分，生成"真实"三体运动轨迹。
2. `closed_loop_evolution.py` — 读取真实轨迹叠加**观测噪声**，驱动神经网络预测器进行预测 → 比较 → 反馈修正 → 元学习器训练的闭环。
3. UI 层通过 `QThread` 在后台运行闭环进化，实时推送数据到前端图表。

## 文件说明

| 文件                         | 说明                                                                  |
| -------------------------- | ------------------------------------------------------------------- |
| `three_body.py`            | 物理引擎：支持等边三角形、8字形、自由参数三种初始条件，Yoshida 辛积分 + CUDA 加速                   |
| `closed_loop_evolution.py` | 核心算法：DynamicsNetwork 预测器、MetaLearner 元学习器、ClosedLoopEvolution 闭环控制器 |
| `ui_panels_v2.py`          | 主 GUI：12 个标签页（3D 轨迹、预测对比、全星图、模型统计）+ 参数设置 + 日志                       |
| `evaluate_predictor.py`    | 命令行版模型评估：加载 checkpoint 进行纯推理，输出误差衰减、预测视界等统计                         |
| `evaluate_ui.py`           | GUI 版模型评估：左右两张 3D 图实时展示 GT 轨迹和累积预测拼接轨迹                              |
| `ui_panels.py.diss`        | v1 旧版 UI（已被 v2 替代，保留供参考）                                            |
| `checkpoints/`             | 训练好的模型文件（predictor\_online.pt、meta\_learner\_online.pt）             |

## 环境依赖

- Python 3.8+
- [PyTorch](https://pytorch.org/) — 神经网络 + 自动微分 + CUDA 加速
- [NumPy](https://numpy.org/) — 数值计算
- [PySide6](https://pypi.org/project/PySide6/) — GUI 框架
- [Matplotlib](https://matplotlib.org/) — 数据可视化

```bash
pip install torch numpy PySide6 matplotlib
```

> 如需 GPU 加速，请安装 CUDA 版 PyTorch，参见 [PyTorch 官网](https://pytorch.org/get-started/locally/)。

## 运行方式

### 1. 命令行快速演示（不推荐，可能存在未知bug）

```bash
# 快速演示（T=10000）
python closed_loop_evolution.py --mode quick

# 完整演示（T=50000）
python closed_loop_evolution.py --mode full
```

运行完成后自动保存模型到 `checkpoints/`，并输出可视化图片 `evolution_online.png`。

### 2. 独立运行物理模拟

```bash
python three_body.py
```

交互式选择初值模式（等边三角形 / 8 字形 / 自由参数 + C 自动求解），运行模拟并显示 3D 星图。

### 3. GUI 主界面（推荐）

```bash
python ui_panels_v2.py
```

- 设置参数后点击"开始闭环进化"，实时观察预测与反馈过程
- 支持"继续下一段 T"：基于当前模型继续训练下一个宇宙周期
- 支持"加载继续"：从磁盘加载 checkpoint 恢复训练
- 支持"重启"：清空状态重新开始

### 4. 模型评估

```bash
# 命令行版
python evaluate_predictor.py checkpoints/predictor_online.pt

# GUI 版
python evaluate_ui.py checkpoints/predictor_online.pt
```

加载已训练模型进行**纯推理评估**（不修改模型），输出预测误差、可靠预测视界等统计信息。

## 核心设计

### 闭环进化

预测器与模拟宇宙同时在线运行，持续循环：

1. **观测**：从真实轨迹获取带噪声的测量数据
2. **预训练**：在初始观测窗口内拟合 DynamicsNetwork
3. **在线共进化**：预测未来 → 比较偏差 → 超出阈值则沿梯度反馈修正
4. **元学习**：MetaLearner 从历史修正记录中学习最优修正策略
5. **多周期**：通过 `continue_state` 机制可无限循环，模拟跨天长周期的学习进化

### 三种辅助设置初始条件

- **等边三角形**：三天体绕质心做稳定圆周运动（特解）
- **8 字形**：Chenciner-Montgomery 周期轨道（经典特解）
- **自由参数 + C 自动求解**：用户指定两天体初值，系统自动求解第三天体使质心动量归零
- **自定义初始条件**：利用特解修改初始条件，实现自定义场景，防止开局就发生碰撞或飞掠

### 神经网络预测器

- 前馈网络（3 层 SiLU + Dropout），输入 18 维（3 天体 × 6 分量），输出 9 维加速度
- 含残差连接和输入标准化
- 将网络输出的加速度用于 **RK4 数值积分**，递归外推未来轨迹

### 元学习器

- 输入：权重统计 + 误差特征 + 上下文信息
- 输出：建议学习率乘数 + 置信度
- 训练目标：从历史反馈中学习"何时调大学习率、何时保守"

### 高精度物理引擎

- **Yoshida 4 阶辛积分器**：保能量，O(dt⁵) 精度
- **自适应步长**：基于天体间最小穿越时间动态调整
- **PyTorch GPU 加速**：可处理近距飞掠等高精度场景
- **观测噪声**：支持相对噪声和绝对噪声两种模式

