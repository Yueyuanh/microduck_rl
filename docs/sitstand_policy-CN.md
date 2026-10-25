# 坐站策略——当前策略是如何训练的

用一个策略按命令完成坐 ↔ 站切换（twist vx 槽中的姿态标志：0 = 站，1 = 坐），
两种姿态下均可控制头部。任务为 `Mjlab-SitStand-Flat-MicroDuck`，cfg 位于
`src/mjlab_microduck/tasks/microduck_sitstand_env_cfg.py`（模块文档字符串包含
奖励设计及其经验演进）。

**当前策略：wandb `pollen-robotics/mjlab_microduck/vfhxgrum` @ `model_2250.pt`**
（2026-10-01 已在真实机器人上验证）。训练分为两个阶段：先完成基础训练，
再在同一任务上**恢复训练**，强化坐姿向后倾倒后的恢复能力。

## 第 1 阶段 — 基本运行 `jngaedcs`（从头开始）

| | |
|---|---|
|代码| commit `c832147`（“更好地坐站”，2026-08-12）|
|训练主机| COACH、RTX 4090 |
|使用的检查点 | `model_1250.pt` — 之前部署的 `alpha_sitstand.onnx`（Hub 上的 `pollen-robotics/microduck-policies` 与 `microduck/policies/`，md5 均为 `ae90646…`，权重精确匹配）|

```bash
uv run train Mjlab-SitStand-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max-iterations 10000
```

能够轻柔地坐下和起身，并在坐姿下控制头部。**已知缺陷：** 坐姿时突然移动头部
会使机器人向后倾倒；之后机器人会保持约 30° 后倾，颈部舵机为托住头部而过载。

## 第 2 阶段 — 恢复训练 `vfhxgrum`（强化坐姿后倾恢复）

| | |
|---|---|
|代码|提交 `77ffadf` （“修复坐站倒退”），分支 `sitstand_no_fall_back` |
|训练主机| COACH2、RTX 5090 |
|开始 | `jngaedcs` @ `model_1250.pt`，已恢复（迭代计数器保持 1250）|
|使用的检查点 | **`model_2250.pt`**（运行在第 2761 次迭代终止；2250 在实机上表现良好）|

```bash
uv run train Mjlab-SitStand-Flat-MicroDuck --env.scene.num-envs 4096 \
  --wandb-run-path pollen-robotics/mjlab_microduck/jngaedcs --wandb-checkpoint-name model_1250.pt \
  --agent.resume True --agent.max_iterations 2250
```

### 为什么使用恢复训练（而不是 `MICRODUCK_WARM_START=1`）

这是**相同的任务**，因此恢复 `common_step_counter` 就是我们想要的：每个现有的
课程（头部范围、DR、外力推动、动作变化率、起身速度上限）都会从基础训练停止时的
第 1250 次迭代继续；新扰动按绝对步数计数器调度，因此会在恢复训练 250 次迭代后启用。
`--agent.max_iterations` 计算的是在已加载迭代数上**新增**的迭代
（从 2250 加到 3500）。`MICRODUCK_WARM_START=1` 则会让已经跨过课程阶段的策略
从第 0 阶段重新经历全部课程，它只适用于跨任务初始化。

### 发生了什么变化（全部在提交 `77ffadf` 中）

修改前通过仿真测得根因（`claude_experiments/sitstand_head_bench.py`、
`sitstand_tip_probe.py`）：训练中的头部命令每隔数秒才重采样一次，从未快速变化；
训练初始状态中也没有坐姿后倾状态，因此既缺少触发问题的扰动，也没有恢复数据。

|项 |类型|作用|调度（迭代）|
|---|---|---|---|
| `seated_head_jerk` (`seated_head_command_jerk`) |间隔事件，0.3–0.8 s |仅限就座环境：头命令符号翻转（快速摆动）或在活动范围内重新采样（跳跃）|概率 0 → **0.2 @1500** → **0.4 @2000** |
| `seated_tip` (`seated_backward_tip`) |间隔事件，3–6 秒，p=0.5 |仅限坐姿环境：施加向后俯仰冲量（−ω_y，混合 ±30% 横滚分量）| 0 → (1.0, 2.5) rad/s @1500 → (1.5, 3.25) @2000 → (2.0, 4.0) @2500 |
| `neck_strain_leaned` (`neck_torque_when_tilted`) |成本，权重 **−0.5** |mean (\|τ_neck\|/0.1)² × 坐姿混合系数 × 后倾系数 × 12°→25° 倾角渐增系数 |始终开启（直立/站立/起身/前倾时为零）|

“坐姿”表示坐姿标志已开启且姿态渐变已完成（α ≥ 0.98），因此两个事件都不会作用于
站立或过渡过程。所选的 `model_2250` 经历了概率 0.4 的头部突变阶段和
(1.5, 3.25) rad/s 的后倾扰动阶段；尚未进入 (2.0, 4.0) 阶段。

### 训练日志是什么样的

平均奖励在 1500 时为 140 → 125，在 2000 时为 131 → 116（每个阶段都让坐着变得更困难），
并在各阶段内逐渐恢复。`neck_strain_leaned` 在第 1500 次迭代后从 −0.11 改善到
第 1990 次迭代时的 −0.013（策略不再让颈部持续受力）。所有惩罚始终 ≤ 0，
没有发生 `nan_state` 重置。

### Sim 评估（`claude_experiments/sitstand_fallback_eval.py`，坐着，128 envs × 30 s，最终强度的扰动）

| | jngaedcs@1250（旧）| vfhxgrum@2000 |
|---|---|---|
|仅头部抽动 — 时间向后倾斜 > 20° | 21% | **3.1%** |
|仅头部突变 — 结束时仍后倾的环境比例 | 32% | **3.1%** |
| + 后倾冲量 — 后倾时间占比 | 47% | 21% |
|倾斜时颈部 \|τ\|| 0.20 N·m | **0.045 N·m** |

在第 2000 次迭代时，从强后倾冲量中恢复的成功率仍约为 0；但在第 2250 次迭代时，
实机上由头部移动引发的后倾问题已经消失。

## 复制/部署

```bash
uv run scripts/export.py Mjlab-SitStand-Flat-MicroDuck \
  --wandb-run-path pollen-robotics/mjlab_microduck/vfhxgrum --checkpoint 2250 --onnx-file sitstand.onnx
uv run python claude_experiments/sitstand_fallback_eval.py sitstand.onnx [--no-tip]   # before/after battery
```

始终通过 `scripts/export.py` 导出（观测标准化器已内嵌到 ONNX 中）。

## 待办事项

游戏手柄（`padd --max-head`，默认 2.5 rad）可发出远超训练范围的头部命令
（颈部/头部俯仰 ±1.10、偏航 ±1.40、横滚 ±0.31），其中大部分还超出关节限位。
应在 `robotd` 中把头部命令限制到训练范围；持续向机械限位发命令本身就是颈部过载源。
扩大训练范围（偏航约 ±2.5、向下俯仰约 −1.7）需要以后另行重新训练。
