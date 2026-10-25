# VelStand 策略——当前策略是如何训练的

这是一个集行走、零命令静止、保护性跌倒和自主起身于一体的策略。
头部可控；从 v7 开始，站立时身体姿态也可控；从 v8 开始，起身平稳可靠，
偏航命令也能得到实际跟踪。任务为
`Mjlab-VelStand-Rough-Backlash-MicroDuck`，cfg
`src/mjlab_microduck/tasks/microduck_velstand_env_cfg.py`（模块文档字符串和每个
`ENABLE_*` 开关的注释记录了设计与课程演进），BC 机制位于
`src/mjlab_microduck/tasks/distill.py`。

**当前策略：wandb `pollen-robotics/mjlab_microduck/0045i2si` @ `model_2749.pt`** —
Hub 仓库 `pollen-robotics/microduck-policies` 中的 `velstand.onnx`，标签为 **v8**
（2026-10-12），已在真实机器人上验证起身、转向和行走。上一版本 v7 为
`j4i6yoq2` @ 1250。

## 训练沿革

直到 v7，每个阶段都采用**热启动**：设置 `MICRODUCK_WARM_START=1`，并通过
`--agent.resume True` 加载另一次运行的 checkpoint。权重、观测标准化器和优化器会保留，
但 `common_step_counter` 与迭代数从 0 重新开始，因此 VelStand 自身的课程
（关闭主动倒地、俯卧/蹲伏初始状态、恢复收益、保护性代价、推倒扰动）会从第 0 阶段重跑。
继承自 velocity 配方的课程则折叠到最终阶段（`WARM_START = True`）。
v8 阶段使用**普通恢复训练**（无 `MICRODUCK_WARM_START`）：步数计数器继续，因此
每门课程都停留在最后阶段——策略是经过微调的，而不是重新教授的。

|阶段|run @ checkpoint |代码|新增内容 |
|---|---|---|---|
|行走教师策略| `441tzs6d` @ 3750 |velocity 配方| `alpha_walking.onnx` — 学生策略的初始化来源，也是行走 BC 锚点 |
|起身教师策略| `69u48n8l` @ 9750 |standup 环境 | `alpha_stand.onnx` — 起身 BC 专家；从 v7 起也作为身体倾斜教师策略 |
| velstand v1 | `6op8a8u8` @ 5999 | `d3edc1f`（标签 `velstand-best-6op8a8u8`）|在平地上通过专家 BC + 行走锚点学习保护性跌倒和起身 |
| v5/v6 生产版 | `fhathosb` @ 3750 | `e40fbda`，分支 `protective_fall` |将同一配方用于 Rough + Backlash，并修复崎岖地形初始状态问题 |
| v7 生产版 | `j4i6yoq2` @ 1250 | `53fb7d1`，分支 `improve_velstand2` |站立时的身体控制（见下文）|
| v8 起身 | `qz5g3bwi` @ 1750 | `333ef7b`，分支 `improve_standup_velstand` |关节观测屏蔽教师策略 + 更广的起身初始状态（见下文）；已在实机测试，未发布 |
| **v8 生产版** | **`0045i2si` @ 2749** | **`3750776`，分支 `improve_standup_velstand`** |向行走锚点输入经校准的偏航重映射，并加入精细偏航奖励（见下文）|

## 阶段 v7 — `j4i6yoq2`（站立时的身体控制）

| | |
|---|---|
|训练主机| COACH2、RTX 5090 |
|开始 | `fhathosb` @ `model_3750.pt`，热启动|
|使用的检查点 | **`model_1250.pt`**（run 在第 1499 次迭代结束；见 checkpoint 选择）|

```bash
MICRODUCK_WARM_START=1 uv run train Mjlab-VelStand-Rough-Backlash-MicroDuck --env.scene.num-envs 4096 \
  --agent.resume True --wandb-run-path pollen-robotics/mjlab_microduck/fhathosb \
  --wandb-checkpoint-name model_3750.pt --agent.max_iterations 1500
```

### 为什么

v5/v6 VelStand 完全忽略身体命令（用 `claude_experiments/velstand_body_probe.py`
测得响应为 0 mm / 0°）：velocity 配方把 `body_pose` 奖励权重设为 0，只保留
±5 mm / ±0.05 rad 的微小“激活”范围；行走锚点又把站立行为约束到从未学过身体控制的
`alpha_walking`。旧流程中的 `alpha_stand` 能跟踪横滚/俯仰
（命令 10° 时达到 9.2–9.5°），但不能跟踪 z。

### 修改内容（均由 `ENABLE_BODY_CONTROL` 控制）

|部分 |作用|
|---|---|
| `StandingGatedPoseCommand`（mdp.py）|身体命令 = 采样值 × `twist.is_standing_env`；行走/转向时严格为 0，与运行时一致。30% 使用全零命令桶，继续训练普通站立；半数重采样将 z 置零，以覆盖仅倾斜命令 |
|范围 |第 300 次迭代时，横滚/俯仰由 ±6° 扩至 **±12°**（约等于 `alpha_stand` 的训练范围）；z 为 −2.5 / +1 cm；x/y/yaw 保持很小，位于 `alpha_stand` 标准化器的有效范围内 |
|蒸馏身体路由（`body_slice`）|带有效身体命令的站立帧不再使用行走锚点；**仅倾斜 → `alpha_stand` BC**，含 z 的帧 → 仅 PPO（`alpha_stand` 不会蹲伏）。命令包含在观测中，因此两类帧可区分，不会泄漏教师信息 |
| `body_pose_tracking` |站立门控的 z/横滚/俯仰高斯奖励（z std 为 1 cm，角度 std 为 5°，权重 2，标称 z=0.116，即实测站立躯干高度）|
| `upright_body_cmd_relative` |相对于命令倾角计算内置 `upright` 奖励；零命令时与原定义相同 |
| `variable_posture_body_relaxed` |身体命令生效时放宽腿部姿态 std（如 hip_roll 从 0.05 增至 0.25）；原本严格的站立 std 会让 10° 横滚每步付出 −0.85 的代价 |

训练前的逐步奖励检查（同一站立环境）表明：跟踪 10° 倾斜比忽略命令分别多得
+1.35（俯仰）、+1.57（横滚）和 +2.1（两者同时）；零命令下奖励不变。

### 所选 checkpoint 的仿真评估（平地 + 齿隙，对比 v6 `fhathosb@3750`）

| | v7 `j4i6yoq2@1250` | v6 `fhathosb@3750` |
|---|---|---|
|身体横滚 ±10° / 俯仰 ±10° | +8.4 / −8.2° · +9.8 / −7.0° | 0（忽略）|
|身体 z −2 cm | ≈ 0（未学过）| 0 |
|原地转动 +1.0 / −1.0 rad/s | 0.45 / -0.37 | 0.12 / −0.19 |
|直行偏航漂移 (0.3 / 0.4 m/s) | 0.05 / 0.03 弧度/秒 | −0.03 / −0.07 |
|0.3 m/s 行走时推倒（向前 0.35 / 向后 / 转向）| 0 / 9 / 4 % | 1 / 12 / 9 % |
|推倒后起身 (1.0 / 1.2 m/s) | 93 / 95 % | 91 / 85 % |

工具：`velstand_body_probe.py`、`velstand_walk_stability.py`、`velstand_improve_bench.py`
（`--only walk|rise`），`velstand_ckpt_eval.sh <run> <ckpt> <ref.onnx>`（全部在`claude_experiments/`，
本地，gitignored）。

### 检查点选择

虽然训练指标看不出差异，各 checkpoint 的偏航行为却相差很大：要求以 0.4 m/s
直行时，`500` 向右偏转（−0.20 rad/s），`1499` 向左偏转（+0.27 rad/s），
`1250` 则没有明显漂移。选择 checkpoint 前必须运行行走扫描。

## 阶段 v8a — `qz5g3bwi`（平稳、可靠地起身）

| | |
|---|---|
|训练主机| COACH2 |
|开始 | `j4i6yoq2` @ `model_1250.pt`，**普通恢复训练**（课程保持最终阶段）|
|使用的检查点 | **`model_1750.pt`**（run 在第 2749 次迭代结束）— 2026-10-08 已通过实机测试，是 v8b 的基础 |

```bash
uv run train Mjlab-VelStand-Rough-Backlash-MicroDuck --env.scene.num-envs 4096 --agent.resume True \
  --wandb-run-path pollen-robotics/mjlab_microduck/j4i6yoq2 --wandb-checkpoint-name model_1250.pt \
  --agent.max_iterations 1500
```

### 为什么

在带随机关节的仰卧初始状态（`rj_fu`，见下文测试组）中，v7 约有 26% 会卡在仰卧，
而且仰卧起身很剧烈（|ω| p50 ≈ 11 rad/s）。根因是 BC 复制了**起身教师策略**
`alpha_stand` 中的一个吸引子：在某些腿部配置下（hip_roll ≈ −0.8 / −0.7，
head_yaw ≈ −1.6），教师标签是“保持原位”，学生策略也照做。仅向教师策略屏蔽头部观测
（run `s4jdgkux`，commit `2ee6dc8`）反而让结果**更差**（`rj_fu` 从 73% 降至 55%），
说明吸引子来自教师策略对*腿部*配置的响应。在倾角超过 75° 时，将
`alpha_stand` 闭环输入中的**全部关节观测置零**，能让各类初始状态的恢复率达到
93–100%，同时使起身更加平缓。

### 修改内容

|部分 |作用|
|---|---|
| `ENABLE_JOINT_MASKED_TEACHER` |蒸馏配置 `bc_cfg` 使用 `expert_mask_cols = range(6, 48)`（关节位置、速度、上一动作）和 `expert_mask_tilt_deg = 75`：倾角超过 75° 时，起身教师策略看到的是 HOME 关节，只能根据朝向给出起身标签；低于 75°（起身中段）时则看到完整观测 |
| `ENABLE_WIDE_SPAWNS` |倒地初始状态切片中：50% 使用包含头部在内的随机关节，25% 使用均匀 SO(3) 朝向 + 随机关节，5% 从坐姿开始；这些正是吸引子出现的初始状态 |

这类修改必须用学生策略执行完整的起身测试组，并结合 `--handoff-s` 验证
（屏蔽关节观测后的教师策略接管学生卡住的环境），不能只看短暂本地训练。
`s4jdgkux` 的 400 次迭代本地检查表现良好，但完整训练最终失败。

## 阶段 v8b — `0045i2si`（偏航跟踪）

| | |
|---|---|
|训练主机| COACH2 |
|开始| `qz5g3bwi` @ `model_1750.pt`，**普通恢复训练** |
|使用的检查点 | **`model_2749.pt`**（最后一个）— 2026-10-12 已通过实机测试 |

```bash
uv run train Mjlab-VelStand-Rough-Backlash-MicroDuck --env.scene.num-envs 4096 --agent.resume True \
  --wandb-run-path pollen-robotics/mjlab_microduck/qz5g3bwi --wandb-checkpoint-name model_1750.pt \
  --agent.max_iterations 1000
```

### 为什么

实机转向表现很差：原地命令 0.3 / 0.6 rad/s 时响应约为 0，命令 1.0 时约为 0.4；
行走时只达到命令的一半。**行走锚点** `alpha_walking`（`441tzs6d` @ 3750）
恰好具有同样的响应曲线：原地输入 0.6 → 0、1.0 → 0.35，行走时约为输入的 0.6 倍；
系数为 1.0 的 BC 会原样复制这一缺陷。让转向帧脱离锚点
（`ENABLE_YAW_FIX`，runs 为 `sape62zb`、`brkpcnn8`）会彻底丧失转向能力，因此保持关闭。

### 修改内容（`ENABLE_ANCHOR_YAW_REMAP`）

|部分 |作用|
|---|---|
| `bc_cfg["anchor_yaw_remap"]`（蒸馏中的 `remap_anchor_yaw`）|向锚点输入能让**锚点自身**达到目标速率的偏航值：使用按符号划分的逆查找表 `ANCHOR_YAW_REMAP`，在 \|v_xy\|（0–0.4）和目标 \|wz\|（0.1–1.0）上做双线性插值，0 映射到 0。只改变锚点输入；学生观测仍保留真实命令，站立教师的帧不受影响 |
|查找表校准| `claude_experiments/calibrate_walk_yaw.py`：在 GPU 上批量运行锚点的 .pt 模型，覆盖 \|vx\| 0–0.4 × 输入偏航 ±0.2–2.4 rad/s，再对实测响应构造单调逆映射。用 `validate_anchor_yaw_remap.py` 闭环验证重映射：原地目标 0.5 / 0.7 / 1.0 → 实际 0.56 / 0.68 / 1.07，跌倒数为 0 |
| `track_yaw_fine` |仅针对偏航的高斯奖励（`track_yaw_rate_fine`，std 0.35，权重 2）；内置项的 std 0.71 在小误差区间梯度过小 |
|原地转向采样桶| 20% 的 twist 重采样为原地转向，且 \|wz\| ≥ 0.2 |

## v8 仿真评估（平地 + 齿隙，0.2 表示 vx=0.2 m/s）

偏航（`velstand_yaw_probe.py --backlash`，达到 rad/s，2 秒后直立环境的平均值）：

|命令| v7 `j4i6@1250` | v8a `qz5g@1750` | **v8 `0045i2si@2749`** |
|---|---|---|---|
|原地 ±0.3 | 0.00 / 0.00 | 0.00 / 0.00 | **+0.08 / −0.22** |
|原地 ±0.6 | +0.08 / −0.03 | +0.05 / −0.04 | **+0.52 / −0.57** |
|原地 ±1.0 | +0.47 / −0.35 | +0.34 / −0.42 | **+0.97 / −0.87** |
| 0.2, ±0.3 | +0.18 / −0.07 | +0.22 / −0.04 | **+0.25 / −0.32** |
| 0.2, ±0.6 | +0.32 / −0.20 | +0.35 / −0.19 | **+0.54 / −0.56** |
| 0.2, ±1.0 | +0.55 / −0.37 | +0.50 / −0.40 | **+0.92 / −0.88** |

起身（`velstand_rise_battery.py --backlash`，每类初始状态 2 × 128 个环境，12 秒；
`rj` = 随机关节，`so3` = 均匀朝向，`limp` = 无力跌倒后的落地状态）：

|初始状态族|v7 恢复率 % / 卡住率 %| **v8 恢复率 % / 卡住率 %** |v7 → v8 仰卧 \|ω\| p50|
|---|---|---|---|
|HOME 俯卧 / 仰卧 / 侧卧| 100 / 100 / 100 | **100 / 100 / 99** | 11.1 → **6.8** rad/s（仰卧）|
| rj 面朝下 | 96 / 3.5 | **100 / 0.4** | |
| rj 面朝上 | 73 / **26.6** | **94 / 5.9** | 10.1 → **7.8** |
| rj 侧卧 | 91 / 9.4 | **96 / 3.5** | |
| so3 HOME / rj / limp | 100 / 86 / 91 | **100 / 96 / 98** | |
|坐姿| 89 / 11.3 | **99 / 0.8** | |

两者越过竖直方向的角度中位数均为 4–5°；v8 起身更慢
（仰卧 t_rec p50 从 1.1 增至 1.5 s）也更平缓，与实机观察一致。

行走/推动测试（`velstand_bench.py --backlash --only walking,push`）：vx 跟踪与 v7 和
`alpha_walking` 大致相同（命令 0.4 m/s → 实际 0.17，与教师策略相同）；以
0.2 / 0.3 / 0.4 m/s 直行时，偏航漂移分别为 −0.02 / +0.03 / +0.08 rad/s
（该 run 的早期 checkpoint 向左漂移 +0.10–0.13，到第 2749 次迭代才稳定）。
以 0.15 m/s 行走时施加 0.8 / 1.0 / 1.2 m/s 推动，跌倒率为 53 / 90 / 98%，
**恢复率为 100 / 100 / 99%**（v7 为 93–95%），t_rec p90 为 3.2 s。

### 检查点选择

`0045i2si` 各 checkpoint 的主要差异是偏航对称性和直行漂移：@2000 的右转响应
约为左转的 80%，直行时还会向左漂移 +0.10–0.13 rad/s；@2749 已接近对称，
漂移 ≤ 0.08。各 checkpoint 的起身表现同样优秀。与之前一样，选择 checkpoint 前
必须运行偏航探针和行走基准测试。

## 已知限制

- **身体 z（蹲伏）** 尚未学会（−2 cm 命令下响应 ≤ 1 mm）：该轴没有教师信号，
  1500 次 PPO 迭代也未能探索出对应行为。
- **慢速原地偏航**（在 v8 中固定为 0.5 rad/s 以上）。以 0.3 rad/s 的速度转动到位
  仍然较弱且呈突发式（+0.08 / −0.22）；重映射后的锚点本身在该区域存在双峰启动行为。
  手柄的默认范围为 ±1.0 rad/s（microduck 中的 `[pad_drive]`），位于 v8 可跟踪范围内。
- **偏航历史记录 (v7)。** v7 的死区来自步行锚点（请参阅 v8b）。运行时重新映射
`wz_obs = sign(c)·(0.33 + |c|/0.6)` 是 v7 的仅 sim 解决方法，对于 v8 **不是**，
它直接接收真实命令。

## 复制/部署

```bash
uv run scripts/export.py Mjlab-VelStand-Rough-Backlash-MicroDuck \
  --wandb-run-path pollen-robotics/mjlab_microduck/0045i2si --checkpoint 2749
uv run python claude_experiments/velstand_yaw_probe.py output.onnx --backlash
uv run python claude_experiments/velstand_rise_battery.py output.onnx --backlash
```

始终通过 `scripts/export.py` 导出（观测标准化器已内嵌到 ONNX 中）。Hub 上的官方
策略集合需手工更新：替换 `velstand.onnx`，只编辑对应的 manifest 条目，提交后执行
`create_tag vN`。`uv run publish` 用于单策略仓库。
