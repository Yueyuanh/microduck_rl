# 葫芦步（Swizzle）滚轮环境——设计

**日期：** 2026-07-23
**分支：** `new_pre_alpha_rollers`

## 目标

创建一个**独立的**滚轮任务，生成**干净、经典的葫芦步（swizzle）**：两侧轮组
始终接地，双腿**对称地**向外展开再向内收拢，形成沙漏形轨迹并推动机器人前进。
它是交替蹬滑任务（`Mjlab-Velocity-Flat-MicroDuck-Rollers`）更简单、稳定的替代方案；
之所以新增该任务，是因为交替蹬滑未能很好地迁移到真实机器人。原蹬滑环境保持不变。

Sim2real 是核心目标：该任务与交替蹬滑环境使用相同的机器人、观测、命令语义、
域随机化和 ONNX 导出流程，因此**部署方式完全相同**
（`microduck_runtime ... --roller`，使用相同参数）。

## 方法（选择 A：移除反葫芦步机制，并奖励对称性）

基础滚轮速度配方会*自然地*收敛到葫芦步，这正是训练交替蹬滑时一直对抗的吸引子。
因此，获得干净葫芦步的最直接方法是**移除抑制葫芦步的机制**，并**奖励葫芦步的
定义性特征**（对称、双脚接地），不使用相位脚本。

未采用方案 B（显式塑造沙漏形足部轨迹）和方案 C（相位驱动的脚本轨迹）：
二者更复杂，只有方案 A 的葫芦步在节奏或幅度上较杂乱时才有必要。

## 结构

- 新文件 `src/mjlab_microduck/tasks/microduck_velocity_swizzle_env_cfg.py` 与
`make_microduck_velocity_swizzle_env_cfg(play=False)` 和 `MicroduckSwizzleRlCfg`。
  由 `make_velocity_env_cfg()` 和滚轮机器人配置构建，并沿用
  `microduck_velocity_rollers_env_cfg.py` 的结构（观测、DR、命令、课程）。
- 将`Mjlab-Velocity-Swizzle-MicroDuck`注册到`tasks/__init__.py`中。
- 复用交替蹬滑环境的全部 sim2real 设置：机器人 cfg、61D 观测布局、命令
  （cmd_x 推进/滑行/制动；直线约束：`ang_vel_z=(0,0)`、`heading_hold`）、全部 DR
  事件和课程（com、wheel_friction）、`action_over_limit` 以及 ONNX 导出路径。

## 奖励配方

**保留**（任务+稳定性+sim2real）：
`wheel_speed`（正向推进、任务）、`braking`、`upright`、`com_height_target`、
`pose`、`forward_lean`、`heading_hold`、`action_over_limit`、`feet_flat`、
`self_collisions`，正则化器（`action_rate_l2` + 课程，`neck_action_rate_l2`，
`neck_joint_pos_l2`、`joint_torques_l2`）。

**移除**（交替蹬滑/抑制葫芦步机制）：
`single_support`、`glide`、`skating_air_time`、`gait_symmetry`、`hip_roll_neutral`
（最后一项会阻碍葫芦步所需的腿部横向外展）。

**新增**（鼓励葫芦步）：
- `leg_symmetry`：奖励左右腿镜像运动。机器人左右侧使用镜像符号约定，因此对称配置中
  每对关节满足 `q_left + q_right ≈ 0`。对腿部关节对（hip_yaw、hip_roll、hip_pitch、
  knee、ankle）返回 `-mean_pairs |q_left + q_right|`，即具有恒定梯度的 L1 形式，
  与现有 `bilateral_symmetry_penalty` 一致。配合正权重后，不对称会受到惩罚，
  对称葫芦步会得到更高奖励。这是葫芦步的核心特征。
  实现上，由一个轻量包装器在运行时按名称解析左右腿关节对，再调用接受显式左右索引列表的
  `bilateral_symmetry_penalty`，从而避免硬编码索引。
- `grounded`：推进时奖励两侧轮组同时接地（n_contact == 2），使双脚保持接地，
  符合经典葫芦步而不抬脚。该项使用较小权重。新增 MDP 函数与
  `single_support_reward` 对应，但奖励双支撑；与其他项一样由 `cmd_x >= 0` 门控。

保持宽松的 `hip_roll` 姿态约束（与交替蹬滑环境相同），让双腿能够向外展开。

## 新的 mdp 函数（在 `tasks/mdp.py` 中）

1. `leg_symmetry_reward(env, asset_cfg)` — 按名称解析 L/R 腿关节对，
   返回 `-mean_pairs |q_left + q_right|`（配合正权重使用）。
2. `grounded_reward(env, sensor_name, command_name)` — 奖励恰好两侧轮组接地，
   并按 `clamp(cmd_x, 0)` 缩放。

## 命令与 sim2real（与交替蹬滑环境相同）

`cmd_x` 用于推进/滑行/制动，`lin_vel_y=0`，`ang_vel_z=(0,0)`（直线运动）；
使用完整 DR（com、head_com、质量/惯量、关节摩擦、转子惯量、轮子摩擦、速度扰动、
IMU 未对准、编码器偏置、观测延迟）、61D 观测和 `vel_scale=0.3`。
部署时使用与交替蹬滑策略相同的运行时参数。

## PPO配置

复用 `MicroduckRollersRlCfg` 的超参数（相同的 Actor/Critic 512-256-128 ELU，
PPO 设置，`entropy_coef=0.03`)，新 `experiment_name`/`run_name` = `velocity_swizzle`。

## 测试/验证

- 冒烟测试：`uv run train Mjlab-Velocity-Swizzle-MicroDuck --env.scene.num-envs 16
  --agent.max-iterations 2` 能无错误运行；奖励日志中出现 `leg_symmetry` 和 `grounded`。
- 完整训练时观察：`leg_symmetry` 较高（动作对称）、`grounded` 较高（双脚接地）、
  `wheel_speed` 上升（向前移动）。视频应显示对称的沙漏形葫芦步，且两侧轮组均保持接地。

## 调谐旋钮（首次运行后）

- 如果不够对称→提高`leg_symmetry`权重。
- 如果抬起脚 → 提高 `grounded` 权重。
- 如果它几乎不动 → 对称性/接地权重相对于 `wheel_speed` 太高；
降低它们。
- 如果葫芦步看起来很混乱（节奏/幅度）→ 升级到方法 B。
