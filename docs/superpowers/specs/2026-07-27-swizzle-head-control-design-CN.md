# 葫芦步头控制（Y 按钮）— 设计

**日期：** 2026-07-27
**分支：** `new_pre_alpha_rollers`
**涉及任务：** `Mjlab-Velocity-Swizzle-MicroDuck` (`microduck_velocity_swizzle_env_cfg.py`)

## 目标

允许操作员在机器人以滚轮做葫芦步时通过 Y 按钮控制不同的头部姿态，
同时保证**头部移动不会破坏葫芦步**。这里的头部姿态指物理头部/颈部关节
（向上、向下、向左、向右），与 `heading_tracking` 无关；后者表示身体行进方向，
本设计不会修改它。

若把头部动作作为外部偏移直接叠加，当前葫芦步策略会因无法补偿质心偏移而倾倒。
因此头部必须由**策略管理**：策略既生成头部姿态，也负责保持平衡。
这与行走策略的 `--new-cmd-obs` 模式一致：头部目标作为命令注入观测，
再由策略生成对应姿态，不在外部重复叠加偏移。

## 方法（采用方案 A：通过观测命令管理头部）

把 `microduck_velocity_env_cfg.py` 中已有的头部命令机制移植到葫芦步环境：
将真实头部姿态命令写入当前补零的 `head_command` 观测槽，奖励头部跟踪该命令，
并通过课程在训练后期逐步引入，以免干扰葫芦步的学习。

不提供外部偏移方案。该方案此前已被否决，因为头部在没有策略补偿的情况下移动时，
机器人无法在葫芦步中保持直立。

## 修改内容（均位于 `make_microduck_velocity_swizzle_env_cfg`）

1. **头部姿态命令项。** 添加 `cfg.commands["head_pose"] = UniformPoseCommandCfg(...)`，
   沿用 velocity 环境的设置：4D `[neck_pitch, head_pitch, head_yaw, head_roll]`
   相对默认值的增量，`resampling_time_range = (2.0, 5.0)`，并分别设置关节范围；
   head_roll 的范围更小，以匹配其较小的机械行程。
2. **真实的 `head_command` 观测。** 对 Actor 和 Critic，都把头部槽从当前的
   `zero_command_padding(dim=4)` 改为真实命令观测
   `func=<head command obs>, params={"command_name": "head_pose"}`。61D 布局保持不变；`body_command` 仍补零，
   因为此任务不控制身体姿态。
3. **`head_pose_tracking` 奖励。** 添加 `cfg.rewards["head_pose_tracking"]`
   （`microduck_mdp.head_pose_tracking`、`command_name="head_pose"`、`std=0.5`），
   初始权重为 0，之后由课程分阶段提升。
4. **后期课程。** 使用 `reward_weight` 课程将 `head_pose_tracking` 从 0 提升到
   **4.0**：在葫芦步稳定前，即约 **1500 次迭代**内保持为 0，再在之后约
   1000 次迭代中逐步提高，参考 VelStand 引入身体姿态控制的方式。同时加入头部姿态
   命令范围课程：从较窄范围（较小的头部增量）开始，在同一时间窗口内逐步扩大；
   这样前期头部几乎不动，策略具备处理能力后才达到完整范围。头部控制是在稳定的
   葫芦步基础上引入的，因此更易训练。上述数值是可调的初始值。
5. **协调颈部惩罚（必需）。** 当前环境中的 `neck_joint_pos_l2` 会把颈部/头部关节
   拉回 HOME，与把关节拉向命令的 `head_pose_tracking` 冲突，导致头部无法移动。
   应从 `neck_joint_pos_l2` 中排除头部姿态关节，或直接移除该项；处理方式与 velocity
   环境一致。保留不会造成目标冲突的平滑项 `neck_action_rate_l2`。

其他内容（葫芦步、后退运动、航向课程、DR、观测布局、
命令）不变。需要**重新训练** swizzle 任务。

## 运行时

没有运行时代码更改。 `microduck_runtime` **Y 按钮** 已驱动
`head_command` 观测槽；new-cmd-obs 模式会把头部偏移作为命令注入，不会重复叠加。
用头部控制重新训练葫芦步策略后，它就会响应 Y 按钮。
部署标志不变 (`--roller --new-cmd-obs ...`)。

## 测试/验证

- 冒烟测试：`uv run train Mjlab-Velocity-Swizzle-MicroDuck --env.scene.num-envs 16
  --agent.max-iterations 2` 能运行；奖励日志中出现 `head_pose_tracking`；
  `head_pose` 命令和真实的 `head_command` 观测均能正确构建。
- 完整训练：课程启用后 `head_pose_tracking` 上升；葫芦步保持稳定，头部课程开启时
  跌倒率不应突增。在查看器或实机中，改变头部命令应使头部移动，同时机器人继续滑行。

## 调参项

- 头部控制启用后干扰葫芦步 → 推迟课程启用时间，或更慢地扩大头部范围。
- 头部跟踪不佳 → 提高 `head_pose_tracking` 的目标权重，或检查是否仍有颈部惩罚与其冲突。
- 头部动作过于抖动 → 保持或提高 `neck_action_rate_l2`。
