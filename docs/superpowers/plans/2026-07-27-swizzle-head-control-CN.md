# 葫芦步（Swizzle）头部控制实施计划

> **面向自主代理：** 必须使用子技能 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans，逐项实施本计划。使用复选框（`- [ ]`）跟踪各步骤。

**目标：** 为滚轮葫芦步任务加入操作员头部姿态控制（Y 按钮），使策略在保持平衡的同时把头部移动到命令姿态。

**架构：** 通过观测中的命令让策略管理头部（与行走任务的 `--new-cmd-obs` 路径一致）。swizzle 环境当前把 `head_command` 观测槽补零；本计划改为写入真实的 `head_pose` 命令，加入 `head_pose_tracking` 奖励，移除两个会把颈部/头部拉回 HOME、从而与命令冲突的奖励项，并通过课程在训练后期才逐步启用头部控制，避免干扰已经稳定的 swizzle。只修改一个配置文件，但需要重新训练。

**技术栈：** mjlab / mjlab_microduck 任务配置（Python）、rsl_rl PPO。复用 `microduck_velocity_env_cfg.py` 中已有的机制（`UniformPoseCommandCfg`、`head_pose_tracking`、`pose_command_range_curriculum`、`reward_weight`）。

## 全局约束

- 仅 swizzle 任务发生变化：`src/mjlab_microduck/tasks/microduck_velocity_swizzle_env_cfg.py`。步幅、速度、站立、滚轮斜坡/蹲伏任务和 `mdp.py` 未修改。
- 保持 61D 观测布局 `[twist(3), head(4), body(6)]`：替换 `head_command` 槽的内容（补零 → 真实命令），但保持 `body_command` 补零（此处不控制身体姿态）。
- 不新增 MDP 函数；所有奖励、命令和课程函数都已存在于 `microduck_mdp` 中。
- 运行时不变：`microduck_runtime` Y 按钮已驱动 `head_command` obs 插槽。

---

### 任务 1：将头部姿态控制接入葫芦步环境

**文件：**
- 修改：`src/mjlab_microduck/tasks/microduck_velocity_swizzle_env_cfg.py`
- 测试：`tests/test_swizzle_head_cfg.py`（创建）

**接口：**
- 输入接口（已存在，不要重新定义）：
  - `microduck_mdp.UniformPoseCommandCfg(resampling_time_range, ranges)` — 头部姿态命令项。
  - `mdp.generated_commands`（来自 `mjlab.tasks.velocity`）— 按名称读取命令的观测函数；以 `params={"command_name": "head_pose"}` 使用。
  - `microduck_mdp.head_pose_tracking` — 奖励 `func`，参数 `{"command_name": "head_pose", "std": 0.5}`。
  - `microduck_mdp.reward_weight` — 课程函数，参数 `{"reward_name", "weight_stages": [{"step","weight"}, ...]}`。
  - `microduck_mdp.pose_command_range_curriculum` — 课程函数，参数 `{"command_name", "range_stages": [{"step","ranges"}, ...]}`。
- 输出：葫芦步环境 cfg 包含 `head_pose` 命令、真实的 `head_command` 观测和 `head_pose_tracking` 奖励；移除 `neck_joint_pos_l2`，将 `pose` 奖励限定到腿部关节，并加入两项头部课程。

- [ ] **第 1 步：编写失败的测试**

创建 `tests/test_swizzle_head_cfg.py`：

```python
from mjlab.tasks.velocity import mdp
from mjlab_microduck.tasks.microduck_velocity_swizzle_env_cfg import (
    make_microduck_velocity_swizzle_env_cfg,
)


def test_swizzle_head_control_wired():
    cfg = make_microduck_velocity_swizzle_env_cfg()

    # Head-pose command term exists.
    assert "head_pose" in cfg.commands

    # head_command obs is the REAL command (not zero-padded) on both groups.
    for group in ("actor", "critic"):
        term = cfg.observations[group].terms["head_command"]
        assert term.func is mdp.generated_commands
        assert term.params["command_name"] == "head_pose"

    # head_pose_tracking reward exists.
    assert "head_pose_tracking" in cfg.rewards

    # The two HOME-pullers that would fight the head command are handled:
    #  - neck_joint_pos_l2 removed
    assert "neck_joint_pos_l2" not in cfg.rewards
    #  - pose reward scoped to leg joints via a negative-lookahead regex that
    #    excludes neck/head (and passive wheels)
    pose_joints = cfg.rewards["pose"].params["asset_cfg"].joint_names
    assert any(
        "(?!" in j and "neck" in j and "head" in j for j in pose_joints
    ), f"pose reward not scoped away from neck/head: {pose_joints}"

    # Late head curricula exist.
    assert "head_pose_tracking_weight" in cfg.curriculum
    assert "head_pose_range" in cfg.curriculum
```

- [ ] **第 2 步：运行测试以验证其是否失败**

运行：`MUJOCO_GL=egl uv run pytest tests/test_swizzle_head_cfg.py -v`
预期：失败（head_pose 命令/ head_pose_tracking 奖励缺失；head_command obs 仍然是 `zero_command_padding`）。

- [ ] **第 3 步：向葫芦步环境 cfg 添加导入**

在 `microduck_velocity_swizzle_env_cfg.py` 中扩展导入：在当前的 `from mjlab.managers import CurriculumTermCfg, RewardTermCfg` 中加入 `ObservationTermCfg`，并导入 velocity MDP 以使用 `generated_commands`：

```python
from mjlab.managers import CurriculumTermCfg, ObservationTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp
```

- [ ] **第 4 步：添加 head_pose 命令、真实 head_command 观测、head_pose_tracking 奖励并协调颈部约束**

在 `make_microduck_velocity_swizzle_env_cfg` 内部，在现有奖励和航向设置之后、`return cfg` 之前添加：

```python
    # --- Head-pose control (Y button): the policy produces the head pose ---------
    # Head-pose command (4D deltas from HOME: [neck_pitch, head_pitch, head_yaw,
    # head_roll]). Ported from the velocity env; ranges start small (widened by the
    # curriculum below). Resample every 2-5 s.
    cfg.commands["head_pose"] = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=(2.0, 5.0),
        ranges=(
            (-0.05, 0.05),    # neck_pitch
            (-0.05, 0.05),    # head_pitch
            (-0.07, 0.07),    # head_yaw
            (-0.015, 0.015),  # head_roll (tighter — small mechanical range)
        ),
    )

    # Feed the REAL head command into the obs (replaces zero_command_padding) on
    # both groups. body_command stays zero-padded (no body-pose control here).
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "head_pose"},
        )

    # Reward the head tracking its command. Weight 0 here — ramped in LATE by the
    # curriculum so it doesn't disturb the swizzle before it's solid.
    cfg.rewards["head_pose_tracking"] = RewardTermCfg(
        func=microduck_mdp.head_pose_tracking,
        weight=0.0,
        params={"command_name": "head_pose", "std": 0.5},
    )

    # Reconcile the two HOME-pullers that would fight head_pose_tracking:
    #  1) neck_joint_pos_l2 pulls the neck/head joints to HOME -> remove it.
    if "neck_joint_pos_l2" in cfg.rewards:
        del cfg.rewards["neck_joint_pos_l2"]
    #  2) the pose reward includes neck/head -> scope it to LEG joints only.
    cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=(r"^(?!passive_|.*neck.*|.*head.*).*",)
    )
```

- [ ] **第5步：添加后期头部课程**

紧接在步骤 4 的块之后（仍在 `return cfg` 之前）：

```python
    # head_pose_tracking ramps 0 -> 4.0, staying 0 until ~1500 it. (swizzle solid),
    # so head control is added on top of a stable swizzle.
    cfg.curriculum["head_pose_tracking_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "head_pose_tracking",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},   # must match initial weight
                {"step": 1500 * 24,  "weight": 0.0},   # head off while swizzle solidifies
                {"step": 2250 * 24,  "weight": 2.0},
                {"step": 3000 * 24,  "weight": 4.0},
            ],
        },
    )
    # Head-command range widens over the SAME window (tiny until 1500, full by 3000),
    # so the commanded head barely moves early and reaches full range once the policy
    # can handle it.
    cfg.curriculum["head_pose_range"] = CurriculumTermCfg(
        func=microduck_mdp.pose_command_range_curriculum,
        params={
            "command_name": "head_pose",
            "range_stages": [
                # step,               ((neck_pitch), (head_pitch), (head_yaw),  (head_roll))
                {"step": 0,          "ranges": ((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015))},
                {"step": 1500 * 24,  "ranges": ((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015))},
                {"step": 2250 * 24,  "ranges": ((-0.55, 0.55), (-0.55, 0.55), (-0.70, 0.70), (-0.15, 0.15))},
                {"step": 3000 * 24,  "ranges": ((-1.10, 1.10), (-1.10, 1.10), (-1.40, 1.40), (-0.31, 0.31))},
            ],
        },
    )
```

- [ ] **第 6 步：运行 cfg 测试以验证其通过**

运行：`MUJOCO_GL=egl uv run pytest tests/test_swizzle_head_cfg.py -v`
预期：通过。

- [ ] **第 7 步：对环境进行端到端冒烟测试**

运行：`MUJOCO_GL=egl uv run train Mjlab-Velocity-Swizzle-MicroDuck --env.scene.num-envs 16 --agent.max-iterations 2`
预期：没有错误；奖励日志列出`head_pose_tracking`，不再列出`neck_joint_pos_l2`； `Curriculum/head_pose_tracking_weight` 线出现在值 0.0 处。

- [ ] **第 8 步：提交改动**

```bash
git add src/mjlab_microduck/tasks/microduck_velocity_swizzle_env_cfg.py tests/test_swizzle_head_cfg.py
git commit -m "swizzle: add head-pose control (Y button, policy-managed, late curriculum)"
```

---

## 完整训练说明（不属于本任务）

由于头部课程约在第 3000 次迭代才完成，因此训练时长要超过之前使用的 2500 次迭代：

```bash
uv run train Mjlab-Velocity-Swizzle-MicroDuck --env.scene.num-envs 4096 --agent.max-iterations 3500
```

观察指标：`head_pose_tracking` 应在约第 1500 次迭代后上升，且课程启用时葫芦步的
跌倒率不应突增。若头部干扰葫芦步，应推迟启用时间或更慢地扩大范围；若头部跟踪不佳，
则提高最终权重。部署方式不变（`--roller --new-cmd-obs`，Y 按钮控制头部）。
