# Microduck RL

<img width="2215" height="884" alt="image" src="https://github.com/user-attachments/assets/5db7cc83-b3ce-4f7c-83f0-0572a63baed7" />


[Microduck](https://github.com/pollen-robotics/microduck) 的 RL 训练环境。
Microduck 是一台约 800 克、约 25 厘米高的双足机器人；本项目基于
[mjlab](https://github.com/mujocolab/mjlab)（MuJoCo Warp）和 PPO 构建。
策略在这里以 50 Hz 训练、导出为 ONNX，并由
[pollen-robotics/microduck](https://github.com/pollen-robotics/microduck) 仓库中的运行时部署到真实机器人。

<!-- 主视觉视频——真实机器人片段：行走、起身、前滚翻、轮滑。
     保持简短（约 30 秒），优先展示真实机器人。 -->

https://github.com/user-attachments/assets/50c3d537-8db2-4005-9d9c-3472faeec4d0

本仓库包含完整的 sim2real 配方：[BAM](https://github.com/Rhoban/bam)
执行器物理、域随机化、齿隙仿真，以及让整套方案真正奏效的奖励设计经验
（精炼后的工作手册见 [AGENTS-CN.md](AGENTS-CN.md)）。

## 快速入门

需要 CUDA GPU（通过 MuJoCo Warp 进行训练）和 [uv](https://docs.astral.sh/uv/)。

> **在 ARM 设备上（DGX Spark / GB10、Jetson）：** 首次运行 `uv sync` 会下载约 2 GB
> 的 CUDA wheel，而 uv 默认的 30 秒 HTTP 超时可能导致下载中断。
> 首次同步前请导出 `UV_HTTP_TIMEOUT=600`。

```bash
git clone https://github.com/pollen-robotics/microduck_rl
cd microduck_rl

# train the walking policy (uses your GPU; ~1-2 h for a usable gait at 4096 envs)
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096

# watch a trained policy in the viewer
uv run play Mjlab-Velocity-Flat-MicroDuck --wandb-run-path <entity/project/run_id>

# export to ONNX for deployment
uv run scripts/export.py Mjlab-Velocity-Flat-MicroDuck --wandb-run-path <...>
uv run publish --onnx output.onnx --repo <user>/microduck-<name> --kind episodic --duration-s 4.0   # share it (see "Publishing a policy")

# drive the exported policy in CPU MuJoCo with the keyboard
uv run scripts/infer_policy.py --walking output.onnx
```

从检查点恢复：

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096 \
    --agent.run-name resume --agent.load-checkpoint model_29999.pt --agent.resume True
```

没有 GPU？在任意训练命令后添加 `--hf-jobs`，即可改用 Hugging Face Jobs
运行（参见 [scripts/hf/README-CN.md](scripts/hf/README-CN.md)）。

## 任务

`uv run list-envs` 会打印当前任务注册表。标注的任务提供平地/崎岖地形变体。

<!-- 展示网格——每个任务系列一个仿真或实机短 GIF，每行 3 个。
     若只录制少量素材，优先顺序为 Velocity、VelStand（跌倒恢复）、
     Roulade、SitStand、Rollers/Swizzle、BallKick。 -->

|任务 ID |地形|描述 |
|---|---|---|
| `Mjlab-Velocity-{Flat,Rough}-MicroDuck` |平坦/粗糙| **主要任务**：按速度命令和头部姿态命令行走 |
| `Mjlab-VelStand-{Flat,Rough}-MicroDuck` |平坦/粗糙|将行走与跌倒恢复集成到一个策略中 |
| `Mjlab-StandUp-{Flat,Rough}-MicroDuck` |平坦/粗糙|从俯卧/仰卧/坐姿起身，然后保持站立并控制身体姿态 |
| `Mjlab-SitStand-{Flat,Rough}-MicroDuck` |平坦/粗糙|由命令控制坐 ↔ 站，动作轻柔，头部可控 |
| `Mjlab-GroundPick-{Flat,Rough}-MicroDuck` |平坦/粗糙|蹲下并用嘴尖接触地面，返回站立|
| `Mjlab-BallKick-Flat-MicroDuck` |平坦|向前踢一个 70 mm / 15 g 的球（Actor 看不到球）|
| `Mjlab-Roulade-Flat-MicroDuck` |平|向前翻滚过头，然后用脚落地 |
| `Mjlab-Velocity-Flat-MicroDuck-Rollers` |平|轮滑速度跟踪（脚下被动轮）|
| `Mjlab-Velocity-Swizzle-MicroDuck` |平|经典对称葫芦步（swizzle）滑行 |
| `Mjlab-RollerCrouch-Flat-MicroDuck` |平|在滚轮上滑行时蹲下|
| `Mjlab-RollerSlope-Flat-MicroDuck` |斜坡|在滚轮上滑下斜坡 |
| `Mjlab-RollerStandUp-Flat-MicroDuck` |平|从地面站到轮子上|
| `Mjlab-Spin-Flat-MicroDuck` |平坦|在滚轮上快速原地旋转 |

部署时，运行时会在这些策略（行走/恢复/技巧）之间热切换。
所有策略共享同一份 61 维观测契约，因此任何策略都能随时接管机器人。
`scripts/infer_policy.py` 会准确复现这一流程：

```bash
uv run scripts/infer_policy.py --walking walk.onnx --standing stand.onnx \
    --sitstand sitstand.onnx --roulade roulade.onnx --new-cmd-obs
```

导出为ONNX

```bash
uv run scripts/export.py Mjlab-Velocity-Flat-MicroDuck \
  --checkpoint-file logs/rsl_rl/velocity/2026-10-25_18-16-40_velocity/model_3000.pt \
  --onnx-file velocity_model_3000.onnx
```


可通过键盘发送速度命令，以及 `G`（触地拾取）、`Y`（坐/站）、`R`（前滚翻）、
`K`/`L`（踢球）；`--debug`、`--save-csv`、`--record` 用于 sim2real 对比。
仿真采用与训练相同的 BAM M6 XL330 模型（电压控制和负载相关摩擦，使用
`bam.mujoco.MujocoController`）。`--vin` / `--vin-drop-gain` / `--kp-fw`
可把训练时的 DR 范围固定为单值，`--no-bam` 则回退到 XML PD 执行器。

### 齿隙变体

每个主要任务都有一个 **Backlash** 对应变体。该模型在 14 个舵机关节中分别串联
±1° 的齿隙（总间隙 2°）。只需在任务 ID 的 `MicroDuck` 前插入 `-Backlash`，
例如 `Mjlab-Velocity-Flat-Backlash-MicroDuck`。

齿隙针对 sim2real 进行了正确建模：每个伺服系统都有一个未驱动的
`passive_<joint>_backlash` 铰链，并且因为真正的编码器位于
齿隙输出端，因此固件 PD 仿真（`BacklashEncoderBamActuator`）和
`joint_pos`/`joint_vel` 观测都会
*透过*齿隙读取（`qpos[servo] + qpos[backlash]`）。观测和动作维度保持不变，
因此 ONNX 导出和运行时无需修改。
参见 `src/mjlab_microduck/tasks/backlash.py`。

## 执行器模型

所有任务均使用面向 Dynamixel XL330 的 [BAM](https://github.com/Rhoban/bam)
M6 执行器模型（电压控制律、反电动势、库仑/斯特里贝克/负载相关摩擦）。
每个环境分别对电池电压、负载下的压降、命令延迟和摩擦幅值进行域随机化
（见 `src/mjlab_microduck/actuator/` 中的 `FrictionDRBamActuator`）。

在这个尺度下，微型舵机驱动约 800 克的双足机器人，执行器保真度构成了
sim2real 差距的主要部分。因此这里一直建模到电压控制律，而不是使用理想 PD。

## 机器人模型

MJCF 模型位于 `src/mjlab_microduck/robot/microduck/`，由 Onshape 通过
[onshape-to-robot](https://github.com/Rhoban/onshape-to-robot) 导出，
每个型号一个 `config_mjcf_*.json`：

| XML |使用者 |
|---|---|
| `robot_walk.xml` |Velocity（移除躯干/头部接触，使跌倒代价较低）|
| `robot_groundcontact.xml` | VelStand、StandUp、SitStand、GroundPick、BallKick、Roulade（针对接触地板的部分精心设计的碰撞集 — 身体可以实际躺在地面上；以前为 `robot_allcollisions.xml`）|
| `robot_groundcontact_rollers.xml` |滚轮任务（被动轮）|
| `robot_allcollisions.xml` |真正的全碰撞模型——每个部分都有一个碰撞几何体。还没有任务使用它 |
| `robot_*_backlash.xml` |齿隙任务变体（由 `add_backlash.py` 生成）|

`scene*.xml` 文件用地板 + 关键帧包裹机器人（站立/坐下/折叠）
用于快速查看和 `infer_policy.py`。

<!-- 图像——并排渲染行走模型与滚轮模型，或显示碰撞几何体；
     用一张图直观展示模型变体的差异。 -->

## 项目结构

```
src/mjlab_microduck/
├── robot/
│   ├── microduck/                    # MJCF exports, export configs, scenes, add_backlash.py
│   └── microduck_constants.py        # robot cfgs, HOME frame, BAM actuator cfg
├── actuator/friction_dr_bam.py       # BAM + friction DR + backlash encoder feedback
├── tasks/
│   ├── __init__.py                   # task registration (base + backlash variants)
│   ├── mdp.py                        # rewards, events, observations, custom classes
│   ├── backlash.py                   # make_backlash_variant() env-cfg wrapper
│   └── microduck_*_env_cfg.py        # one cfg module per task family
├── train_cli.py                      # `train` script (identical to mjlab's)
├── train_hook.py                     # intercepts `train ... --hf-jobs`
└── hf_jobs.py                        # Hugging Face Jobs submission
```

值得了解的约定：

- 所有策略共享相同的观测布局（61 维 Actor 观测：48 维本体感觉 + 命令
  `[twist(3), head_pose(4), body_pose(6)]`），因此运行时可以热切换策略。
  环境不使用某个命令槽时会将其补零，而不是删除该槽。
- 非驱动关节统一命名为 `passive_*`（滚轮、齿隙铰链）；执行器、关节观测和
  姿态奖励使用 `^(?!passive_).*` 选择舵机关节。
- 各环境 cfg 文件顶部使用 `ENABLE_*` 布尔开关控制域随机化。
- 关节布局（14 个舵机）：0–4 左腿（hip_yaw、hip_roll、hip_pitch、knee、
  ankle），5–8 颈部/头部（neck_pitch、head_pitch、head_yaw、head_roll），
  9–13 右腿。
- 导出器会把观测标准化器内嵌到 ONNX 图中。必须部署由
  `scripts/export.py` 生成的 ONNX，绝不能手工转换 checkpoint；否则策略在
  运行时会收到未经标准化的观测。

[AGENTS-CN.md](AGENTS-CN.md) 记录了环境构建流程和项目积累的奖励设计规则，
也供在本仓库中工作的 AI 编码代理使用。

## 发布策略

`uv run publish` 会按机器人守护进程可直接加载的结构把策略发布到 Hugging Face Hub：
一个已内嵌观测标准化器的 `policy.onnx`、一个遵循 schema 2 的 `manifest.json`（规范见
[microduck 策略清单](https://github.com/pollen-robotics/microduck/blob/main/docs/policy-manifest.md)），
以及说明运行方式的 README。任何拥有 Microduck 的用户都可用一条命令安装，
无需发布新的守护进程版本。

```bash
# From a wandb run — exports through the one safe path, then uploads
uv run publish --task Mjlab-PoliteBow-Flat-MicroDuck \
    --wandb-run-path <entity/project/run_id> --checkpoint 3000 \
    --repo <user>/microduck-polite-bow --kind episodic --duration-s 4.0 \
    --description "Bows from a two-foot stand and comes back up."

# From an ONNX you already exported (validated, not re-exported)
uv run publish --onnx output.onnx --repo <user>/microduck-flamingo \
    --kind perpetual --unwind-s 1.5 --twist-help "[flag, side, 0]"

# A new gait for a slot
uv run publish --onnx output.onnx --repo <user>/microduck-my-walk --kind perpetual --slot walk

# Ship a rollout video too: uploaded as replay.mp4, which the Hub's replay widget plays
uv run scripts/render_policy.py output.onnx --seconds 10 --video media/bow.mp4
uv run publish --onnx output.onnx --repo <user>/microduck-bow --kind episodic --duration-s 4.0 \
    --video media/bow.mp4

# A remix (warm-started / fine-tuned from someone's published policy): name the parent so the
# model card sets `base_model` and the Hub links the two
uv run publish --onnx output.onnx --repo <user>/microduck-deep-bow --kind episodic --duration-s 5.0 \
    --base-model <owner>/microduck-polite-bow

# See what would be uploaded without touching the Hub
uv run publish --onnx output.onnx --repo <user>/microduck-bow --kind episodic --duration-s 4.0 --dry-run
```

然后在机器人上：

```bash
sudo robotctl policy add polite-bow <user>/microduck-polite-bow   # episodic: length comes from the manifest
sudo robotctl policy add flamingo <user>/microduck-flamingo --hold 5   # held pose: you pick how long
sudo robotctl policy load walk <user>/microduck-my-walk                # gait: into the walk slot
robotctl robot do polite-bow
```

`--kind` 的含义以及各自的需求：

- **回合式（episodic）**：运行 `--duration-s` 指定的时间，并自行回到站立姿态
  （如踢球、前滚翻、鞠躬）。若按住按钮时需要重复执行，可添加 `--chain`。
- **持续式（perpetual）**：持续运行，直到收到其他指令。分为两类：
  - **步态**（新的行走或站立策略）：只需添加 `--slot walk`（或 `stand`）；
    用户通过 `robotctl policy load walk <repo>` 安装到对应槽位。
  - **保持姿态**（如单脚站立）：使用 `--unwind-s` 指定守护进程在交还控制权前，
    以空闲 twist 命令（`--idle`，默认为全零）驱动多长时间，避免机器人仍单脚站立时
    突然失去控制。用户通过 `policy add ... --hold <seconds>` 将其作为单次动作运行。

上传前，`publish` 会验证计算图形状为 `[1,61] -> [1,14]`（旧版 51 维策略会被
明确拒绝），使用合理输入执行模型，并拒绝 NaN 或恒定输出。它还会根据 git 和
wandb 填写 `training` 块（任务、commit、分支、dirty 标记、run、checkpoint），
且在未指定 `--force` 时拒绝覆盖仓库中已有的 `.onnx`。仓库默认创建为私有；
使用 `--no-private` 创建公开仓库，使用 `--tag v1` 为该修订版本打标签。

只有使用恒定命令的策略才能通过这种方式发布。阶段驱动动作（触地拾取）和由姿态标志
驱动的 sit↔stand 策略由守护进程直接控制，收录在官方策略集合
`pollen-robotics/microduck-policies` 中。

## 测试

```bash
uv run --with pytest pytest tests/
```

这些仅需 CPU 的配置不变量与奖励函数回归测试，会锁定关节索引映射、
奖励符号约定和 NaN 防护。

## 相关项目

- [microduck](https://github.com/pollen-robotics/microduck) — Microduck 项目主页，包括运行导出策略的板载运行时
- [mjlab](https://github.com/mujocolab/mjlab) — 训练框架 (MuJoCo Warp + rsl_rl)
- [BAM](https://github.com/Rhoban/bam) — 更好的执行器模型，作者：Rhoban

## 许可证

本仓库代码采用 Apache License 2.0，详情见 [LICENSE](LICENSE)。

`src/mjlab_microduck/robot/microduck/assets/` 和
`src/mjlab_microduck/robot/xl330_test_bench/assets/` 中的 3D 模型（`.stl` 网格）采用
[知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议（CC BY-NC-SA 4.0）](https://creativecommons.org/licenses/by-nc-sa/4.0/)。
请参阅 [LICENSE-MODELS-CN.md](LICENSE-MODELS-CN.md) 了解确切范围和署名要求。
