# AGENTS.md 中文版

Microduck 的 RL 训练环境。Microduck 是一台约 800 克、约 25 厘米高、
配备 14 个 Dynamixel XL330 舵机的双足机器人。本项目基于
[mjlab](https://github.com/mujocolab/mjlab)（MuJoCo Warp）和 PPO（rsl_rl）构建。
策略在这里以 50 Hz 训练、导出为 ONNX，再由 `pollen-robotics/microduck`
仓库中的运行时部署到真实机器人。Sim2real 迁移是本项目的核心：下面的每条约定
都源于实际教训，破坏它们会导致策略在查看器中正常、上硬件却失败。

## 命令

```bash
uv run list-envs                                    # live task registry
uv run train <TASK_ID> --env.scene.num-envs 4096    # train (add --hf-jobs for Hugging Face Jobs)
uv run train <TASK_ID> --env.scene.num-envs 64 --agent.max_iterations 5   # SMOKE TEST — always run first
uv run play <TASK_ID> --wandb-run-path <entity/project/run_id>
uv run scripts/export.py <TASK_ID> --wandb-run-path <...>   # → ONNX (bakes obs normalizer — mandatory path)
uv run publish --task <TASK_ID> --wandb-run-path <...> --checkpoint N --repo <user>/microduck-<name> --kind episodic --duration-s 4.0
                                                    # → HF Hub repo (policy.onnx + schema-2 manifest.json + README) the daemon loads via `robotctl policy add`
uv run scripts/infer_policy.py --walking out.onnx   # CPU MuJoCo deployment rehearsal (BAM M6 actuators as in training; --no-bam = XML PD)
uv run --with pytest pytest tests/
```

64 个环境、5 次迭代的冒烟测试能以极低成本发现约 95% 的配置错误。
未经冒烟测试，绝不要启动长时间训练。

## 仓库结构

- `src/mjlab_microduck/tasks/mdp.py` — 所有自定义 MDP 函数（奖励、事件、
观测、命令、课程）。新增函数放在这里，并按任务分组。
- `src/mjlab_microduck/tasks/microduck_*_env_cfg.py` — 每个任务一个 cfg 模块
系列。`microduck_velocity_env_cfg.py` 是主要行走配方，也是其他环境构建或
镜像使用的共享基础（机器人、DR、观测、命令）。
- `src/mjlab_microduck/tasks/__init__.py` — 任务注册（基础 + `-Backlash-` 变体）。
- `src/mjlab_microduck/tasks/backlash.py` — 将任意 env cfg 包装为对应的齿隙变体。
- `src/mjlab_microduck/robot/microduck_constants.py` — 机器人配置、HOME 框架、BAM 执行器配置。
- `src/mjlab_microduck/robot/microduck/` — MJCF 从 Onshape 导出
（onshape-to-robot，每个模型一个 `config_mjcf_*.json`）+ 场景 + `add_backlash.py`。
碰撞系列：`walk`（仅脚）、`groundcontact`（精选地板组）、
`allcollisions`（每个零件；XL330 外壳命名为 `*_servo_collision`
`name_servo_collision_geoms` → VelStand 的伺服冲击传感器）。每个都有一个
`_backlash` 由 `add_backlash.py <xml> --backlash-deg 2.0` 生成的对应变体。
- `src/mjlab_microduck/actuator/friction_dr_bam.py` — BAM 执行器 + 摩擦 DR + 间隙编码器。
- `src/mjlab_microduck/export.py` — ONNX 导出（内置标准化器）； `scripts/export.py` 包裹它。
- `src/mjlab_microduck/publish/` — `uv run publish`：schema-2 manifest 构建器、
ONNX 形状/冒烟检查和 Hub 上传。契约定义在 `microduck` 仓库的
`docs/policy-manifest.md`；只有恒定命令的 episodic/perpetual 策略可发布
（相位/姿态标志驱动的策略属于固定策略集合）。
- `scripts/` — 导出包装器、推断、sim2real 比较、wandb 帮助程序。
- `tests/` — cfg 不变和 mdp 函数回归测试（CPU，不需要 GPU）。

## 不变量——不要破坏这些

- **观测布局为 61D（Actor），并在整个策略系列中共享**，因此
策略在运行时可热插拔：48 个基本本体感觉 +
13D指令块`[twist(3), head_pose(4), body_pose(6)]`，按此顺序。
不使用某个命令槽的环境也必须将其补零（保留观测项，并采样极小范围），
绝不能删除该槽位。
- **关节布局**（14 个舵机，ctrl idx = 行走/地面接触上的关节 idx
模型）：0–4 左腿（hip_yaw、hip_roll、hip_pitch、膝盖、脚踝）、5–8
颈部/头部（neck_pitch、head_pitch、head_yaw、head_roll），9–13 右腿。
在滚轮/齿隙模型中，被动关节会穿插其间，因此 MDP 函数绝不能硬编码关节索引；
请使用 mdp.py 中的 `_servo_joint_ids` / `_servo_joint_pos` 辅助函数
（普通模型上是恒等映射，其他模型上也能正确映射）。
- **非驱动关节全部命名为 `passive_*`**（轮子、间隙铰链）。
每个执行器/obs/奖励选择器都使用 `^(?!passive_).*` — 保留前缀
添加关节时的约定，新的 `passive_` 正则表达式不得
意外匹配间隙接头（`^passive_.*wheel`，而不是 `^passive_.*`）。
- **执行器采用 BAM**（电压控制的 XL330 模型，摩擦由执行器计算）。
由此有两个后果：任何独立环境配置都必须注册
`expand_bam_friction_fields` 启动事件，关节摩擦 DR 必须缩放
执行器的 `friction_scale` — `dof_frictionloss` 在 BAM 下归零，因此
随机化它是一个静默空操作。
- **观测标准化已开启**，因此标准化器必须内嵌到 ONNX 中。
`scripts/export.py` 会完成这一步；仿真内 play 始终会应用标准化器，因而会掩盖问题。
绝不要手工转换检查点。
- **策略不使用滤波**（训练期间没有动作低通）。除非运行时有匹配开关且完成迁移测试，
否则不要添加 EMA 滤波；训练和部署两端只要有一端使用、另一端不用，就会破坏迁移效果。
- **域随机化不得在重置过程中累积。** mjlab 1.3.0 的
`dr.*` 与 `operation="add"/"scale"` 的操作本质上是非累积的（它们
重新读取编译时默认值）；自定义 DR 功能必须先恢复再应用。
累积的 CoM 随机发生器曾经在几个月的每次长期运行中都会降级。
- 如果某个观测量被映射为传感器视角（齿隙编码器、偏置），那么跟踪同一物理量的
奖励也必须使用相同视角；否则策略会因修正自己观测到的误差而受到惩罚。
- `-Backlash-` 任务变体必须与基础任务使用同一种机器人模型
（walk / groundcontact / rollers），保证齿隙 A/B 对比不受其他变量干扰。

## 构建新环境——工作流程

1. **选择最接近的模板**并在其基础上构建，不要从零开始：
   运动控制用 velocity 配方；以某个姿态结束的单次动作使用 standup；
   命令控制的双状态切换使用 sitstand；动态动作使用 roulade
   （其配置文档字符串记录了 5 次训练总结出的课程设计）。基于
   `make_microduck_velocity*_env_cfg` 可以自动保持 DR、观测、噪声和延迟同步。
   如果直接从 mjlab 基础模板构建独立环境，则必须自行移植完整的 DR、观测噪声和
   NaN 防护栈；可检索 velocity 环境接入的内容：`_safe` Critic 观测项、带
   sensor_names 的 `nan_state` 终止项、`expand_bam_friction_fields`、编码器偏置和 IMU 安装偏差。
2. **训练前先在仿真中验证物理假设**，这是最有效的省时手段：
   - 目标/静止姿态必须是稳定平衡点：从带噪初始状态开始，保持控制量 3 秒并检查倾角，
     不能只看高度。只记录 z 的稳定性测试会把已经倒地的状态误判为“稳定静止”。
   - 目标高度必须在当前机器人模型的仿真中实测（例如让站立策略运行后读取躯干 z），
     不要沿用其他模型版本的数值。STAND_Z 偏差 5 mm 就曾让目标连续数日不可达。
3. **配置约定**：在配置文件顶部放置 `ENABLE_*` 开关和调参常量；提供
   `make_..._env_cfg(play: bool, rough: bool)` 工厂；在 `tasks/__init__.py` 注册
   （适用时也加入 `_BACKLASH_TASKS` 表）；使用独立的 `RslRl...RunnerCfg` 和
   `experiment_name`。可使用 `symmetry.py` 中的 61D 镜像表做对称损失，但默认关闭，
   非对称任务绝不能启用。
4. **编写配置测试**（参见 `tests/test_*_cfg.py`）：验证关节索引能在实际模型上解析、
   奖励权重符号正确、门控按预期开启或关闭。这些测试可在 CPU 上运行并锁定不变量。
5. **冒烟测试**（64 个环境、5 次迭代）：确认环境能构建和步进、没有 NaN、Actor
   观测为 61D、所有奖励项都能计算、ONNX 能导出。
6. 开始训练并观察日志（见下文）。通常要用 2–5 次迭代修正奖励投机行为，
这是正常现象；下面的经验能省去大部分试错。

## 奖励设计——每条规则都是经过惨痛教训才学会的

- **符号约定（曾影响四个环境）：** mdp.py 中有两种惩罚形式。mjlab 基础
成本函数返回 ≥ 0，因此使用负权重；Microduck 自带的自取负函数
（`*_penalty`、`*_l1` 返回 ≤ 0）则使用正权重。给自取负惩罚配负权重会发生
双重取负，等于奖励违规行为，策略会主动利用它（臀部弹跳、撞击式坐下）。
**最可靠的检查：每次运行中，wandb 的每个 `Episode_Reward/<penalty>` 都必须 ≤ 0。**
- **RL 只会严格按奖励字面优化。** 每个未约束的自由度都会被利用
（用弹射甩动代替翻滚、用肩滚代替矢状面翻滚、用头部三点支撑代替站立）。
应使用基于状态的硬门控来定义动作是否成立（支撑接触、方向轴检查、锁存状态），
而不是依赖很小的惩罚项去暗示。
- **禁止大奖池：**任何“到达 X”的奖励都必须限速或渐变。
过早到达目标后逐步领取奖励，会形成足以抵消任意暴力动作的大奖池。
对命令式转换，应跟踪以恒定速率变化的内部目标；领先目标斜坡不会得分，
因此慢速动作才是最优解。仅靠速度上限惩罚的累计成本有界，最终会失效。
- **不要把正奖励门控在坏状态上**（倒地、低高度），否则策略会停在满足门控且成本最低的
  姿态持续刷奖励。应使用基于势函数的塑形，只奖励进度变化，例如 Δcos(倾角)：
  起身有奖励，保持不动为零，无法反复刷取。对静止类任务，要用仰卧、俯卧、侧卧等
  每一种稳定倒地姿态审计所有正奖励；如果倒地仍能保留奖励栈的大部分，策略就会选择倒地。
- **回合式姿态落地任务：**从 t=0 开始使用单一固定目标，对关节和高度采用较宽容的
  高斯项 + L1 项，再加 |a_z| 冲击惩罚和两层 upright 约束。不要使用关键帧/航点轨迹，
  否则策略会停在航点刷奖励；具体路径应由 RL 自行发现。
- **正则项分为两类。**运动抑制项（body_ang_vel、角动量、姿态标准差）会惩罚动态动作
  本身必需的运动，动态任务中应保持较低权重。平滑项（`action_rate`、
  joint_torque_rate）能抑制抖动而不阻止缓慢的大幅动作，可以使用较高权重，但应在技能
  已经学会后再通过课程从接近 0 引入。困难技能探索期间的任何“尝试成本”都会让“不动”胜出。
  缓慢精细的任务（如 reaching）需要比行走更强的平滑约束。
- **在不同环境间复制正则项时，应比较奖励量级，而不是只看权重。** PPO 使用相对优势；
  当正向任务奖励总量增大 4 倍时，相同的 `action_rate` 权重相对上就弱了 4 倍。
- **跟踪高斯的标准差**应接近“仍然需要关注的误差”，而不是最大误差；过宽会让小误差区间
  几乎没有梯度。但收紧前要判断误差是策略可消除的，还是目标动作的固有组成部分。
  Microduck 的头部占体重约 38%，行走时必然摆动；过紧的瞬时头部跟踪标准差会重罚行走，
  使策略宁可站着不动。只惩罚可消除部分，例如对 1 秒 EMA 使用 L1，只计入直流偏差并让振荡抵消。
- **目标状态中，乘法组合通常优于加法求和。** 加法奖励容易形成折中吸引域，例如通过倾斜
  同时获得每项 80% 的分数；高斯乘积会在任一因素不足时整体下降。不过标准差必须足够宽，
  让当前策略也能得到可见分数，否则梯度不可见，训练不会改变。
- **关节卡在硬限位时，**应对问题关节增加基于 qpos 的限位接近惩罚。
  内置 `dof_pos_limits` 只在行程最后约 7.5% 生效，控制命令侧惩罚也不起作用；较宽的
  ctrlrange 是有意设计，因为低 kp 舵机需要超调。

## 命令、观测与死权重

- **从未出现非零值的命令输入，其连接权重会永久失活。** 每个命令槽从第 0 步起
都应保留很小的非零采样范围，即使对应奖励权重为 0，也要让输入神经元保持活跃，
以便后续课程继续使用。
- **零命令行为必须经过显式训练**（`zero_command_prob` 式
精确零采样）：均匀采样基本上不会产生全零
命令，这正是部署空闲状态。
- 罕见但重要的命令区域必须设置显式采样桶，例如原地转向
  (`rel_turn_in_place_envs`)。各维独立均匀采样时，原地旋转只占约 2% 的样本，无法学会。

## 课程设置

- 步骤是环境步骤：`iteration × 24` (`NUM_STEPS_PER_ENV = 24`)。
- 使用已验证的职责划分：通过 `microduck_mdp.reward_weight` 调度奖励权重，
  用专门的参数课程调整命令和事件范围。`mdp.reward_weight` 是阶跃函数而非插值函数，
  因此要把连续斜坡离散为多个阶段。
- 必须通过 manager 修改项配置（`env.event_manager.get_term_cfg(...)`），绝不能写
  `env.cfg.events[...]`。manager 初始化时会深拷贝配置，之后修改 `env.cfg` 会静默失效；
  强制修改初始状态的评估脚本也容易踩到这个问题。
- **每个阶段都要与策略实际掌握的能力对齐：**当前阶段尚未稳定前，不要提高初始状态混合的
  难度；技能尚未形成前，不要加入惩罚项。如果 wandb 指标恰好在课程阶段边界下降，说明
  节奏设置错误，应延长阶段或推迟引入，绝不能提前。
- 反向课程初始状态（从动作中途开始回合，包括接近完成的位置）是解决“学会起步却学不会
  最后一段”的可靠方法；否则策略前沿没有 on-policy 数据。

## 训练操作与运行分析

- wandb 项目为 `mjlab_microduck`；本地日志位于 `logs/<experiment_name>/`；使用
`--agent.load-checkpoint model_XXXX.pt --agent.resume True` 恢复训练。
- **热启动不等于恢复训练。** mjlab runner 会把 `common_step_counter` 和迭代数保存到
  checkpoint 并在加载时恢复，因此直接加载其他任务的 checkpoint，会让所有基于步数的课程
  在第 1 次迭代就跳到最终阶段。加前缀 `MICRODUCK_WARM_START=1`（mdp.py Patch 5，
  也会转发给 HF Jobs）可保留权重、标准化器和优化器，同时把两种计数器重置为 0。
  还要在目标配置中把源任务继承的课程折叠到最终阶段（见 velstand 的
  `_collapse_curricula_to_final`），因为加载的策略是在这些条件下训练的。只有标准化器兼容时
  才能跨任务热启动；stand 专家的 twist 标准差为 0.005，绝不能用它初始化 walking 环境。
- 已部署策略的来源追踪：Hub 上的 ONNX 文件包含 `run_path=None`；可以把最后一层权重与
  wandb checkpoint 精确匹配来定位训练运行（2026-09：walk = 441tzs6d@3750，
  stand = 69u48n8l@9750；2026-10 Hub v7 velstand = j4i6yoq2@1250；训练沿革见
  [docs/velstand_policy-CN.md](docs/velstand_policy-CN.md)，sitstand 见
  [docs/sitstand_policy-CN.md](docs/sitstand_policy-CN.md)）。
- 每次迭代都要检查：平均奖励上升；回合长度符合任务预期；所有惩罚项 ≤ 0；主要任务奖励
  确实在增长。总奖励可能只因正则项改善而上涨，而目标动作根本没有学会。
  `Episode_Reward/<term>` 记录的是加权值；权重为 0 时，无论行为如何都显示 0，必须结合权重调度解释。
- 训练预算：简单的回合式动作约需 1000 次迭代、4096 个环境；步态和课程较重的恢复任务
  需要 4000–6000 次迭代。
- **先测量，再推测。** 训练“失败”时，应先对实际 checkpoint 做无界面评估，包括各类初始状态
  测试组、终态聚类和角速度曲线，再修改奖励。过去所谓的“失败”曾经只是 checkpoint 太早、
  成功判据把同一种行为簇切成两半，或奖励上限与实测物理冲突。仿真指标通过但视频观感仍可能
  失败，因此既要看视频，也要检查实际接触的 geom 和轴。
- 报告 rollout 的真实表现，例如“能翻滚，但有三分之一概率脸部着地”，不要只说“可用”。
  是否达到要求由用户决定。

## Sim2real 易踩坑（曾耗费数周排查）

- 全新环境中的 `uv sync` 才是基准（HF Jobs 每次都会执行）：任何依赖手工安装
本地包的功能都会在远程失败。务必让 `pyproject.toml` 如实声明依赖。
- **wheel 包按架构区分。** 在 linux-`aarch64`（DGX Spark / GB10）上，PyPI 的
  PyTorch wheel 只有 CPU 版本（`2.9.1+cpu`，`torch.version.cuda is None`），因此
  `torch.cuda.device_count() == 0`，mjlab 的 `select_gpus()` 会索引空列表，在第 0 次迭代前
  抛出 `IndexError`。`[tool.uv.sources]` 只在 `aarch64` 上把 PyTorch 路由到 cu129 索引；
  cu129 与 Warp bundle 使用的 CUDA toolkit 匹配，而 x86_64 / HF Jobs 仍使用 PyPI。
  `tests/test_aarch64_cuda_torch.py` 锁定了两个容易静默失效的条件：PyTorch 必须保持为直接依赖
  （uv 只对直接依赖应用 `[tool.uv.sources]`，删除看似冗余的 `torch==` 会让路由失效），
  且版本约束必须保留 `==`；CUDA 索引包含比 PyPI 更新的构建，使用 `>=` 曾把版本从
  PyTorch 2.9.1 静默升级到 2.13.0。
- 与物理尺度匹配的限制：25 cm 高的机器人在翻滚时自然会达到 3.5–5.5 rad/s。
  不要用人类尺度的直觉设置速度上限；应通过冲击和剧烈抖动（|a_z|、action_rate、
  支撑门控）抑制暴力动作，而不是惩罚旋转速度本身。
- IMU DR 以零为中心——它训练对未对准幅度的容忍度，并且
无法补偿系统安装偏差（即运行时校准）。
- 真实部署会在共享观测契约下热切换 ONNX 策略（行走/站立/动作）。接触机器人前，
  必须在 `scripts/infer_policy.py` 中演练，并确保命令槽写入正确。姿态标志位于 twist 的
  vx 槽；输入全零表示“站立”，否则会表现得像“策略忽略了按钮”。
