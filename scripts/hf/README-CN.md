# HF Jobs 训练

在 Hugging Face 托管的 GPU 上训练 mjlab-microduck。身份验证使用缓存的 HF
令牌（`hf auth login` 或 `HF_TOKEN`）；所有操作都通过
`huggingface_hub` Python API 完成，不要求安装独立的 `hf` CLI。

## 一次性设置

```fish
hf auth login    # or export HF_TOKEN (any tool that caches the token works)
wandb login      # auto-detected from ~/.netrc and forwarded
```

## 提交训练任务

在常规训练命令后加上 `--hf-jobs`：

```fish
uv run train Mjlab-Kick-Flat-MicroDuck \
    --env.scene.num-envs 4096 --agent.max_iterations 4000 --hf-jobs
```

系统会询问任务在哪个命名空间下运行，可以选择个人账户或所属组织。
代码仓库、uv 缓存 bucket、计费和任务本身都位于所选命名空间中。
传入 `--namespace <name>` 可跳过提示；非交互运行默认使用个人命名空间。

不带 `--hf-jobs` 时，命令行为与之前完全相同，即在本地训练。提交相关参数
在本地消费，其余参数会原样转发给任务容器内的 `uv run train`。

常用参数：

- `--namespace <name>`：运行任务的账户或组织，并跳过选择提示
- `--flavor l4x1`（默认）/ `a10g-large` / `a100-large`
- `--timeout 12h`（默认）：超过该时间后终止任务
- `--detach`：提交后立即返回；默认会持续输出日志，按 Ctrl-C 只会断开而不会终止任务
- `--dry-run`：构建 tarball 并打印任务配置，但不提交
- `--run-name <tag>`：覆盖自动生成的 `<task>-<timestamp>` 名称
- `--no-uv-cache`：禁用持久化的 `uv` 缓存 bucket，每次运行都会承担首次下载成本
- `--no-wandb`：不转发 wandb 密钥

`uv run scripts/hf/train_hf.py <task> ...` 仍然可用；它只是同一套代码的
兼容入口，实际实现位于 `src/mjlab_microduck/hf_jobs.py`。

## 内部流程

1. `git ls-files` 会对当前仓库中已跟踪和未提交的文件制作快照（感知工作树），生成 `src-<stamp>.tar.gz`。
2. tarball 上传到私有数据集 `<namespace>/mjlab-microduck-src`。
3. 创建私有模型仓库 `<namespace>/<run-name>`，用于保存检查点。
4. 私有 HF bucket `<namespace>/mjlab-uv-cache` 挂载到 `/uv-cache`，并用作 `UV_CACHE_DIR`，使 wheel 下载可跨任务复用；首次运行较慢，后续运行较快。
5. `HfApi.run_job` 启动容器，容器会：
   - 安装 `uv`、解压 tarball，并运行 `uv sync`（使用预热缓存）；
   - 在后台启动 `scripts/hf/uploader.py`，监视 `logs/rsl_rl/**/model_*.pt` 并每 60 秒上传一次；
   - 运行 `uv run train <task> <args>`；
   - 退出时执行最后一次上传。
6. wandb 凭据作为 secret 转发，训练会实时出现在你的 wandb 项目中。

## 浏览检查点

任务开始时，提交程序会打印
`https://huggingface.co/<namespace>/<run-name>`；训练期间新的 `.pt` 文件会持续出现在该页面。

## 管理任务

提交时会打印任务 ID 和 URL。可以通过 Python 管理任务：

```python
from huggingface_hub import HfApi
api = HfApi()
api.list_jobs()                                  # or namespace="pollen-robotics"
for l in api.fetch_job_logs(job_id="...", follow=True): print(l)
api.cancel_job(job_id="...")
```

如果已安装 CLI，也可以使用 `hf jobs ps/logs/cancel`。
