# CLI 参数分层与兼容矩阵

`distill-run` 目前提供 61 个一级参数。参数不是所有引擎通用：解析后会根据
`--engine` 校验，不属于当前引擎的参数会直接报错，不会被静默忽略。

参数分为六层：

1. 全局运行参数：所有引擎通用。
2. 教师生成参数：`easydistill` 与 `easydistill-swift` 使用。
3. 通用训练参数：`swift`、`trl` 与 `easydistill-swift` 使用。
4. Swift 专属参数。
5. TRL 白盒蒸馏专属参数。
6. 模型、数据与分布式运行参数。

当前命令形式是 `distill-run --engine trl ...`。如果未来改为
`distill-run trl ...` 子命令，参数归属保持不变，只改变帮助信息和解析形式：
`distill-run trl --help` 只显示全局、通用训练及 TRL 专属参数。

图例：✓ 表示引擎接受该参数，— 表示不接受。`E` = EasyDistill，
`S` = Swift，`T` = TRL，`ES` = EasyDistill-Swift。

## 全局运行参数（15）

| 参数 | 类型 | 默认值 | E | S | T | ES | 说明 |
| --- | --- | --- | :-: | :-: | :-: | :-: | --- |
| `--engine` | string | 必填 | ✓ | ✓ | ✓ | ✓ | 选择 `easydistill`、`swift`、`trl` 或 `easydistill-swift` |
| `--output` | path | 必填 | ✓ | ✓ | ✓ | ✓ | 生成引擎使用 JSONL 文件，其余引擎使用目录 |
| `--config` | path | 无 | ✓ | ✓ | ✓ | ✓ | 可选外部 YAML/JSON 配置 |
| `--config-json` | JSON/YAML | 无 | ✓ | ✓ | ✓ | ✓ | 命令内传入完整或部分配置 |
| `--show-default-config` | bool | false | ✓ | ✓ | ✓ | ✓ | 打印当前引擎内置配置后退出 |
| `--set` | `key=value` | 无 | ✓ | ✓ | ✓ | ✓ | 覆盖未提升为一级参数的框架字段，可重复 |
| `--run-id` | string | 时间戳 ID | ✓ | ✓ | ✓ | ✓ | 本次运行标识 |
| `--work-dir` | path | `~/.cache/distill-run` | ✓ | ✓ | ✓ | ✓ | 缓存及中间文件目录 |
| `--model-cache-dir` | path | `<work-dir>/models` | — | ✓ | ✓ | ✓ | Hub 模型的容器本地缓存 |
| `--model-revision` | string | Hub 默认版本 | — | ✓ | ✓ | ✓ | Teacher/Student 模型 revision 或 commit |
| `--dataset-cache-dir` | path | `<work-dir>/cache` | ✓ | ✓ | ✓ | ✓ | 网络数据集缓存目录 |
| `--dataset-format` | string | `auto` | ✓ | ✓ | ✓ | ✓ | `auto`、`jsonl`、`json`、`parquet` 或 `arrow` |
| `--dataset-revision` | string | Hub 默认版本 | ✓ | ✓ | ✓ | ✓ | 数据集 revision 或 commit |
| `--dataset-subset` | string | 无 | ✓ | ✓ | ✓ | ✓ | Hugging Face / ModelScope 数据集子集 |
| `--dataset-split` | string | `train` | ✓ | ✓ | ✓ | ✓ | 选择 train、validation、test 等 split |
| `--dataset-max-samples` | int | 无限制 | ✓ | ✓ | ✓ | ✓ | 确定性抽样的最大标准化样本数 |
| `--dataset-shuffle-seed` | int | `42` | ✓ | ✓ | ✓ | ✓ | 抽样随机种子 |

## 教师生成参数（14）

| 参数 | 类型 | 默认值 | E | S | T | ES | 说明 |
| --- | --- | --- | :-: | :-: | :-: | :-: | --- |
| `--job-type` | string | `instruct_distill` | ✓ | — | — | ✓ | EasyDistill 任务类型 |
| `--backend-type` | string | `openai` | ✓ | — | — | ✓ | 教师 API 后端类型 |
| `--system-prompt` | string | `You are a helpful assistant.` | ✓ | — | — | ✓ | 教师系统提示词 |
| `--temperature` | float | E/ES `0.7`；T `1.0` | ✓ | — | ✓ | ✓ | E/ES 为生成温度，T 为蒸馏温度 |
| `--max-tokens` | int | E `512`；ES `1024` | ✓ | — | — | ✓ | 教师单次生成的最大 token 数 |
| `--max-workers` | int | `4` | ✓ | — | — | ✓ | 并发教师请求数 |
| `--show-progress` / `--no-show-progress` | bool | true | ✓ | — | — | ✓ | 是否显示生成进度 |
| `--instruction-key` | string | `instruction` | ✓ | — | — | ✓ | 种子 JSON 中的指令字段 |
| `--skip-empty` / `--no-skip-empty` | bool | true | ✓ | — | — | ✓ | 是否跳过空指令 |
| `--min-length` | int | `10` | ✓ | — | — | ✓ | 最短指令长度 |
| `--teacher-base-url` | URL | 必填或配置提供 | ✓ | — | — | ✓ | OpenAI 兼容教师地址 |
| `--teacher-model-id` | string | 必填或配置提供 | ✓ | — | — | ✓ | API 暴露的教师模型 ID |
| `--teacher-api-key` | string | `EMPTY` | ✓ | — | — | ✓ | 教师 API 密钥 |
| `--input` | URI/path | 必填或配置提供 | ✓ | — | — | ✓ | 种子指令数据 |

`--temperature` 是跨领域参数，但含义由引擎限定。若后续采用子命令，
可以继续保留相同名称，因为每个子命令拥有独立帮助信息。

## 通用训练参数（12）

| 参数 | 类型 | 默认值 | E | S | T | ES | 说明 |
| --- | --- | --- | :-: | :-: | :-: | :-: | --- |
| `--lora-rank` | int | `8` | — | ✓ | ✓ | ✓ | LoRA rank |
| `--lora-alpha` | int | `32` | — | ✓ | ✓ | ✓ | LoRA alpha |
| `--num-train-epochs` | float | S/ES `3`；T `1` | — | ✓ | ✓ | ✓ | 训练轮数 |
| `--max-steps` | int | S/ES 框架默认；T `100` | — | ✓ | ✓ | ✓ | 最大优化步数 |
| `--per-device-train-batch-size` | int | `1` | — | ✓ | ✓ | ✓ | 每设备训练批量 |
| `--gradient-accumulation-steps` | int | `8` | — | ✓ | ✓ | ✓ | 梯度累积步数 |
| `--learning-rate` | float | S/ES `1e-4`；T `1e-5` | — | ✓ | ✓ | ✓ | 学习率 |
| `--logging-steps` | int | S/ES `5`；T `1` | — | ✓ | ✓ | ✓ | 日志间隔 |
| `--save-steps` | int | `200` | — | ✓ | ✓ | ✓ | checkpoint 间隔 |
| `--save-total-limit` | int | `2` | — | ✓ | ✓ | ✓ | 最多保留 checkpoint 数 |
| `--gradient-checkpointing` / `--no-gradient-checkpointing` | bool | true | — | ✓ | ✓ | ✓ | 梯度检查点 |
| `--torch-dtype` | string | `bfloat16` | — | ✓ | ✓ | ✓ | Swift/EasyDistill-Swift 模型或 TRL student dtype |

## Swift 专属参数（2）

| 参数 | 类型 | 默认值 | E | S | T | ES | 说明 |
| --- | --- | --- | :-: | :-: | :-: | :-: | --- |
| `--tuner-type` | string | `lora` | — | ✓ | — | ✓ | ms-swift tuner 类型 |
| `--max-length` | int | `2048` | — | ✓ | — | ✓ | SFT 最大序列长度 |

## TRL 白盒蒸馏专属参数（9）

| 参数 | 类型 | 默认值 | E | S | T | ES | 说明 |
| --- | --- | --- | :-: | :-: | :-: | :-: | --- |
| `--lora-dropout` | float | `0.05` | — | — | ✓ | — | Student LoRA dropout |
| `--lora-enabled` / `--no-lora-enabled` | bool | true | — | — | ✓ | — | 是否使用 LoRA |
| `--target-modules` | CSV | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` | — | — | ✓ | — | LoRA 目标模块 |
| `--max-completion-length` | int | `256` | — | — | ✓ | — | 教师生成 completion 的最大长度 |
| `--bf16` / `--no-bf16` | bool | true | — | — | ✓ | — | TRL bf16 训练 |
| `--beta` | float | `0.5` | — | — | ✓ | — | 蒸馏损失权重 |
| `--teacher-torch-dtype` | string | `bfloat16` | — | — | ✓ | — | Teacher 权重 dtype |
| `--trust-remote-code` / `--no-trust-remote-code` | bool | true | — | — | ✓ | — | Student 是否信任模型仓库代码 |
| `--teacher-trust-remote-code` / `--no-teacher-trust-remote-code` | bool | true | — | — | ✓ | — | Teacher 是否信任模型仓库代码 |

TRL 同时接受教师生成参数表中的 `--temperature`。

## 模型、数据与分布式参数（9）

| 参数 | 类型 | 默认值 | E | S | T | ES | 说明 |
| --- | --- | --- | :-: | :-: | :-: | :-: | --- |
| `--student` / `--model` | URI/path | 必填 | — | ✓ | ✓ | ✓ | 本地、`hf://` 或 `ms://` Student 权重 |
| `--teacher` | URI/path | 必填 | — | — | ✓ | — | 本地、`hf://` 或 `ms://` 白盒 Teacher 权重 |
| `--dataset` / `--data` | URI/path | 必填或配置提供 | — | ✓ | ✓ | — | Swift SFT 数据或 TRL prompts |
| `--resume` / `--no-resume` | bool | false | — | ✓ | ✓ | ✓ | 从输出目录最新 checkpoint 恢复 |
| `--num-processes` / `--nproc-per-node` | int | `1` | — | ✓ | ✓ | ✓ | 单机训练进程/GPU 数 |
| `--distributed-strategy` | enum | `zero3` | — | — | ✓ | — | `fsdp2`、`zero1`、`zero2` 或 `zero3` |
| `--accelerate-config` | path | 无 | — | — | ✓ | — | 自定义 Accelerate 配置，覆盖 strategy |
| `--main-process-port` | int | `29500` | — | ✓ | ✓ | ✓ | 分布式 rendezvous 端口 |
| `--device` | int | `0` | — | ✓ | ✓ | ✓ | 单进程使用的逻辑 GPU |

## 配置优先级

从低到高：

```text
内置默认值
  < --config
  < --config-json
  < --set
  < 一级 CLI 参数 / 对应环境变量
```

一级参数只在用户显式传入或对应环境变量存在时覆盖配置。未传一级参数时，
不会覆盖 `--config-json` 或 `--set` 中的值。

## 无外部配置文件示例

### EasyDistill

```bash
distill-run --engine easydistill \
  --teacher-base-url http://teacher:8000/v1 \
  --teacher-model-id Qwen3.5-27B \
  --input hf://organization/seed-dataset \
  --output /output/sft.jsonl \
  --temperature 0.7 --max-tokens 512 --max-workers 4
```

### Swift

```bash
distill-run --engine swift \
  --student /models/Qwen3-0.6B \
  --dataset /data/sft.jsonl \
  --output /output/swift \
  --lora-rank 8 --lora-alpha 32 \
  --num-train-epochs 3 --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 --learning-rate 1e-4 \
  --max-length 2048
```

### TRL

```bash
distill-run --engine trl \
  --teacher /models/Qwen3.8-27B \
  --student /models/Qwen3.5-0.8B \
  --dataset /data/prompts.jsonl \
  --output /output/trl \
  --num-processes 6 --distributed-strategy zero3 \
  --max-steps 100 --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 --learning-rate 1e-5 \
  --temperature 1.0 --beta 0.5 \
  --lora-rank 8 --lora-alpha 32
```

### EasyDistill-Swift

```bash
distill-run --engine easydistill-swift \
  --teacher-base-url http://teacher:8000/v1 \
  --teacher-model-id Qwen3.5-27B \
  --input /data/seed.jsonl \
  --student /models/Qwen3-0.6B \
  --output /output/easydistill-swift \
  --temperature 0.7 --max-tokens 1024 --max-workers 4 \
  --lora-rank 8 --lora-alpha 32 \
  --num-train-epochs 3 --learning-rate 1e-4 --max-length 2048
```

`--set dotted.path=value` 只用于尚未提升为一级参数的框架选项，例如：

```bash
distill-run --engine trl ... \
  --set trl.weight_decay=0.01 \
  --set trl.lr_scheduler_type=cosine
```
