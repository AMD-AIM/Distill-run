# 数据集下载与标准化

`distill-run` 会在模型加载前完成数据下载、格式识别、字段标准化和抽样。
所有引擎最终接收本地规范 JSONL，因此 Hugging Face 或 ModelScope 仓库中的
Parquet/Arrow 文件不需要手工转换。

## 支持的数据来源与格式

- 本地路径或 `file://`
- `http://`、`https://`
- `hf://组织/数据集`
- `ms://组织/数据集`、`modelscope://组织/数据集`
- JSONL、JSON 数组、Parquet、Arrow
- 包含多个 shard 和 split 的数据集目录

Parquet 和 Arrow 需要 `datasets` 与 `pyarrow`。容器训练镜像已包含这些依赖；
独立安装可使用：

```bash
pip install 'distill-run[dataset,hub]'
```

## 自动字段映射

Swift SFT 会统一生成：

```json
{"messages":[
  {"role":"user","content":"问题"},
  {"role":"assistant","content":"答案"}
]}
```

可识别：

- `messages`，字段为 `role/content`
- `conversations`，字段为 `from/value`
- `instruction` + `output`
- `prompt` + `response` 或 `completion`
- `question` + `answer`
- `problem` + `solution`
- `query` + `response`

角色名称 `human/user` 会变为 `user`，`gpt/bot/model/assistant` 会变为
`assistant`。

EasyDistill 和 TRL 只需要 prompt。若输入行包含参考答案，标准化过程会删除
assistant 及其后的内容，避免把答案泄漏给 teacher/student generation。

无法转换的行会被跳过；若没有任何有效行，命令在加载模型前以数据错误退出。

## Split 与抽样

```bash
--dataset-split train \
--dataset-max-samples 5000 \
--dataset-shuffle-seed 42
```

当设置最大样本数时使用 reservoir sampling：无需把所有行放进内存，且同一
seed 会得到相同样本。未设置 `--dataset-max-samples` 时保留原始顺序和全部
有效数据。

## 真实 Hub 数据集示例

直接使用 Hugging Face Parquet SFT 数据训练 Swift：

```bash
distill-run --engine swift \
  --student /models/Qwen3-0.6B \
  --dataset hf://HuggingFaceH4/no_robots \
  --dataset-split train \
  --dataset-max-samples 5000 \
  --dataset-shuffle-seed 42 \
  --output /output/no-robots-sft \
  --num-train-epochs 1 \
  --learning-rate 1e-4
```

作为 EasyDistill-Swift 的 seed 时，同一数据集会自动移除已有 assistant
回答，再交给 teacher 重新生成：

```bash
distill-run --engine easydistill-swift \
  --teacher-base-url http://127.0.0.1:8000/v1 \
  --teacher-model-id /models/Qwen3.8-27B \
  --input hf://HuggingFaceH4/no_robots \
  --dataset-max-samples 1000 \
  --student /models/Qwen3-0.6B \
  --output /output/no-robots-distill \
  --num-processes 2 \
  --num-train-epochs 1
```

适合后续测试的主流数据包括：

- `HuggingFaceH4/no_robots`：通用对话，小规模验证方便
- `allenai/tulu-3-sft-mixture`：综合英文 SFT
- `BAAI/Infinity-Instruct`：中英综合指令
- COIG-CQIA：中文指令
- NuminaMath-TIR / OpenR1-Math：数学
- NVIDIA OpenCodeInstruct：代码

大型 Hub 数据集当前会先下载所选 snapshot/shard，再执行抽样。
`--dataset-max-samples` 限制训练行数，但不等于网络端流式下载限制。
