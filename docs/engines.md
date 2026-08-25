# Engines

Each engine owns one framework call. Production runs can be configured entirely
with first-level CLI flags; `--config`, `--config-json`, and `--set` remain
available for advanced framework fields.

See [`cli-parameters.md`](cli-parameters.md) for all 61 parameters, defaults, and
the engine compatibility matrix. An engine rejects parameters that belong to a
different one instead of ignoring them.

---

## `easydistill` — black box, stage one

Seed instructions go to a teacher served over an OpenAI-compatible API; the
answers come back as SFT rows.

`--output` is a **JSONL file path**, not a directory. Teacher connection,
generation and dataset fields all have first-level CLI parameters.

The optional config (`configs/easydistill/instruct_distill.yaml`) uses
EasyDistill's own schema. Two fields are always overwritten with resolved values:
`dataset.input_path` (the seed after URI resolution) and `dataset.output_path`
(`--output`). `backend.base_url`, `model_id` and `api_key` are overwritten only
when passed as an argument or environment variable, so a config can hold the
defaults.

```yaml
job_type: instruct_distill
backend:
  type: openai
  base_url: ${TEACHER_BASE_URL}
  model_id: ${TEACHER_MODEL_ID}
  api_key: ${TEACHER_API_KEY:-EMPTY}
generation:
  temperature: 0.7
  max_tokens: 1024
  max_workers: 4
dataset:
  instruction_key: instruction
```

`${VAR}` and `${VAR:-default}` are expanded when the config loads, and the
expansion environment includes values that arrived as command-line arguments —
`${TEACHER_BASE_URL}` resolves even when it was passed as `--teacher-base-url`.

**Seed format:** JSONL where each row has one of `instruction`, `prompt`, `query`
or `messages`. Checked before the first API call.

**Preflight:** `GET <base-url>/models` must answer. A teacher that is still
loading fails here in seconds rather than mid-generation.

**Called as a library**, not a subprocess. Set
`DISTILL_EASYDISTILL_MODE=subprocess` to use its CLI instead.

---

## `swift` — black box, stage two

Student SFT on an SFT JSONL, via ms-swift.

`--output` is a directory. Model, data, LoRA and training hyper-parameters are
available as first-level CLI flags.

In an optional config, everything under `swift:` is passed through as
`--key value` to `swift sft`.

```yaml
swift:
  tuner_type: lora        # ms-swift 4.x name; it was train_type in 3.x
  lora_rank: 8
  num_train_epochs: 3
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 1.0e-4
  max_length: 2048
  torch_dtype: bfloat16
```

**Preflight** checks every key against the installed `SftArguments` and suggests
close matches:

```
ConfigError: ms-swift sft does not accept these config keys:
  swift.train_type (did you mean: tuner_type, task_type?)
```

Worth having, because ms-swift only rejects an unknown argument *after* loading
the model.

**Multi-GPU:** single-process unless `--num-processes` is greater than 1.
distill-run passes the requested count to ms-swift, which launches torchrun. A
container that can see six GPUs will not silently become a distributed job.

```bash
distill-run --engine swift \
  --num-processes 4 \
  --student /models/Qwen3-0.6B \
  --dataset /output/generate-x/sft.jsonl \
  --learning-rate 1e-4 --lora-rank 8 --max-length 2048 \
  --output /output/sft-4gpu
```

**Resume:** `--resume` picks the highest-numbered `checkpoint-N`, looking both
directly under `--output` and one level down, since ms-swift creates its own run
directory.

---

## `easydistill-swift` — both stages in one run

Seed → teacher API → `sft.jsonl` → student checkpoint. Exists because there is no
orchestration layer locally to chain the two stages.

EasyDistill-Swift accepts the union of EasyDistill generation and Swift training
parameters. `--output` is a directory:

```text
<output>/sft.jsonl      generated SFT data, kept so it can be inspected or reused
<output>/checkpoint/    the student checkpoint
```

An optional config holds one section per stage, each identical to the standalone
config:

```yaml
easydistill:
  job_type: instruct_distill
  backend: {...}
  generation: {...}
swift:
  tuner_type: lora
  max_steps: 3
```

**Preflight validates both stages**, so a wrong student path fails before any
tokens are generated instead of after.

**`--resume`** reuses an existing non-empty `sft.jsonl` and skips straight to
training, which makes iterating on hyper-parameters cheap. Ctrl-C between the
stages exits 130 without starting training.

---

## `trl` — white box

Teacher and student weights are loaded directly; the student learns from the
teacher's logits. The student remains a complete Qwen model—LoRA only controls
which student parameters are updated.

`--teacher` is a **weights path**, never a URL. `--output` is a directory; the
final adapter lands in `<output>/final`. Model, LoRA, distillation and distributed
parameters are available as first-level CLI flags.

In an optional config, keys under `trl:` go to `DistillationConfig`, and the LoRA
section is separate. Names are checked against the installed
`DistillationConfig`.

```yaml
trl:
  max_steps: 100             # required: generation dataloader is iterable
  num_train_epochs: 1
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 1.0e-5
  beta: 0.5                 # distillation loss weight
  temperature: 1.0
  max_completion_length: 512
  bf16: true
  gradient_checkpointing: true
lora:
  enabled: true
  r: 8
  lora_alpha: 32
model_init:                 # passed to the student's from_pretrained
  torch_dtype: bfloat16
teacher_model_init:
  torch_dtype: bfloat16
```

Do not use `device_map: auto` for ordinary training: it is an inference-oriented
model split and can leave Trainer inputs on a different GPU from embedding
weights. The default lets Trainer place both models on its current device.
Configure FSDP or DeepSpeed explicitly when one device cannot hold both models.

Plain TRL runs remain single-process unless `--num-processes` is greater than 1.
For multi-GPU runs, distill-run relaunches itself under Accelerate using
`--distributed-strategy fsdp2`, `zero1`, `zero2`, or `zero3`.
`--accelerate-config` can supply a custom Accelerate YAML. ZeRO-1 shards
optimizer state, ZeRO-2 additionally shards gradients, and ZeRO-3 additionally
shards model parameters; use ZeRO-3 when either model does not fit on one GPU.

The local `/models/Qwen3.8-27B` checkpoint is a Qwen3.5 model with a 248,320
token vocabulary. Qwen3-0.6B has a 151,936 token vocabulary and therefore cannot
be used for full-logit distillation. Use Qwen3.5-0.8B as the student.

FSDP2 is available for models that can be wrapped without a full-device
initialization peak. The 27B checkpoint exceeds one 48 GiB card during that
transition, so the formal path uses ZeRO-3, which partitions parameters while
loading:

FSDP2 also cannot shard a checkpoint whose input embeddings and LM head share
the same parameter across separate FSDP groups. Preflight reads `config.json`
and rejects `tie_word_embeddings=true` before loading weights. Use ZeRO-3 or an
untied checkpoint; distill-run does not silently rewrite model weights.

```bash
distill-run --engine trl \
  --num-processes 6 \
  --distributed-strategy zero3 \
  --teacher /models/Qwen3.8-27B \
  --student /work/downloads/Qwen3.5-0.8B \
  --dataset examples/seed_instructions_6rank.jsonl \
  --max-steps 100 --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 --learning-rate 1e-5 \
  --max-completion-length 256 --temperature 1.0 --beta 0.5 \
  --lora-rank 8 --lora-alpha 32 \
  --output /output/trl-zero3
```

This launches one rank on each logical training GPU and shards both the
trainable 0.8B LoRA student and the frozen 27B teacher. `device_map` must be
absent because the distributed runtime owns placement. The teacher and student
must share a vocabulary, which preflight checks before loading either model.

**Data format:** JSONL with `prompt`, `instruction`, `query` or a `messages`
list. Rows are converted to chat form; only prompts are needed, since the teacher
supplies the targets.

**Memory:** both models are resident at once. Start from
`configs/smoke/trl_distill.yaml` and raise the batch size only once it survives a
few steps. An OOM exits 9, distinctly from a generic engine failure.

**Ctrl-C** stops at the next step boundary and saves, rather than leaving a
half-written adapter.

---

## `noop`

No frameworks, no GPU. Used by the test suite and `scripts/smoke-cli.sh` to
exercise argument handling, exit codes and cancellation.

```yaml
steps: 3
step_delay_sec: 0
fail_at_step: 2    # optional, for testing the failure path
```
