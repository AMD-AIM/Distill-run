# Engines

Each engine owns one framework call. Production runs can be configured entirely
with first-level CLI flags; `--config`, `--config-json`, and `--set` remain
available for advanced framework fields.

See [`cli-parameters.md`](cli-parameters.md) for all 69 parameters, defaults, and
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

`--teacher` and `--student` accept a local weights path, `hf://org/model`, or
`ms://org/model`. Hub models are downloaded into `<work-dir>/models` before
preflight. `--output` is a directory; the final adapter lands in
`<output>/final`. Model, LoRA, distillation and distributed parameters are
available as first-level CLI flags.

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

## `yolo` — object detection training

A built-in PyTorch implementation of YOLOv5u, YOLOv8, YOLO11 and YOLO26 (scales
`n s m l x`) in `src/distill_run/yolo/`. It needs torch, torchvision, OpenCV and
safetensors, and does not use the Ultralytics package. YOLO26 trains its
end-to-end one-to-one head next to the one-to-many head, and inference takes the
top-k without NMS.

```bash
distill-run --engine yolo --model yolo26n.pt --data https://example.com/coco8.zip \
  --epochs 100 --imgsz 640 --batch -1 --device 0 --output /output/yolo26n-exp
```

**`--model`**

| value | meaning |
|---|---|
| `yolo26n`, `yolo11s`, `yolov8m`, `yolov5lu` | architecture, random init (`--pretrained` loads official weights) |
| `yolo26n.pt` | official weights, downloaded once to `<model-cache-dir>/yolo` (`YOLO_WEIGHTS_URL` sets a mirror) |
| `/path/w.pt` | an official checkpoint or one written by this engine |
| `/path/run-dir` | an earlier `--output` (`config.json` + `best.safetensors`) |
| `http(s)://…/w.pt`, `hf://org/repo/w.pt`, `ms://org/repo/w.pt` | downloaded weights |

Official `.pt` files are read with a restricted unpickler: `ultralytics.*`
classes become inert placeholders and only tensors are kept. If the dataset has
a different class count, the class-dependent head layers are reinitialised.

**`--data`**: a local path, `http(s)://`, `hf://` or `ms://`. The value can be a
directory, a `data.yaml` or a zip/tar archive, which is unpacked once next to the
cached download. The following layouts are detected:

- a YOLO txt `data.yaml` (`path`, `train`, `val`, `names`), or `images/<split>` with
  `labels/<split>`. Names come from `classes.txt`, or default to COCO-80 when the
  directory name starts with `coco`.
- COCO json: `annotations/instances_<split>.json` next to `images/<split>`, or
  `data.yaml` split entries that point at `.json` files.
- Ultralytics Platform NDJSON (`*.ndjson`, for example a signed export URL). The
  first line holds `task` and `class_names`; each further line holds one image's
  `file`, `split`, `url` and `annotations.boxes` (`[cls, cx, cy, w, h]`). Images are
  downloaded with 32 threads into `<dataset-cache-dir>/ndjson/<name>-<hash>/`.
  The hash ignores URL query strings, so a re-signed export of the same data reuses
  the earlier download. Without a `val` split, a seeded 10% of `train` becomes
  `val`, as on the Platform. Only `task: detect` is accepted. If any image fails to
  download, the run fails with exit code 5.

If there is no `val` split, the engine validates on the training images. Set
`yolo.val_fraction` to hold out a fraction of them instead.

**Flags**: `--epochs`, `--imgsz`, `--batch` (`-1` sizes the batch to about 60% of
GPU memory), `--workers`, `--pretrained`, `--amp`/`--no-amp`, `--optimizer`
(`auto`, `SGD`, `Adam`, `AdamW`), `--close-mosaic`, `--learning-rate`,
`--max-steps` (stops after N batches, useful for smoke tests), `--logging-steps`,
`--resume` and `--device`. The engine sets `MIOPEN_FIND_MODE=FAST` unless it
is already set (only ROCm reads it): MIOpen's default mode spends about 30 s
tuning each new convolution shape. Every other hyper-parameter (loss gains, warmup,
augmentation) is under `yolo.*`; see `distill-run --engine yolo
--show-default-config`.

**Outputs** (`--output`): `weights/last.pt` holds the resumable state;
`weights/best.pt` holds the EMA weights with the best fitness
(`0.1·mAP50 + 0.9·mAP50-95`). The run also writes `best.safetensors` and
`config.json` for reloading, and `summary.json` (final metrics, `best_epoch`,
peak memory).

Training curves are written once per epoch:

- `results.csv` uses the Ultralytics column names (`epoch`, `time`,
  `train/{box,cls,dfl}_loss`, `metrics/{precision,recall,mAP50,mAP50-95}(B)`,
  `val/{box,cls,dfl}_loss`, `lr/pg0..2`), so the Ultralytics Platform and its
  plotting tools read it unchanged. A cell is empty when an epoch was not
  validated.
- `metrics.jsonl` holds the same values as one JSON object per epoch, plus
  `step`, `fitness`, `best_epoch` and `total_epochs`. Read it to follow a run
  while it trains.
- `results.png` plots the ten loss and metric curves when matplotlib is
  installed.

`--resume` drops the rows it is about to train again. Ctrl-C stops at the next
batch and still writes all of these files.

---

## `noop`

No frameworks, no GPU. Used by the test suite and `scripts/smoke-cli.sh` to
exercise argument handling, exit codes and cancellation.

```yaml
steps: 3
step_delay_sec: 0
fail_at_step: 2    # optional, for testing the failure path
```
