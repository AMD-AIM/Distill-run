# Independent installation

`distill-run` owns its source dependencies and does not import repositories next
to its checkout.

## Development checkout

```bash
git clone --recursive <repository>
cd distill-run
make dev PROFILE=rocm714
```

The pinned source revisions live in:

- `third_party/trl`
- `third_party/easydistill`

`make dev` initializes missing submodules and delegates environment creation to
`scripts/setup_envs.py`. The script creates a lightweight core environment and
one environment per heavy engine:

```text
.venv/
.venvs/swift/
.venvs/trl/
.venvs/easydistill/
```

The container wrapper uses `.venvs/core` for its core interpreter so a
container-created environment never overwrites a host developer's `.venv`.

The environments use `--system-site-packages` so the ROCm Torch supplied by the
base image is reused. A generated constraint file prevents pip from replacing
Torch or vLLM while resolving engine dependencies.

Install only selected engines with:

```bash
make dev PROFILE=rocm714 ENGINES=trl
```

For an offline installation, build the direct dependency wheelhouse on a
networked machine with the same Python version, then invoke:

```bash
make wheelhouse PROFILE=rocm714
python3 scripts/setup_envs.py --profile rocm714 --offline
```

The wheelhouse intentionally excludes Torch, vLLM, ROCm libraries, and
transitive packages already supplied by the base image.

## Updating source dependencies

Updates are explicit. Never track an upstream branch at runtime.

```bash
git -C third_party/trl fetch origin
git -C third_party/trl checkout <reviewed-commit>
git add third_party/trl
```

The parent repository records the selected commit as its gitlink. Run
`make check` and the multi-GPU smoke suite before accepting an update.

## Runtime selection

The public CLI detects `.venvs/<engine>/bin/python` and relaunches the complete
command once in that environment. This preserves the existing CLI and
Accelerate/torchrun behavior while isolating conflicting TRL and ms-swift
dependency graphs. Set `DISTILL_RUN_ENVS_ROOT` when the environments are mounted
at a different path.

Run diagnostics with:

```bash
distill-run doctor
distill-run doctor --engine trl --json
```
