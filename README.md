# InferBench

GPU inference optimization, benchmarking, and scheduling platform.

InferBench is a production-style engineering project for benchmarking LLM
inference across GPU configurations, batching strategies, prefix caching,
quantization, and custom GPU kernels. It aims to grow into a full platform
with a Go API, a benchmark worker, a web UI, a shared Python benchmarking
core, and infrastructure for running workloads both locally and on remote
GPU providers.

## Phase 0 status (current)

This repository is currently in **Phase 0: repository foundation only**.

There is **no** inference, CUDA, Triton, vLLM, database, frontend, API, or
GPU functionality yet. Phase 0 exists purely to establish:

- a clean monorepo layout,
- working Python and Go toolchains with tests, linting, and type checking,
- a Makefile and CI workflow that actually pass,

so that later phases can build on a verified foundation instead of fixing
tooling issues while also building features.

## Repository structure

```
apps/
  api-go/            Go module for the future HTTP API (no framework yet)
  benchmark-worker/  Future benchmark execution worker
  web/               Future frontend

packages/
  benchmark-core/    Shared Python benchmarking library (importable, empty logic)
  proto/             Future shared protocol / schema definitions

infra/
  docker/            Future Dockerfiles / Compose files
  modal/             Future Modal (remote GPU) deployment config
  prometheus/        Future metrics/monitoring config
  kubernetes/        Future k8s manifests

benchmarks/
  workloads/         Future benchmark workload definitions
  results/           Generated benchmark results (git-ignored contents)

tests/               Root-level Python tests
scripts/             Future automation/dev scripts
docs/                Project documentation
```

## Local setup

Requirements:

- Python **3.12** (exactly `python3.12`; the system Python is left untouched)
- Go **1.27**
- Docker Desktop + Docker Compose (for future phases)

```bash
make setup
```

This creates a local `.venv` (via `python3.12 -m venv .venv`), installs the
`benchmark-core` package plus dev dependencies (pytest, ruff, mypy) into it,
and runs `go mod tidy` for the `apps/api-go` module.

## Test / lint / format commands

```bash
make test      # runs pytest (Python) and go test ./... (Go)
make lint      # runs ruff check + mypy (Python) and gofmt/go vet (Go)
make format    # runs ruff format + ruff check --fix (Python) and gofmt -w (Go)
```

## Python version policy

This project targets **Python 3.12 only**. It does not depend on, and is not
tested against, Python 3.14. The system Python installation is never
modified — all Python tooling runs inside the project-local `.venv`.

## Apple Silicon local development

Local development happens on Apple Silicon (macOS/arm64). Phase 0 has no
GPU or CUDA dependencies, so everything in this repository runs natively on
Apple Silicon today.

## GPU work is remote

InferBench targets NVIDIA GPUs for inference benchmarking. Apple Silicon has
no NVIDIA GPU, so all actual GPU/CUDA/Triton/vLLM work in later phases will
run on remote infrastructure (e.g. cloud GPU providers or Modal), not on the
local development machine.

## Generate a synthetic workload (Phase 3)

```bash
python -m benchmark_core generate-workload \
  --profile shared_prefix \
  --requests 100 \
  --seed 42 \
  --output benchmarks/workloads/shared_prefix.json
```

Deterministic, local, no network access, no model execution. See
[`docs/workloads.md`](docs/workloads.md) for the full methodology.

## Run local inference (Phase 4)

```bash
python -m benchmark_core run-local \
  --model sshleifer/tiny-gpt2 \
  --workload benchmarks/workloads/short_prompt_short_output.json \
  --warmup 1 \
  --output benchmarks/results/local_transformers_smoke.json
```

Runs a small causal LM locally on Apple MPS (falling back to CPU) via
PyTorch + `transformers` -- no CUDA, no remote GPU. See
[`docs/local-inference.md`](docs/local-inference.md) for the full
methodology.

## Run the remote CUDA smoke benchmark (Phase 5)

Authenticate once, then launch a pay-per-run Modal A10G job:

```bash
.venv/bin/modal setup
.venv/bin/modal run infra/modal/gpu_smoke.py
```

This is a one-shot PyTorch CUDA matrix-multiplication benchmark, not a
persistent service. See [`docs/cuda-smoke.md`](docs/cuda-smoke.md) for GPU,
dtype, timing, correctness, memory, and low-cost smoke options.
