# InferBench Modal CUDA smoke benchmark

`gpu_smoke.py` runs a one-shot PyTorch CUDA matrix-multiplication benchmark
on a Modal NVIDIA GPU. It creates no endpoint, schedule, volume, secret, or
persistent deployment.

## Setup

Install project dependencies and authenticate the Modal CLI:

```bash
make setup
.venv/bin/modal setup
```

Modal 1.5.5 also reports `modal token new` as the direct command for
creating credentials from an authenticated browser session. Credentials
remain in Modal's normal local configuration; never add them to this
repository.

## Run

Defaults: A10G, 2048×2048 matrices, 50 measured iterations, 10 warmups,
float16, seed 42.

```bash
.venv/bin/modal run infra/modal/gpu_smoke.py
```

Low-cost smoke configuration:

```bash
.venv/bin/modal run infra/modal/gpu_smoke.py \
  --gpu A10G \
  --matrix-size 1024 \
  --iterations 20 \
  --warmup 5 \
  --dtype float16 \
  --output /tmp/inferbench_cuda_smoke.json
```

`--gpu` accepts `A10G` or `A100`; `--dtype` accepts `float32`, `float16`,
or `bfloat16`. Unsupported dtypes and unavailable bfloat16 support fail
instead of silently substituting another dtype.

See [`../../docs/cuda-smoke.md`](../../docs/cuda-smoke.md) for timing,
correctness, memory, result-mapping, and limitation details.
