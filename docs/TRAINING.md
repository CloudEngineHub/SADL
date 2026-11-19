# Training guide

## Requirements

- One machine with 4 GPUs. Each optimizer step uses 64 windows per GPU (global batch 256 windows of 8 frames).
  Peak memory is about 2.1 GB per GPU with the fast kernels and about 2.6 GB without them.
- The four HDF5 datasets and `datasets.json` (see the main README):

```bash
python tools/prepare_datasets.py --root /path/to/datasets --output datasets.json
```

## Cells

A cell is `<MODEL>_<TASKS>_S<seed>`:

| Part | Values |
|---|---|
| `<MODEL>` | `INTACT` (latent prediction + goal-conditioned action) or `LEWM` (latent prediction only) |
| `<TASKS>` | `4T` (PushT, Cube, Reacher and TwoRoom together) or one of `PUSHT`, `CUBE`, `REACHER`, `TWOROOM` |
| `<seed>` | `3072`, `3073`, `3074` |

All cells are listed in `src/matrix.json` (regenerate with `python src/protocol.py --write src/matrix.json`).

## Four-task training

```bash
python train.py --cell INTACT_4T_S3072 --gpus 0,1,2,3 --datasets datasets.json
python train.py --cell LEWM_4T_S3072 --gpus 0,1,2,3 --datasets datasets.json
```

One epoch passes the largest task once and repeats the smaller tasks to the same length; every global batch holds
64 windows of each task, shuffled before it is split across GPUs. One epoch is 22,782 optimizer steps.

## Single-task training

```bash
python train.py --cell INTACT_PUSHT_S3072 --gpus 0,1,2,3 --datasets datasets.json
python train.py --cell LEWM_REACHER_S3072 --gpus 0,1,2,3 --datasets datasets.json
```

One epoch passes the task's training split once: 5,654 steps for PushT, 5,696 for Cube and Reacher, 1,867 for
TwoRoom.

## Hyperparameters (`src/protocol.py`)

| Setting | Value |
|---|---|
| Encoder | ViT-tiny, patch 14, 224 x 224 images, CLS token, 192-d |
| Projector / prediction projector | 192-2048-192 MLP with BatchNorm |
| Predictor and actor | shared 2-layer Transformer core (4 heads), MLP actor head (INTACT) |
| History | 3 frames; frameskip 5; 8 frames per window |
| Regularizers | SIGReg 0.015, SADL 0.01 |
| Optimizer | AdamW, lr 5e-4, weight decay 1e-3, gradient clipping 1.0 |
| Schedule | cosine decay to 0 with 1% linear warmup, 1 epoch |
| Precision | BF16 autocast, FP32 weights and optimizer state |
| Split | 90% of windows for training, per task, fixed by the seed |
| Data loader | `workers` = 5 per GPU, prefetch 4 |

## Outputs

`runs/<cell>/` contains

- `contract.json`: the exact configuration of the run;
- `metrics.jsonl`: loss terms, SIGReg and SADL values, gradient norm and throughput every 50 steps;
- `last.ckpt`: full training state, written every 1,000 steps;
- `exports/<task>/`: `weights_epoch_1.pt`, `config.yaml`, `action_stats.json` for evaluation;
- `COMPLETE.json`: written when the epoch finishes.

Re-running the same `train.py` command resumes from `last.ckpt`.

## Evaluation after training

```bash
python evaluate.py --checkpoint runs/INTACT_4T_S3072/exports/pusht --task pusht --data-dir /path/to/datasets --planner direct
python evaluate.py --checkpoint runs/LEWM_REACHER_S3072/exports/reacher --task reacher --data-dir /path/to/datasets --planner cem
```

A four-task cell has one export per task; evaluate each task with its own folder.

## Options

- `requirements-fast.txt` installs FlashAttention 4 and CuTe kernels used for the encoder on Hopper (SM90) and
  Blackwell (SM120) GPUs. Without them, or on other GPUs, the same model trains with standard PyTorch operators;
  `runs/<cell>/effective_model.json` records which path ran.
- On single-node RTX 5090 or 5090 D machines that hit `illegal memory access` at the first NCCL collective, launch training with `NCCL_SAFE=1`:

  ```bash
  NCCL_SAFE=1 python train.py --cell INTACT_4T_S3072 --gpus 0,1,2,3 --datasets datasets.json
  ```

  This selects the Socket transport on `lo` with the Ring algorithm and Simple protocol. It disables P2P, InfiniBand, shared-memory transport, and NCCL's cuMem device and host allocations.
- More data-loader workers (`workers` in `src/protocol.py`) help on machines with spare CPU cores.
