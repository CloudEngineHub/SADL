"""Train one cell. Each cell runs on 4 GPUs with 64 windows per GPU (global batch 256).

    python train.py --cell INTACT_4T_S3072 --gpus 0,1,2,3 --datasets datasets.json

Cells are listed in src/matrix.json; see docs/TRAINING.md. Outputs go to runs/<cell>/ (contract.json,
metrics.jsonl, last.ckpt, exports/<task>/). Re-running the same command resumes from last.ckpt.
Set NCCL_SAFE=1 to run NCCL over loopback sockets on machines where its P2P / shared-memory transport fails.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell', required=True)
    parser.add_argument('--gpus', required=True, help='four comma-separated GPU indices')
    parser.add_argument('--datasets', type=Path, required=True)
    parser.add_argument('--runs', type=Path, default=ROOT / 'runs')
    args = parser.parse_args()
    if len(args.gpus.split(',')) != 4:
        parser.error('--gpus needs four GPU indices')
    run = args.runs.resolve() / args.cell
    if (run / 'COMPLETE.json').exists():
        print(f'{args.cell} is already complete: {run}')
        return
    cache = ROOT / '.cache'
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpus, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4',
               INTACT_HDF5_STRIDED_READ='1', INTACT_SKIP_UNUSED_PATCH_MEAN='1',
               TORCHINDUCTOR_CACHE_DIR=str(cache / 'inductor'), TRITON_CACHE_DIR=str(cache / 'triton'),
               MPLCONFIGDIR=str(cache / 'matplotlib'))
    if os.environ.get('NCCL_SAFE') == '1':
        env.update(NCCL_P2P_DISABLE='1', NCCL_IB_DISABLE='1', NCCL_SHM_DISABLE='1', NCCL_NET='Socket',
                   NCCL_ALGO='Ring', NCCL_PROTO='Simple', NCCL_SOCKET_IFNAME='lo',
                   NCCL_CUMEM_ENABLE='0', NCCL_CUMEM_HOST_ENABLE='0')
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
               str(ROOT / 'src' / 'train_mixed.py'), '--cell', args.cell,
               '--datasets', str(args.datasets.resolve()), '--output', str(run)]
    if (run / 'last.ckpt').exists():
        command.append('--resume')
    elif (run / 'contract.json').exists():
        command.append('--restart-unstarted')
    sys.exit(subprocess.call(command, env=env))


if __name__ == '__main__':
    main()
