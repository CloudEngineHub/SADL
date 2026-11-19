"""Evaluate an exported checkpoint with the LeWM goal-reaching protocol.

    python evaluate.py --checkpoint runs/INTACT_4T_S3072/exports/reacher --task reacher \
        --data-dir /path/to/datasets --planner direct

Protocol: start states and goals from dataset episodes (goal 25 steps ahead), at most 50 environment actions,
planning horizon 5 with replanning every 5 actions, solver batch size 1, FP32 model, PyTorch default numerics.
INTACT checkpoints use --planner direct or guarded_a; LeWM checkpoints use --planner cem. INTACT models receive the
causal history of the previous 5 primitive actions. Reported results use seeds 0, 1 and 42 with 100 episodes each on
an NVIDIA RTX 4090.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / 'src' / 'model'
sys.path.insert(0, str(ROOT / 'eval'))
import planners  # noqa: E402

DATASETS = dict(pusht=('pusht_expert_train.h5', 'pusht_expert_train.h5'),
                cube=('cube_single_expert.h5', 'ogbench/cube_single_expert.h5'),
                reacher=('reacher.h5', 'dmc/reacher_random.h5'),
                tworoom=('tworoom.h5', 'tworoom.h5'))
PLANNERS = dict(direct='direct', guarded_a='guarded_a', cem='cem')


def run_seed(a, seed, work):
    ckpt = a.checkpoint.resolve()
    cache = work / 'cache'
    policy = cache / 'checkpoints' / 'model'
    policy.mkdir(parents=True, exist_ok=True)
    for f in ckpt.iterdir():
        target = policy / f.name
        if not target.exists():
            target.symlink_to(f)
    data = cache / 'datasets'
    for task, (file, name) in DATASETS.items():
        link = data / name
        source = a.data_dir.resolve() / file
        if source.exists() and not link.exists():
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(source)
    intact = a.planner != 'cem'
    spec = dict(history=dict(n_blocks=1, block_size=5) if intact else None, no_history_placeholder='nan',
                action_scaler=None, source_roots=[str(MODEL)])
    (work / 'spec.json').write_text(json.dumps(spec))
    env = {k: v for k, v in os.environ.items() if not k.startswith(('INVERSE_', 'INTACT_'))}
    env.update(CUDA_VISIBLE_DEVICES=str(a.gpu), MUJOCO_EGL_DEVICE_ID=str(a.gpu), MUJOCO_GL='egl',
               PYOPENGL_PLATFORM='egl', STABLEWM_HOME=str(cache), WANDB_MODE='disabled', PYTHONUNBUFFERED='1',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', EVAL_SPEC=str(work / 'spec.json'),
               PYTHONPATH=str(MODEL))
    if intact:
        env.update(INVERSE_QUERY_CONDITION_SOURCE='goal', INVERSE_DIRECT_TARGET_MODE='query')
    planner = PLANNERS[a.planner]
    command = [sys.executable, str(ROOT / 'eval' / 'protocol.py'), f'--config-path={MODEL / "config" / "eval"}',
               f'--config-name={a.task}', 'policy=model/weights_epoch_1.pt', *planners.hydra_overrides(planner),
               'plan_config.receding_horizon=5', f'eval.num_eval={a.episodes}', f'seed={seed}',
               f'cache_dir={cache}', f'output.filename={work / "evaluation.txt"}', '++eval.save_video=false',
               f'hydra.run.dir={work / "hydra"}']
    with open(work / 'evaluation.log', 'w') as log:
        code = subprocess.run(command, cwd=MODEL, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    sidecar = work / 'evaluation.txt.json'
    if code or not sidecar.exists():
        raise SystemExit(f'evaluation failed (seed {seed}); see {work / "evaluation.log"}')
    side = json.loads(sidecar.read_text())
    episodes = side.get('episodes') or []
    successes = sum(bool(e.get('success')) for e in episodes)
    trace = side.get('solver_trace') or {}
    if planner == 'direct' and (trace.get('get_cost_calls') or trace.get('candidate_sequences')):
        raise SystemExit('Direct evaluation must not score candidates')
    return dict(seed=seed, successes=successes, episodes=len(episodes),
                success_rate=100.0 * successes / max(len(episodes), 1),
                gpu=(side.get('process') or {}).get('gpu_name'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True, help='exports/<task> directory of a trained cell')
    p.add_argument('--task', choices=list(DATASETS), required=True)
    p.add_argument('--data-dir', type=Path, required=True, help='directory holding the four HDF5 files')
    p.add_argument('--planner', choices=list(PLANNERS), required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 42])
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--gpu', default='0')
    p.add_argument('--output', type=Path, help='result JSON (default: <checkpoint>/eval_<planner>.json)')
    a = p.parse_args()
    out = (a.output or a.checkpoint / f'eval_{a.planner}.json').resolve()
    rows = []
    for seed in a.seeds:
        work = out.parent / f'{out.stem}_seed{seed}'
        work.mkdir(parents=True, exist_ok=True)
        rows.append(run_seed(a, seed, work))
        print(json.dumps(rows[-1]), flush=True)
    total = sum(r['successes'] for r in rows)
    episodes = sum(r['episodes'] for r in rows)
    result = dict(checkpoint=str(a.checkpoint.resolve()), task=a.task, planner=a.planner, seeds=rows,
                  success_rate=100.0 * total / max(episodes, 1))
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(f'{a.task} {a.planner}: {result["success_rate"]:.2f}% over {episodes} episodes')


if __name__ == '__main__':
    main()
