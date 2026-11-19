"""LeWM goal-reaching evaluation: stable_worldmodel World.evaluate on dataset starts, goal 25 steps ahead,
budget 50 actions. The model code directory comes from --config-path and PYTHONPATH; the action history follows
the model's history spec (action_history.py); no numeric backend flags are forced. Launched by evaluate.py,
which writes the spec JSON named by $EVAL_SPEC and reads the sidecar <output.filename>.json.
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
if "MUJOCO_EGL_DEVICE_ID" not in os.environ:
    visible_device = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",", 1)[0].strip()
    if visible_device.isdigit():
        os.environ["MUJOCO_EGL_DEVICE_ID"] = visible_device

import importlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.append(str(HERE))
import action_history as ch  # noqa: E402
import checks as val  # noqa: E402

SPEC = json.loads(Path(os.environ["EVAL_SPEC"]).read_text()) if os.environ.get("EVAL_SPEC") else {}


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, Path):
        return str(value)
    return value


def img_transform(cfg):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    return swm.data.HDF5Dataset(dataset_name, keys_to_cache=cfg.dataset.keys_to_cache, cache_dir=dataset_path)


def _load_attr(ref):
    """'module:attr' resolved on sys.path (the model code directory)."""
    mod, attr = ref.split(":", 1)
    return getattr(importlib.import_module(mod), attr)


def _action_scaler(col_data):
    ref = SPEC.get("action_scaler")
    if ref:
        return _load_attr(ref).fit(col_data)
    return preprocessing.StandardScaler().fit(col_data[~np.isnan(col_data).any(axis=1)])


def _trace_module():
    path = SPEC.get("trace_hooks")
    if not path:
        return None
    spec = importlib.util.spec_from_file_location("trace_hooks", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@hydra.main(version_base=None, config_path=None, config_name="pusht")
def run(cfg: DictConfig):
    assert cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget, \
        "Planning horizon must be smaller than or equal to eval_budget"

    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))
    transform = {"pixels": img_transform(cfg), "goal": img_transform(cfg)}

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    episode_manifest_path = OmegaConf.select(cfg, "eval.episode_manifest", default=None)
    split_manifest = None
    stats_episode_ids = None
    if episode_manifest_path:
        from utils import load_episode_split_manifest  # model code directory
        split_manifest = load_episode_split_manifest(episode_manifest_path, dataset=dataset)
        stats_episode_ids = split_manifest["split_arrays"]["train"]
    stats_dataset = dataset
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        col_data = stats_dataset.get_col_data(col)
        if stats_episode_ids is not None:
            from utils import select_episode_rows
            col_data = select_episode_rows(stats_dataset, col_data, stats_episode_ids)
        if col == "action":
            processor = _action_scaler(col_data)
        else:
            processor = preprocessing.StandardScaler()
            processor.fit(col_data[~np.isnan(col_data).any(axis=1)])
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = process[col]

    policy = cfg.get("policy", "random")
    solver = None
    model = None
    counter = None
    trace = _trace_module()
    if policy != "random":
        model = swm.wm.utils.load_pretrained(cfg.policy)
        model = model.to("cuda")
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        actor_warmstart = bool(OmegaConf.select(cfg, "eval.actor_warmstart", default=True))
        if hasattr(model, "set_actor_warmstart"):
            model.set_actor_warmstart(actor_warmstart)
        elif hasattr(model, "actor_warmstart"):
            model.actor_warmstart = actor_warmstart
        counter = val.CostCallCounter(model)
        if trace is not None and not trace.install_model(model):
            raise RuntimeError("trace hooks requested but could not be installed on the model")
        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        policy = swm.policy.WorldModelPolicy(solver=solver, config=config, process=process, transform=transform)
    else:
        policy = swm.policy.RandomPolicy()

    results_path = (Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
                    if cfg.policy != "random" else Path(__file__).parent)

    # sample the episodes and the starting indices (as in LeWM eval.py)
    g = np.random.default_rng(cfg.seed)
    requested_split = None
    if split_manifest is not None:
        requested_split = str(OmegaConf.select(cfg, "eval.episode_split", default="test"))
        if requested_split not in split_manifest["split_arrays"]:
            raise ValueError(f"Unknown evaluation episode split: {requested_split}")
        candidate_episodes = split_manifest["split_arrays"][requested_split]
        max_start_idx = np.asarray(dataset.lengths)[candidate_episodes] - cfg.eval.goal_offset_steps - 1
        valid = max_start_idx >= 0
        candidate_episodes = candidate_episodes[valid]
        max_start_idx = max_start_idx[valid]
        if candidate_episodes.size < cfg.eval.num_eval:
            raise ValueError(f"Split {requested_split} has only {candidate_episodes.size} "
                             f"valid episodes for {cfg.eval.num_eval} evaluations")
        selected_positions = g.choice(candidate_episodes.size, size=cfg.eval.num_eval, replace=False)
        eval_episodes = candidate_episodes[selected_positions]
        selected_max_starts = max_start_idx[selected_positions]
        eval_start_idx = np.asarray([g.integers(0, int(value) + 1) for value in selected_max_starts],
                                    dtype=np.int64)
        order = np.argsort(eval_episodes)
        eval_episodes = eval_episodes[order]
        eval_start_idx = eval_start_idx[order]
        print("EPISODE_DISJOINT_EVAL=" + json.dumps(
            {"manifest": split_manifest["path"], "split": requested_split, "episodes": eval_episodes.tolist(),
             "start_steps": eval_start_idx.tolist()}, sort_keys=True))
    else:
        episode_len = get_episodes_length(dataset, ep_indices)
        max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
        max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
        max_start_per_row = np.array([max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)])
        valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
        valid_indices = np.nonzero(valid_mask)[0]
        print(valid_mask.sum(), "valid starting points found for evaluation.")
        random_episode_indices = g.choice(len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False)
        random_episode_indices = np.sort(valid_indices[random_episode_indices])
        print(random_episode_indices)
        eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
        eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    # action history: model-specific spec, shared causal implementation
    hist_spec = ch.HistorySpec.from_dict(SPEC.get("history"), default_block_size=int(cfg.plan_config.action_block))
    if hist_spec.uses_history and hist_spec.block_size != int(cfg.plan_config.action_block):
        raise ValueError(f"history block_size {hist_spec.block_size} != plan action_block "
                         f"{cfg.plan_config.action_block} (exact block width required)")
    history_protocol = ch.protocol_record(hist_spec)
    if cfg.policy != "random":
        policy, history_protocol = ch.wrap_policy(
            policy, dataset, eval_episodes.tolist(), eval_start_idx.tolist(), hist_spec,
            placeholder=SPEC.get("no_history_placeholder", "nan"))
        if trace is not None and hist_spec.uses_history:
            trace.install_history(type(policy))
    elif hist_spec.uses_history:
        raise ValueError("a model history spec needs a model policy")
    print("EVAL_ACTION_HISTORY=" + json.dumps(history_protocol, sort_keys=True))

    world.set_policy(policy)
    results_path.mkdir(parents=True, exist_ok=True)

    numerics_start = val.numerics_snapshot()
    start_time = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video=(results_path if bool(OmegaConf.select(cfg, "eval.save_video", default=True)) else None),
    )
    eval_total_time = time.time() - start_time
    numerics_end = val.numerics_snapshot()
    timing = solver.timing_summary() if hasattr(solver, "timing_summary") else {}
    if isinstance(metrics, dict) and timing:
        metrics["solver_timing"] = timing
        metrics["eval_total_time"] = eval_total_time
        metrics["cem_time_per_step"] = timing.get("solve_time_mean", 0.0)
        metrics["cem_time_per_episode"] = timing.get("solve_time_sum", 0.0) / max(int(cfg.eval.num_eval), 1)
        metrics["get_cost_calls"] = timing.get("get_cost_calls_sum", 0.0)
        metrics["rollout_candidates"] = timing.get("candidate_action_steps_sum", 0.0)
        metrics["configured_rollout_candidates_per_solve"] = timing.get("configured_rollout_budget_mean", 0.0)
    print(metrics)

    results_path = results_path / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path = results_path.with_suffix(results_path.suffix + ".json")
    with results_path.open("a") as f:
        f.write("\n==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {eval_total_time} seconds\n")

    successes = [bool(x) for x in np.asarray(metrics.get("episode_successes", []), dtype=bool).tolist()]
    records = policy.episode_records() if hasattr(policy, "episode_records") else []
    for r in records:
        r["success"] = successes[r["env"]] if r["env"] < len(successes) else None
    solver_cfg = OmegaConf.to_container(cfg.solver, resolve=True) if "solver" in cfg else {}
    solver_cfg.pop("model", None)
    roots = [HERE] + [p for p in SPEC.get("source_roots", [])]
    payload = {
        "rule": val.RULE,
        "policy": cfg.policy,
        "actor_warmstart": OmegaConf.select(cfg, "eval.actor_warmstart", default=None),
        "actor_warmstart_effective": getattr(model, "actor_warmstart", None) if model is not None else None,
        "num_eval": cfg.eval.num_eval,
        "num_samples": OmegaConf.select(cfg, "solver.num_samples", default=None),
        "n_steps": OmegaConf.select(cfg, "solver.n_steps", default=None),
        "topk": OmegaConf.select(cfg, "solver.topk", default=None),
        "solver_batch_size": OmegaConf.select(cfg, "solver.batch_size", default=None),
        "solver_resolved": solver_cfg,
        "solver_trace": counter.record() if counter is not None else None,
        "inference_mode": getattr(solver, "evaluation_mode", None),
        "horizon": cfg.plan_config.horizon,
        "receding_horizon": cfg.plan_config.receding_horizon,
        "action_block": cfg.plan_config.action_block,
        "goal_offset_steps": cfg.eval.goal_offset_steps,
        "eval_budget": cfg.eval.eval_budget,
        "seed": cfg.seed,
        "metrics": _json_safe(metrics),
        "eval_total_time": eval_total_time,
        "output_file": str(results_path),
        "episode_split_manifest": split_manifest["path"] if split_manifest is not None else None,
        "episode_split": requested_split,
        "eval_episodes": _json_safe(eval_episodes),
        "eval_start_steps": _json_safe(eval_start_idx),
        "episodes": _json_safe(records),
        "action_history_protocol": history_protocol,
        "history_spec": hist_spec.to_dict(),
        "dataset_file": str(getattr(dataset, "h5_path", "")),
        "numerics_start": numerics_start,
        "numerics_end": numerics_end,
        "process": val.process_snapshot(),
        "imported_sources": val.imported_sources(roots),
        "trace_hooks": trace is not None,
        "eval_spec": SPEC,
    }
    with sidecar_path.open("w") as f:
        json.dump(_json_safe(payload), f, indent=2, sort_keys=True)


if __name__ == "__main__":
    run()
