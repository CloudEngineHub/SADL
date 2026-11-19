"""Planner specifications (solver batch size 1) and their Hydra overrides."""
from __future__ import annotations

import copy

BATCH_SIZE = 1


def _direct():
    return {"kind": "direct", "config": "cem", "target": "prior_only_solver.PriorOnlySolver",
            "batch_size": BATCH_SIZE, "actor_warmstart": True, "extra": {}}


def _cem(k, i, topk, std, warm, target=None, config="timed_cem", extra=None):
    return {"kind": "search", "config": config, "target": target, "batch_size": BATCH_SIZE,
            "num_samples": k, "n_steps": i, "topk": topk, "var_scale": std,
            "actor_warmstart": bool(warm), "extra": dict(extra or {})}


def _b(v):
    return "true" if v else "false"


def _guarded(k, i, topk, alpha, floor, cap, actor_cov, trust):
    return _cem(k, i, topk, 0.25, True, "abcd_solvers.GuardedCEMSolver", extra={
        "update_alpha": f"{alpha}", "std_floor": f"{floor}", "std_cap": f"{cap}",
        "actor_covariance": _b(actor_cov), "actor_cov_target_rms": "0.25",
        "actor_cov_min": "0.05", "actor_cov_max": "0.5", "trust_lambda": f"{trust}"})




PLANNERS = {
    "direct": _direct(),
    "guarded_a": _guarded(128, 3, 16, 1.0, 0.0001, 10.0, False, 0.0),
    "cem": _cem(300, 30, 30, 1.0, False),
}


def get(planner) -> dict:
    """Planner spec from a name or an inline dict (validated)."""
    p = copy.deepcopy(PLANNERS[planner] if isinstance(planner, str) else planner)
    if p.get("batch_size", BATCH_SIZE) != BATCH_SIZE:
        raise ValueError(f"solver batch size must be {BATCH_SIZE}")
    p["batch_size"] = BATCH_SIZE
    if p["kind"] not in ("direct", "search"):
        raise ValueError(f"unknown planner kind {p['kind']}")
    p.setdefault("extra", {})
    if isinstance(planner, str):
        p["name"] = planner
    return p


def hydra_overrides(planner) -> list[str]:
    """Hydra overrides for a stable_worldmodel/LeWM-style eval config."""
    p = get(planner)
    if p["kind"] == "direct":
        return [f"solver={p['config']}", f"solver._target_={p['target']}", f"solver.batch_size={p['batch_size']}",
                f"++eval.actor_warmstart={_b(p['actor_warmstart'])}"]
    a = [f"solver={p['config']}", f"solver.batch_size={p['batch_size']}", f"solver.num_samples={p['num_samples']}",
         f"solver.n_steps={p['n_steps']}", f"solver.topk={p['topk']}", f"solver.var_scale={p['var_scale']}",
         f"++eval.actor_warmstart={_b(p['actor_warmstart'])}"]
    if p.get("target"):
        a.insert(1, f"solver._target_={p['target']}")
    return a + [f"++solver.{k}={v}" for k, v in p["extra"].items()]


def _typed(v):
    if isinstance(v, str):
        if v in ("true", "false"):
            return v == "true"
        for cast in (int, float):
            try:
                return cast(v)
            except ValueError:
                pass
    return v


def solver_fields(planner) -> dict:
    """Typed solver fields (what the resolved solver config must contain)."""
    p = get(planner)
    out = {"batch_size": p["batch_size"]}
    if p.get("target"):
        out["_target_"] = p["target"]
    if p["kind"] == "search":
        out.update(num_samples=p["num_samples"], n_steps=p["n_steps"], topk=p["topk"], var_scale=p["var_scale"])
    out.update({k: _typed(v) for k, v in p["extra"].items()})
    return out


def expected_solver_mismatches(planner, resolved_solver_cfg: dict, actor_warmstart) -> list[str]:
    """Compare a resolved solver config (dict) with the spec; return readable mismatches."""
    p = get(planner)
    bad = []
    for k, v in solver_fields(p).items():
        got = resolved_solver_cfg.get(k)
        if isinstance(v, float) or isinstance(got, float):
            try:
                ok = got is not None and abs(float(got) - float(v)) <= 1e-12
            except (TypeError, ValueError):
                ok = False
        else:
            ok = got == v
        if not ok:
            bad.append(f"solver.{k}={got!r} expected {v!r}")
    if actor_warmstart is not None and bool(actor_warmstart) != bool(p["actor_warmstart"]):
        bad.append(f"actor_warmstart={actor_warmstart} expected {p['actor_warmstart']}")
    return bad
