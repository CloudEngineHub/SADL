"""Guarded CEM ablations for the E goal-displacement checkpoints.

This module is intentionally separate from the completed 396-job planner
matrix.  Every solver keeps the deterministic Direct-E plan in the population,
tracks the best observed sequence across rounds, re-scores the final Gaussian
mean, and returns the best of those *observed* choices under one common cost.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from stable_worldmodel.solver import CEMSolver
from stable_worldmodel.solver.utils import prepare_init_action


def math_sdpa_flags() -> dict[str, bool]:
    return {
        "flash": bool(torch.backends.cuda.flash_sdp_enabled()),
        "mem_efficient": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
        "math": bool(torch.backends.cuda.math_sdp_enabled()),
        "cudnn": bool(
            getattr(torch.backends.cuda, "cudnn_sdp_enabled", lambda: False)()
        ),
    }


def assert_math_only_sdpa() -> None:
    expected = {
        "flash": False,
        "mem_efficient": False,
        "math": True,
        "cudnn": False,
    }
    observed = math_sdpa_flags()
    if observed != expected:
        raise RuntimeError(
            f"ABCD matrix requires MATH-only SDPA: expected {expected}, got {observed}"
        )


def _mixture_marginal_std(stats: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return diagonal marginal std for Gaussian or Gaussian-mixture actors."""

    if stats["kind"] == "gaussian":
        return stats["log_std"].float().exp()
    if stats["kind"] != "mixture_gaussian":
        raise ValueError(f"unsupported actor distribution: {stats['kind']}")
    weights = stats["weights"].float()
    means = stats["means"].float()
    component_var = (2.0 * stats["log_stds"].float()).exp()
    expectation = (weights[..., None] * means).sum(dim=-2)
    second = (weights[..., None] * (component_var + means.square())).sum(dim=-2)
    return (second - expectation.square()).clamp_min(1e-8).sqrt()


@torch.inference_mode()
def actor_sequence_scale(
    model: Any,
    info: dict[str, Any],
    reference: torch.Tensor,
    *,
    target_rms: float,
    minimum: float,
    maximum: float,
) -> torch.Tensor:
    """Build an actor-aware diagonal scale along the deterministic plan.

    The actor's mixture marginal standard deviations supply the anisotropic
    shape.  Their whole-sequence RMS is normalized to ``target_rms`` and then
    clipped.  Rollout follows the deterministic reference actions, so the
    covariance is aligned with exactly the plan used as the CEM center.
    """

    if model.inverse_actor is None:
        raise RuntimeError("actor covariance requires an inverse actor")
    device = reference.device
    model_dtype = next(model.parameters()).dtype
    batch_size, horizon, action_dim = reference.shape

    init = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in info.items()
        if torch.is_tensor(value)
    }
    init["action"] = model._coerce_action_history(
        init.get("action"),
        batch_size=batch_size,
        seq_len=init["pixels"].size(1),
        action_dim=action_dim,
        device=device,
        dtype=model_dtype,
    )
    encoded = model.encode(init)
    emb = encoded["emb"]
    act_hist = encoded["action"]
    history_size = model.predictor.pos_embedding.size(1)

    goal = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in info.items()
        if torch.is_tensor(value)
    }
    goal["pixels"] = goal["goal"]
    for key in list(goal):
        if key.startswith("goal_"):
            goal[key[len("goal_") :]] = goal.pop(key)
    goal.pop("action", None)
    goal_emb = model.encode(goal)["emb"][:, -1]

    scales = []
    for step in range(horizon):
        z = emb[:, -1]
        previous_action_embedding = model.action_encoder(act_hist[:, -1:])[:, -1]
        stats = model.inverse_action_parameters(z, goal_emb, previous_action_embedding)
        step_scale = _mixture_marginal_std(stats)
        if tuple(step_scale.shape) != (batch_size, action_dim):
            raise ValueError(
                f"actor scale shape mismatch: expected {(batch_size, action_dim)}, "
                f"got {tuple(step_scale.shape)}"
            )
        scales.append(step_scale)
        emb, act_hist = model.rollout_one_step(
            emb,
            reference[:, step].to(dtype=model_dtype),
            act_hist,
            history_size=history_size,
        )

    raw = torch.stack(scales, dim=1).float()
    if not torch.isfinite(raw).all() or not (raw > 0).all():
        raise FloatingPointError("actor covariance produced invalid scales")
    rms = raw.square().mean(dim=(-1, -2), keepdim=True).sqrt().clamp_min(1e-8)
    normalized = raw * (float(target_rms) / rms)
    return normalized.clamp(min=float(minimum), max=float(maximum)).to(reference.dtype)


class GuardedCEMSolver(CEMSolver):
    """Factorized raw CEM with Direct-plan preservation and safe selection."""

    def __init__(
        self,
        *args: Any,
        update_alpha: float = 1.0,
        std_floor: float = 1e-4,
        std_cap: float = 10.0,
        actor_covariance: bool = False,
        actor_cov_target_rms: float = 0.25,
        actor_cov_min: float = 0.05,
        actor_cov_max: float = 0.50,
        trust_lambda: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not 0.0 < update_alpha <= 1.0:
            raise ValueError("update_alpha must be in (0, 1]")
        if not 0.0 < std_floor <= std_cap:
            raise ValueError("require 0 < std_floor <= std_cap")
        if actor_covariance and not 0.0 < actor_cov_min <= actor_cov_max:
            raise ValueError("invalid actor covariance clip")
        if trust_lambda < 0.0:
            raise ValueError("trust_lambda must be nonnegative")
        self.update_alpha = float(update_alpha)
        self.std_floor = float(std_floor)
        self.std_cap = float(std_cap)
        self.actor_covariance = bool(actor_covariance)
        self.actor_cov_target_rms = float(actor_cov_target_rms)
        self.actor_cov_min = float(actor_cov_min)
        self.actor_cov_max = float(actor_cov_max)
        self.trust_lambda = float(trust_lambda)
        self.timing_history: list[dict[str, float]] = []

    @staticmethod
    def _slice_info(info: dict[str, Any], start: int, end: int) -> dict[str, Any]:
        return {key: value[start:end] for key, value in info.items()}

    @staticmethod
    def _expand_info(
        info: dict[str, Any], samples: int, device: torch.device, dtype: torch.dtype
    ) -> dict[str, Any]:
        expanded: dict[str, Any] = {}
        for key, value in info.items():
            if torch.is_tensor(value):
                target_dtype = dtype if value.is_floating_point() else None
                value = value.to(device=device, dtype=target_dtype)
                expanded[key] = value.unsqueeze(1).expand(
                    value.shape[0], samples, *value.shape[1:]
                )
            elif isinstance(value, np.ndarray):
                expanded[key] = np.repeat(value[:, None, ...], samples, axis=1)
            else:
                expanded[key] = value
        return expanded

    def _score(
        self,
        info: dict[str, Any],
        candidates: torch.Tensor,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base = self.model.get_cost(
            self._expand_info(info, candidates.shape[1], self.device, self.dtype),
            candidates,
        ).float()
        trust = (candidates.float() - reference[:, None].float()).square().mean(
            dim=(-1, -2)
        )
        total = base + self.trust_lambda * trust
        expected = candidates.shape[:2]
        if tuple(total.shape) != tuple(expected) or not torch.isfinite(total).all():
            raise FloatingPointError(
                f"invalid CEM cost: expected {tuple(expected)}, got {tuple(total.shape)}"
            )
        return total, base, trust

    @staticmethod
    def _gather(candidates: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        batch = torch.arange(candidates.shape[0], device=candidates.device)
        return candidates[batch, index]

    @torch.inference_mode()
    def solve(
        self, info_dict: dict[str, Any], init_action: torch.Tensor | None = None
    ) -> dict[str, Any]:
        started = time.time()
        total_envs = len(next(iter(info_dict.values())))
        reference = prepare_init_action(
            self.model,
            info_dict,
            init_action,
            self.horizon,
            n_envs=total_envs,
            action_dim=self.action_dim,
        ).to(device=self.device, dtype=self.dtype)
        mean = reference.clone()
        if self.actor_covariance:
            scale = actor_sequence_scale(
                self.model,
                info_dict,
                reference,
                target_rms=self.actor_cov_target_rms,
                minimum=self.actor_cov_min,
                maximum=self.actor_cov_max,
            )
        else:
            scale = torch.full_like(reference, float(self.var_scale))

        output_actions = torch.empty_like(reference)
        output_costs = torch.empty(total_envs, device=self.device, dtype=torch.float32)
        stats = defaultdict(float)
        stats["configured_num_samples"] = float(self.num_samples)
        stats["configured_iterations"] = float(self.n_steps)
        stats["configured_topk"] = float(self.topk)
        stats["configured_horizon"] = float(self.horizon)
        stats["configured_initial_std"] = float(self.var_scale)
        stats["configured_update_alpha"] = self.update_alpha
        stats["configured_std_floor"] = self.std_floor
        stats["configured_std_cap"] = self.std_cap
        stats["configured_actor_covariance"] = float(self.actor_covariance)
        stats["configured_actor_cov_target_rms"] = self.actor_cov_target_rms
        stats["configured_actor_cov_min"] = self.actor_cov_min
        stats["configured_actor_cov_max"] = self.actor_cov_max
        stats["configured_trust_lambda"] = self.trust_lambda
        stats["configured_rollout_budget"] = float(
            self.num_samples * self.n_steps * self.horizon
        )
        stats["reference_forced_every_round"] = 1.0
        stats["global_best_preserved"] = 1.0
        stats["final_mean_rescored"] = 1.0
        stats["returns_observed_choice"] = 1.0
        stats["sdpa_math_only"] = float(
            math_sdpa_flags() == {"flash": False, "mem_efficient": False, "math": True, "cudnn": False}
        )
        stats["initial_scale_rms"] = float(
            scale.float().square().mean().sqrt().item()
        )

        for start in range(0, total_envs, self.batch_size):
            end = min(start + self.batch_size, total_envs)
            batch_size = end - start
            info = self._slice_info(info_dict, start, end)
            batch_reference = reference[start:end]
            batch_mean = mean[start:end]
            batch_scale = scale[start:end]
            global_best = batch_reference.clone()
            global_best_cost = torch.full(
                (batch_size,), float("inf"), device=self.device
            )
            reference_cost = None

            for _round in range(self.n_steps):
                candidates = torch.randn(
                    batch_size,
                    self.num_samples,
                    self.horizon,
                    self.action_dim,
                    generator=self.torch_gen,
                    device=self.device,
                    dtype=self.dtype,
                )
                candidates = candidates * batch_scale[:, None] + batch_mean[:, None]
                candidates[:, 0] = batch_reference
                if self.num_samples > 1:
                    candidates[:, 1] = global_best

                score_started = time.time()
                cost, base, trust = self._score(info, candidates, batch_reference)
                stats["get_cost_time"] += time.time() - score_started
                stats["get_cost_calls"] += 1.0
                stats["candidate_action_sequences"] += float(
                    batch_size * self.num_samples
                )
                stats["candidate_action_steps"] += float(
                    batch_size * self.num_samples * self.horizon
                )
                stats["base_cost_sum"] += float(base.mean().item())
                stats["trust_cost_sum"] += float(trust.mean().item())
                stats["scored_rounds"] += 1.0
                if reference_cost is None:
                    reference_cost = cost[:, 0].clone()

                best_cost, best_index = cost.min(dim=1)
                best_candidate = self._gather(candidates, best_index)
                improved = best_cost < global_best_cost
                global_best_cost = torch.where(
                    improved, best_cost, global_best_cost
                )
                global_best = torch.where(
                    improved[:, None, None], best_candidate, global_best
                )

                elite_cost, elite_index = torch.topk(
                    cost, k=self.topk, dim=1, largest=False
                )
                del elite_cost
                batch_index = torch.arange(batch_size, device=self.device)[:, None]
                elites = candidates[batch_index, elite_index]
                elite_mean = elites.mean(dim=1)
                elite_scale = elites.std(dim=1, unbiased=False)
                alpha = self.update_alpha
                batch_mean = (1.0 - alpha) * batch_mean + alpha * elite_mean
                batch_scale = (
                    (1.0 - alpha) * batch_scale + alpha * elite_scale
                ).clamp(min=self.std_floor, max=self.std_cap)

            final_mean = batch_mean[:, None]
            final_cost, _, _ = self._score(info, final_mean, batch_reference)
            stats["get_cost_calls"] += 1.0
            stats["final_rescore_calls"] += 1.0
            final_cost = final_cost[:, 0]
            improved = final_cost < global_best_cost
            global_best_cost = torch.where(improved, final_cost, global_best_cost)
            global_best = torch.where(
                improved[:, None, None], batch_mean, global_best
            )

            if reference_cost is None:
                raise RuntimeError("reference plan was never scored")
            stats["selected_reference"] += float(
                (global_best_cost == reference_cost).sum().item()
            )
            stats["solved_envs"] += float(batch_size)
            stats["predicted_non_degrading"] += float(
                (global_best_cost <= reference_cost + 1e-7).sum().item()
            )
            output_actions[start:end] = global_best
            output_costs[start:end] = global_best_cost

        stats["solve_time"] = time.time() - started
        if stats["scored_rounds"]:
            stats["base_cost_mean"] = stats["base_cost_sum"] / stats["scored_rounds"]
            stats["trust_cost_mean"] = stats["trust_cost_sum"] / stats["scored_rounds"]
        stats["predicted_non_degrading_fraction"] = (
            stats["predicted_non_degrading"] / max(stats["solved_envs"], 1.0)
        )
        self.timing_history.append(dict(stats))
        actions = output_actions.detach().cpu()
        return {
            "actions": actions,
            "mean": [actions],
            "var": [torch.zeros_like(actions)],
            "costs": output_costs.detach().cpu().tolist(),
            "timing": dict(stats),
        }

    def timing_summary(self) -> dict[str, float]:
        if not self.timing_history:
            return {}
        keys = sorted({key for row in self.timing_history for key in row})
        summary: dict[str, float] = {}
        for key in keys:
            values = [float(row.get(key, 0.0)) for row in self.timing_history]
            summary[f"{key}_sum"] = sum(values)
            summary[f"{key}_mean"] = sum(values) / len(values)
        summary["num_solves"] = len(self.timing_history)
        return summary
