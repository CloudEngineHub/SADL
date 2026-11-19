import time
from collections import defaultdict

import numpy as np
import torch
from stable_worldmodel.solver import CEMSolver
from stable_worldmodel.solver.cem import prepare_init_action


def _mean_float(x: torch.Tensor) -> float:
    return float(x.detach().float().mean().cpu().item())


class TimedCEMSolver(CEMSolver):
    """CEMSolver with lightweight timing and rollout-budget instrumentation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timing_history = []
        self.last_timing = {}

    def _solve_impl(
        self, info_dict: dict, init_action: torch.Tensor | None = None
    ) -> dict:
        return CEMSolver.solve(self, info_dict, init_action=init_action)

    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        stats = defaultdict(float)
        cost_trace = []
        stats["get_cost_calls"] = 0
        stats["candidate_action_sequences"] = 0
        stats["candidate_action_steps"] = 0
        stats["configured_num_samples"] = self.num_samples
        stats["configured_n_steps"] = self.n_steps
        stats["configured_horizon"] = self.horizon
        stats["configured_rollout_budget"] = (
            self.num_samples * self.n_steps * self.horizon
        )
        stats["mog_population_mode"] = float(
            isinstance(self, MixturePopulationCEMSolver)
        )

        original_get_cost = self.model.get_cost

        def timed_get_cost(timed_info_dict, action_candidates):
            start = time.time()
            out = original_get_cost(timed_info_dict, action_candidates)
            elapsed = time.time() - start
            stats["get_cost_calls"] += 1
            stats["get_cost_time"] += elapsed
            if torch.is_tensor(action_candidates) and action_candidates.ndim >= 4:
                b, n, h = action_candidates.shape[:3]
                stats["candidate_action_sequences"] += int(b * n)
                stats["candidate_action_steps"] += int(b * n * h)
            if torch.is_tensor(out) and out.ndim == 2 and out.numel() > 0:
                k = min(int(self.topk), out.shape[1])
                topk_vals = torch.topk(out, k=k, dim=1, largest=False).values
                cost_trace.append(
                    {
                        "mean_action_cost": _mean_float(out[:, 0]),
                        "population_mean_cost": _mean_float(out),
                        "best_cost": _mean_float(out.min(dim=1).values),
                        "elite_mean_cost": _mean_float(topk_vals.mean(dim=1)),
                    }
                )
            return out

        solve_start = time.time()
        try:
            object.__setattr__(self.model, "get_cost", timed_get_cost)
            outputs = self._solve_impl(info_dict, init_action=init_action)
        finally:
            object.__setattr__(self.model, "get_cost", original_get_cost)

        stats["solve_time"] = time.time() - solve_start
        stats["get_cost_time_fraction"] = (
            stats["get_cost_time"] / stats["solve_time"]
            if stats["solve_time"] > 0
            else 0.0
        )
        if cost_trace:
            n_steps = max(int(self.n_steps), 1)
            first_steps = [
                item for i, item in enumerate(cost_trace) if i % n_steps == 0
            ]
            final_steps = [
                item
                for i, item in enumerate(cost_trace)
                if i % n_steps == n_steps - 1
            ]
            if not final_steps:
                final_steps = [cost_trace[-1]]
            for prefix, items in (
                ("cem_initial", first_steps),
                ("cem_final", final_steps),
            ):
                for key in items[0]:
                    stats[f"{prefix}_{key}"] = sum(
                        float(item[key]) for item in items
                    ) / len(items)
            stats["cem_initial_to_final_elite_delta"] = (
                stats["cem_initial_mean_action_cost"]
                - stats["cem_final_elite_mean_cost"]
            )
            stats["cem_initial_best_to_final_best_delta"] = (
                stats["cem_initial_best_cost"] - stats["cem_final_best_cost"]
            )
            stats["cem_final_elite_over_initial_mean_ratio"] = (
                stats["cem_final_elite_mean_cost"]
                / stats["cem_initial_mean_action_cost"]
                if abs(stats["cem_initial_mean_action_cost"]) > 1e-12
                else 0.0
            )
        if hasattr(self.model, "actor_warmstart"):
            stats["actor_warmstart_enabled"] = float(bool(self.model.actor_warmstart))
        self.last_timing = dict(stats)
        self.timing_history.append(self.last_timing)
        outputs["timing"] = self.last_timing
        return outputs

    def timing_summary(self):
        if not self.timing_history:
            return {}
        keys = sorted({k for item in self.timing_history for k in item})
        summary = {}
        for key in keys:
            vals = [float(item.get(key, 0.0)) for item in self.timing_history]
            summary[f"{key}_sum"] = sum(vals)
            summary[f"{key}_mean"] = sum(vals) / len(vals)
        summary["num_solves"] = len(self.timing_history)
        return summary


class MixturePopulationCEMSolver(TimedCEMSolver):
    """CEM whose first population is stratified from a learned action mixture."""

    @torch.inference_mode()
    def _solve_impl(
        self, info_dict: dict, init_action: torch.Tensor | None = None
    ) -> dict:
        start_time = time.time()
        outputs = {"costs": [], "mean": [], "var": []}
        total_envs = len(next(iter(info_dict.values())))

        prefix_actions = init_action
        init_action = prepare_init_action(
            self.model,
            info_dict,
            init_action,
            self.horizon,
            n_envs=total_envs,
            action_dim=self.action_dim,
        )
        mean, var = self.init_action_distrib(total_envs, init_action)
        mean = mean.to(self.device)
        var = var.to(self.device)

        population = None
        n_prefix = prefix_actions.shape[1] if prefix_actions is not None else 0
        remaining = self.horizon - n_prefix
        if remaining > 0 and hasattr(self.model, "get_action_population"):
            population_tail = self.model.get_action_population(
                info_dict,
                horizon=remaining,
                num_samples=self.num_samples,
                prefix_actions=prefix_actions,
                generator=self.torch_gen,
            )
            if population_tail is not None:
                population_tail = population_tail.to(
                    device=self.device, dtype=self.dtype
                )
                if prefix_actions is not None and prefix_actions.numel() > 0:
                    prefix = prefix_actions.to(
                        device=self.device, dtype=self.dtype
                    )[:, None].expand(
                        total_envs,
                        self.num_samples,
                        prefix_actions.shape[1],
                        self.action_dim,
                    )
                    population = torch.cat([prefix, population_tail], dim=2)
                else:
                    population = population_tail
                if population.shape != (
                    total_envs,
                    self.num_samples,
                    self.horizon,
                    self.action_dim,
                ):
                    raise ValueError(
                        "MoG population has invalid shape: "
                        f"{tuple(population.shape)}"
                    )

        for callback in self.callbacks:
            callback.reset()

        for start_index in range(0, total_envs, self.batch_size):
            end_index = min(start_index + self.batch_size, total_envs)
            current_batch_size = end_index - start_index
            batch_mean = mean[start_index:end_index]
            batch_var = var[start_index:end_index]
            batch_population = (
                population[start_index:end_index] if population is not None else None
            )

            expanded_infos = {}
            for key, value in info_dict.items():
                value_batch = value[start_index:end_index]
                if torch.is_tensor(value):
                    target_dtype = self.dtype if value_batch.is_floating_point() else None
                    value_batch = (
                        value_batch.to(device=self.device, dtype=target_dtype)
                        .unsqueeze(1)
                        .expand(
                            current_batch_size,
                            self.num_samples,
                            *value_batch.shape[1:],
                        )
                    )
                elif isinstance(value, np.ndarray):
                    value_batch = np.repeat(
                        value_batch[:, None, ...], self.num_samples, axis=1
                    )
                expanded_infos[key] = value_batch

            for callback in self.callbacks:
                callback.start_batch()
            final_batch_cost = None

            for step in range(self.n_steps):
                if step == 0 and batch_population is not None:
                    candidates = batch_population.clone()
                else:
                    candidates = torch.randn(
                        current_batch_size,
                        self.num_samples,
                        self.horizon,
                        self.action_dim,
                        generator=self.torch_gen,
                        device=self.device,
                        dtype=self.dtype,
                    )
                    candidates = (
                        candidates * batch_var.unsqueeze(1)
                        + batch_mean.unsqueeze(1)
                    )
                candidates[:, 0] = batch_mean

                costs = self.model.get_cost(expanded_infos, candidates)
                if not isinstance(costs, torch.Tensor):
                    raise TypeError(
                        f"Expected cost tensor, got {type(costs)}"
                    )
                expected_shape = (current_batch_size, self.num_samples)
                if tuple(costs.shape) != expected_shape:
                    raise ValueError(
                        f"Expected cost shape {expected_shape}, got {tuple(costs.shape)}"
                    )

                topk_vals, topk_inds = torch.topk(
                    costs, k=self.topk, dim=1, largest=False
                )
                batch_indices = torch.arange(
                    current_batch_size, device=self.device
                ).unsqueeze(1).expand(-1, self.topk)
                topk_candidates = candidates[batch_indices, topk_inds]
                previous_mean = batch_mean
                previous_var = batch_var
                batch_mean = topk_candidates.mean(dim=1)
                batch_var = topk_candidates.std(dim=1)

                for callback in self.callbacks:
                    callback(
                        step=step,
                        candidates=candidates,
                        costs=costs,
                        topk_vals=topk_vals,
                        topk_inds=topk_inds,
                        topk_candidates=topk_candidates,
                        mean=batch_mean,
                        var=batch_var,
                        prev_mean=previous_mean,
                        prev_var=previous_var,
                    )
                final_batch_cost = topk_vals.mean(dim=1).cpu().tolist()

            mean[start_index:end_index] = batch_mean
            var[start_index:end_index] = batch_var
            outputs["costs"].extend(final_batch_cost)

        outputs["actions"] = mean.detach().cpu()
        outputs["mean"] = [mean.detach().cpu()]
        outputs["var"] = [var.detach().cpu()]
        if self.callbacks:
            outputs["callbacks"] = {}
            for callback in self.callbacks:
                callback.end_solve()
                outputs["callbacks"][callback.output_key] = callback.history
        print(f"CEM solve time: {time.time() - start_time:.4f} seconds")
        return outputs
