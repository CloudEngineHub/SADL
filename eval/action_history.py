"""Causal action history for evaluation.

At a start step t a model that conditions on K previous primitive actions receives dataset rows t-K..t-1;
slots before the episode start hold the raw zero action (normalized with the training statistics), and every
executed action enters the window after each environment step. Models without history never see the dataset
action of row t.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MODULE_ID = "action_history"


@dataclass(frozen=True)
class HistorySpec:
    """How many previous primitive actions a model conditions on.

    n_blocks:   number of previous action blocks (0 = no action history).
    block_size: primitive actions per block (= the plan's action_block / frameskip).
    """

    n_blocks: int
    block_size: int

    def __post_init__(self):
        if int(self.n_blocks) < 0:
            raise ValueError(f"n_blocks must be >= 0, got {self.n_blocks}")
        if int(self.n_blocks) > 0 and int(self.block_size) < 1:
            raise ValueError(f"block_size must be >= 1, got {self.block_size}")

    @property
    def slots(self) -> int:
        return int(self.n_blocks) * int(self.block_size)

    @property
    def uses_history(self) -> bool:
        return self.slots > 0

    def to_dict(self) -> dict:
        return {"n_blocks": int(self.n_blocks), "block_size": int(self.block_size) if self.uses_history else 0}

    @classmethod
    def from_dict(cls, data: dict | None, default_block_size: int | None = None) -> "HistorySpec":
        """None / {} / n_blocks == 0 -> no history; block_size defaults to the plan's action_block."""
        if not data or int(data.get("n_blocks", 0)) == 0:
            return cls(0, 0)
        block = data.get("block_size")
        if block is None:
            block = default_block_size
        if block is None:
            raise ValueError("history.block_size is not set and no default was given")
        return cls(int(data["n_blocks"]), int(block))


def protocol_record(spec: HistorySpec) -> dict:
    """The history protocol written into every result (compared exactly by the validator)."""
    if not spec.uses_history:
        return {
            "mode": "none",
            "module": MODULE_ID,
            "slots": 0,
            "n_blocks": 0,
            "action_input": "none (info['action'] replaced by a NaN placeholder or removed)",
            "current_dataset_row_action_used": False,
            "target_action_visible": False,
        }
    return {
        "mode": "expert_continuation",
        "module": MODULE_ID,
        "slots": spec.slots,
        "n_blocks": int(spec.n_blocks),
        "block_size": int(spec.block_size),
        "initial_rows": "rows[t-slots:t]",
        "padding": "raw-zero left padding only before episode start",
        "online_update": "shift in executed primitive actions",
        "current_dataset_row_action_used": False,
        "target_action_visible": False,
    }


def left_pad_window(previous, slots: int, action_dim: int) -> tuple[np.ndarray, int]:
    """Place up to `slots` previous primitive actions at the end of a raw-zero window.

    Returns (window [slots, action_dim] float32, number of padded slots).
    Raises on NaN, on a wrong action width, or on more rows than slots.
    """
    previous = np.asarray(previous, dtype=np.float32)
    if previous.size == 0:
        previous = np.zeros((0, action_dim), dtype=np.float32)
    elif previous.ndim != 2 or previous.shape[1] != action_dim:
        raise ValueError(f"previous actions must be [n, {action_dim}], got {previous.shape}")
    if previous.shape[0] > slots:
        raise ValueError(f"Loaded {previous.shape[0]} previous actions for {slots} slots")
    if np.isnan(previous).any():
        raise ValueError("NaN action inside the causal history window (rows before the start step)")
    window = np.zeros((slots, action_dim), dtype=np.float32)
    if previous.shape[0]:
        window[-previous.shape[0]:] = previous
    return window, slots - previous.shape[0]


def history_window(episode_actions, start: int, slots: int) -> tuple[np.ndarray, int]:
    """Window for one episode given its full primitive action array (rows 0..T-1)."""
    episode_actions = np.asarray(episode_actions, dtype=np.float32)
    start = int(start)
    if start < 0:
        raise ValueError(f"negative start step {start}")
    if start > episode_actions.shape[0]:
        raise ValueError(f"start step {start} beyond the episode length {episode_actions.shape[0]}")
    begin = max(0, start - slots)
    return left_pad_window(episode_actions[begin:start], slots, episode_actions.shape[-1])


def build_initial_history(dataset, episodes, start_steps, spec: HistorySpec,
                          action_dim: int | None = None) -> tuple[np.ndarray, list[int]]:
    """Causal windows ending immediately before each start step.

    `dataset` is a stable_worldmodel dataset (load_chunk(episodes, starts, ends) with an
    exclusive end and frameskip 1).  Returns (history [B, K, action_dim] float32,
    padded slot count per episode).
    """
    slots = spec.slots
    if action_dim is None:
        action_dim = int(np.asarray(dataset.get_col_data("action")).shape[-1])
    histories, padded = [], []
    for episode, start in zip(episodes, start_steps):
        start = int(start)
        if start < 0:
            raise ValueError(f"episode {episode}: negative start step {start}")
        previous = np.zeros((0, action_dim), dtype=np.float32)
        if start > 0 and slots > 0:
            begin = max(0, start - slots)
            chunks = dataset.load_chunk(np.asarray([episode]), np.asarray([begin]), np.asarray([start]))
            previous = np.asarray(chunks[0]["action"], dtype=np.float32).reshape(-1, action_dim)
            if previous.shape[0] != start - begin:
                raise ValueError(
                    f"episode {episode}: expected {start - begin} rows [{begin},{start}), got {previous.shape[0]}")
        window, n_pad = left_pad_window(previous, slots, action_dim)
        histories.append(window)
        padded.append(int(n_pad))
    if not histories:
        return np.zeros((0, slots, action_dim), dtype=np.float32), []
    return np.stack(histories), padded


@dataclass
class BlockStandardScaler:
    """Apply primitive-action statistics independently to a flattened block (time-major)."""

    scaler: object

    def _dim(self):
        mean = getattr(self.scaler, "mean_", getattr(self.scaler, "mean", None))
        return int(np.asarray(mean).size)

    def _apply(self, values, fn):
        array = np.asarray(values)
        primitive_dim = self._dim()
        if array.shape[-1] % primitive_dim:
            raise ValueError(f"Action width {array.shape[-1]} is not divisible by {primitive_dim}")
        shape = array.shape
        return fn(array.reshape(-1, primitive_dim)).reshape(shape)

    def transform(self, values):
        return self._apply(values, self.scaler.transform)

    def inverse_transform(self, values):
        return self._apply(values, self.scaler.inverse_transform)


def _dead_mask(info_dict, batch):
    dead = np.zeros(batch, dtype=bool)
    for key in ("terminated", "truncated"):
        value = info_dict.get(key)
        if value is not None:
            dead |= np.asarray(value, dtype=bool).reshape(batch, -1).any(axis=1)
    return dead


class _PolicyWrapper:
    """Forward everything except get_action to the wrapped stable_worldmodel policy."""

    def __getattr__(self, name):
        if name == "policy":  # not yet set (e.g. during unpickling)
            raise AttributeError(name)
        return getattr(self.policy, name)

    def set_env(self, env):
        return self.policy.set_env(env)

    def set_seed(self, seed):
        self.seed = seed
        setter = getattr(self.policy, "set_seed", None)
        if setter is not None:
            return setter(seed)
        return None


class CausalHistoryPolicy(_PolicyWrapper):
    """Hand the policy the causal history; shift in the executed action after every env step.

    initial_history: [B, K, action_dim] raw primitive actions (build_initial_history).
    The wrapped policy receives info["action"] with shape [B, n_blocks, block_size*action_dim]
    (time-major inside each block, oldest block first).
    """

    def __init__(self, policy, initial_history, spec: HistorySpec, padded_slots=None,
                 episodes=None, start_steps=None):
        history = np.asarray(initial_history, dtype=np.float32)
        if history.ndim != 3:
            raise ValueError(f"Expected [B, slots, action_dim], got {history.shape}")
        if not spec.uses_history or history.shape[1] != spec.slots:
            raise ValueError(f"history has {history.shape[1]} slots, spec needs {spec.slots}")
        if np.isnan(history).any():
            raise ValueError("NaN in the initial action history")
        self.policy = policy
        self.spec = spec
        self.initial_history = history
        self.history = history.copy()
        self.action_dim = int(history.shape[-1])
        self._fresh = np.ones(history.shape[0], dtype=bool)
        self.padded_slots = list(padded_slots) if padded_slots is not None else None
        self.episodes = list(episodes) if episodes is not None else None
        self.start_steps = list(start_steps) if start_steps is not None else None
        self.calls = 0
        self.seed = getattr(policy, "seed", None)

    def reset_envs(self, env_ids, windows, padded_slots=None, episodes=None, start_steps=None):
        """Start new episodes in the given envs with new causal windows [n, K, action_dim]."""
        windows = np.asarray(windows, dtype=np.float32)
        env_ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        if windows.shape != (len(env_ids),) + self.history.shape[1:]:
            raise ValueError(f"reset windows {windows.shape} do not match {self.history.shape[1:]}")
        if np.isnan(windows).any():
            raise ValueError("NaN in a reset action history")
        self.history[env_ids] = windows
        self._fresh[env_ids] = True
        for name, values in (("padded_slots", padded_slots), ("episodes", episodes), ("start_steps", start_steps)):
            current = getattr(self, name)
            if values is not None and current is not None:
                for i, v in zip(env_ids, values):
                    current[int(i)] = int(v)

    def observe(self, info_dict):
        """Shift in the executed primitive action of every env that already acted."""
        batch = self.history.shape[0]
        flush = info_dict.get("_needs_flush")
        if flush is not None and np.asarray(flush, dtype=bool).any():
            raise RuntimeError("environment auto-reset: call reset_envs() with the new episodes' causal windows")
        update = ~self._fresh
        if update.any():
            latest = np.asarray(info_dict["action"], dtype=np.float32)
            if latest.ndim == 3:
                latest = latest[:, -1]
            if latest.ndim != 2 or latest.shape[0] != batch:
                raise ValueError(f"Expected latest actions [B,D], got {latest.shape}")
            if latest.shape[1] != self.action_dim:
                raise ValueError(
                    f"executed action width {latest.shape[1]} != primitive action_dim {self.action_dim}")
            nan_rows = np.isnan(latest).any(axis=1) & update
            if (nan_rows & ~_dead_mask(info_dict, batch)).any():
                raise ValueError("NaN executed action for a live environment")
            shifted = np.concatenate([self.history[:, 1:], latest[:, None]], axis=1)
            self.history = np.where(update[:, None, None], shifted, self.history)
        self._fresh[:] = False

    def model_history(self):
        """History as model tokens: [B, n_blocks, block_size * action_dim] (exact block width)."""
        batch, slots, dim = self.history.shape
        if slots != self.spec.slots or dim != self.action_dim:
            raise ValueError(f"history window {self.history.shape} != [B, {self.spec.slots}, {self.action_dim}]")
        return self.history.reshape(batch, self.spec.n_blocks, self.spec.block_size * dim)

    def get_action(self, info_dict, **kwargs):
        self.observe(info_dict)
        self.calls += 1
        prepared = dict(info_dict)
        prepared["action"] = self.model_history()
        return self.policy.get_action(prepared, **kwargs)

    def episode_records(self):
        return [
            {"env": i,
             "episode": None if self.episodes is None else int(self.episodes[i]),
             "start_step": None if self.start_steps is None else int(self.start_steps[i]),
             "padded_slots": None if self.padded_slots is None else int(self.padded_slots[i])}
            for i in range(self.history.shape[0])
        ]


class NoActionHistoryPolicy(_PolicyWrapper):
    """For models without action history: the dataset's row-t action never reaches the model.

    placeholder="nan":  info["action"] keeps its shape but is all NaN (for model code
                        that requires the key, e.g. LeWM get_cost pops it);
    placeholder="drop": the key is removed.
    """

    def __init__(self, policy, placeholder: str = "nan", episodes=None, start_steps=None):
        if placeholder not in ("nan", "drop"):
            raise ValueError(f"unknown placeholder {placeholder!r}")
        self.policy = policy
        self.placeholder = placeholder
        self.spec = HistorySpec(0, 0)
        self.episodes = list(episodes) if episodes is not None else None
        self.start_steps = list(start_steps) if start_steps is not None else None
        self.calls = 0
        self.seed = getattr(policy, "seed", None)

    def get_action(self, info_dict, **kwargs):
        self.calls += 1
        prepared = dict(info_dict)
        if "action" in prepared:
            if self.placeholder == "drop":
                del prepared["action"]
            else:
                prepared["action"] = np.full(np.shape(prepared["action"]), np.nan, dtype=np.float32)
        return self.policy.get_action(prepared, **kwargs)

    def episode_records(self):
        n = len(self.episodes) if self.episodes is not None else 0
        return [{"env": i, "episode": int(self.episodes[i]),
                 "start_step": None if self.start_steps is None else int(self.start_steps[i]),
                 "padded_slots": 0} for i in range(n)]


def wrap_policy(policy, dataset, episodes, start_steps, spec: HistorySpec, placeholder: str = "nan"):
    """Wrap a stable_worldmodel model policy for dataset-start evaluation per its history spec.

    History models: the policy's action processor becomes block-aware (BlockStandardScaler),
    so a [B, n_blocks, block*D] token is normalized per primitive with the training statistics.
    Returns (wrapped policy, protocol record).
    """
    episodes = [int(e) for e in episodes]
    start_steps = [int(s) for s in start_steps]
    if not spec.uses_history:
        return NoActionHistoryPolicy(policy, placeholder, episodes, start_steps), protocol_record(spec)
    initial, padded = build_initial_history(dataset, episodes, start_steps, spec)
    process = getattr(policy, "process", None)
    if isinstance(process, dict) and "action" in process and not isinstance(process["action"], BlockStandardScaler):
        process["action"] = BlockStandardScaler(process["action"])
    wrapped = CausalHistoryPolicy(policy, initial, spec, padded, episodes, start_steps)
    return wrapped, protocol_record(spec)
