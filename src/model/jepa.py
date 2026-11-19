"""INTACT: a LeWM-style JEPA with a shared intent-to-action operator."""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import torch.utils.checkpoint

from collapse_study.sdpa import LayerwiseSDPARouter, sdpa_context


class JEPA(nn.Module):
    """Forward world model plus a deployable local/goal intent action law."""

    VALID_INTENT_MODES = {"waypoint", "goal_displacement"}
    VALID_FRAME_READOUTS = {"cls", "mean_pool"}

    def __init__(
        self,
        encoder: nn.Module,
        predictor: nn.Module,
        action_encoder: nn.Module,
        projector: nn.Module | None = None,
        pred_proj: nn.Module | None = None,
        intent_actor: nn.Module | None = None,
        intent_mode: str = "goal_displacement",
        predict_residual: bool = False,
        actor_warmstart: bool = True,
        encoder_sdpa_layers: str = "all_math",
        predictor_sdpa_backend: str = "math",
        frame_readout: str = "cls",
    ) -> None:
        super().__init__()
        if intent_mode not in self.VALID_INTENT_MODES:
            raise ValueError(
                f"intent_mode must be one of {sorted(self.VALID_INTENT_MODES)}, got {intent_mode}"
            )
        if frame_readout not in self.VALID_FRAME_READOUTS:
            raise ValueError(
                "frame_readout must be one of "
                f"{sorted(self.VALID_FRAME_READOUTS)}, got {frame_readout}"
            )
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        self.intent_actor = intent_actor
        self.intent_mode = intent_mode
        self.predict_residual = predict_residual
        self.actor_warmstart = actor_warmstart
        self.predictor_sdpa_backend = str(predictor_sdpa_backend)
        self.frame_readout = frame_readout
        self._capture_block_outputs = False
        self._encoder_sdpa_router = LayerwiseSDPARouter(
            self.encoder, encoder_sdpa_layers
        )

    def set_diagnostic_capture(self, enabled: bool) -> None:
        self._capture_block_outputs = bool(enabled)

    def encode(self, info: dict) -> dict:
        pixels = info["pixels"].float()
        batch_size = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...")
        self._encoder_sdpa_router.begin_capture(self._capture_block_outputs)
        encoder_kwargs = {"interpolate_pos_encoding": True}
        if self._capture_block_outputs:
            encoder_kwargs["output_hidden_states"] = True
        encoded = self.encoder(pixels, **encoder_kwargs)
        token_features = encoded.last_hidden_state
        h_cls = token_features[:, 0]
        # Readout: the default ``cls`` path feeds
        # the CLS token to the projector, while ``mean_pool`` uses the patch
        # token mean. In the throughput profile the unused mean is omitted
        # only when diagnostics are disabled; the projected representation is
        # unchanged.
        skip_unused_patch_mean = (
            os.environ.get("INTACT_SKIP_UNUSED_PATCH_MEAN", "0") == "1"
            and self.frame_readout == "cls"
            and not self._capture_block_outputs
        )
        h_patch_mean = (
            None if skip_unused_patch_mean else token_features[:, 1:].mean(dim=1)
        )
        h_repr = h_cls if self.frame_readout == "cls" else h_patch_mean
        embedding = self.projector(h_repr)
        info["h_cls"] = rearrange(h_cls, "(b t) d -> b t d", b=batch_size)
        if h_patch_mean is not None:
            info["h_patch_mean"] = rearrange(
                h_patch_mean, "(b t) d -> b t d", b=batch_size
            )
        info["h_repr"] = rearrange(h_repr, "(b t) d -> b t d", b=batch_size)
        if self._capture_block_outputs:
            info["h_patch_tokens"] = rearrange(
                token_features[:, 1:], "(b t) n d -> b t n d", b=batch_size
            )
        info["emb"] = rearrange(
            embedding, "(b t) d -> b t d", b=batch_size
        )
        hook_outputs = self._encoder_sdpa_router.consume_captures()
        hidden_states = getattr(encoded, "hidden_states", None)
        # ViT hidden_states starts with the data-dependent patch embedding and
        # then stores one output per transformer block. C-RVF deliberately
        # starts at block 1, not at the learned pre-data CLS token.
        block_outputs = (
            [state[:, 0] for state in hidden_states[1:]]
            if self._capture_block_outputs and hidden_states is not None
            else hook_outputs
        )
        if block_outputs:
            info["encoder_block_cls"] = [
                rearrange(value, "(b t) d -> b t d", b=batch_size)
                for value in block_outputs
            ]
        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])
        return info

    def predict(self, embedding: torch.Tensor, action_embedding: torch.Tensor) -> torch.Tensor:
        with sdpa_context(self.predictor_sdpa_backend):
            use_checkpoint = (
                os.environ.get("INTACT_CHECKPOINT_PREDICTOR", "0") == "1"
                and self.training
                and embedding.requires_grad
            )
            if use_checkpoint:
                # The predictor is pure tensor code. Recomputing its six
                # transformer blocks during backward keeps a real batch=256
                # within 32 GiB without changing values or optimizer math.
                prediction = torch.utils.checkpoint.checkpoint(
                    self.predictor,
                    embedding,
                    action_embedding,
                    use_reentrant=False,
                )
            else:
                prediction = self.predictor(embedding, action_embedding)
        prediction = self.pred_proj(rearrange(prediction, "b t d -> (b t) d"))
        prediction = rearrange(
            prediction, "(b t) d -> b t d", b=embedding.size(0)
        )
        return embedding + prediction if self.predict_residual else prediction

    def has_intent_actor(self) -> bool:
        return self.intent_actor is not None

    def set_actor_warmstart(self, enabled: bool) -> None:
        self.actor_warmstart = bool(enabled)

    def get_action_dim(self, fallback: int | None = None) -> int:
        if self.intent_actor is not None:
            return self.intent_actor.action_dim
        if hasattr(self.action_encoder, "patch_embed"):
            return self.action_encoder.patch_embed.in_channels
        if fallback is not None:
            return fallback
        raise RuntimeError("Could not infer action dimension")

    def goal_intent(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
        steps_remaining: int | float,
    ) -> torch.Tensor:
        displacement = goal - current
        if self.intent_mode == "waypoint":
            if float(steps_remaining) <= 0:
                raise ValueError("steps_remaining must be positive")
            return displacement / float(steps_remaining)
        return displacement

    def action_parameters(
        self,
        z: torch.Tensor,
        intent: torch.Tensor,
        previous_action: torch.Tensor,
        previous_embedding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.intent_actor is None:
            raise RuntimeError("intent_actor is not configured")
        if previous_embedding is None:
            previous_embedding = self.action_encoder(previous_action)
        return self.intent_actor(z, intent, previous_embedding)

    def action_nll(
        self,
        z: torch.Tensor,
        intent: torch.Tensor,
        previous_action: torch.Tensor,
        target_action: torch.Tensor,
        reduction: str = "mean",
        previous_embedding: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        mean, log_std = self.action_parameters(
            z, intent, previous_action, previous_embedding=previous_embedding
        )
        nll = 0.5 * (
            (target_action - mean).square() * torch.exp(-2 * log_std)
            + 2 * log_std
        ).mean(dim=-1)
        if reduction == "mean":
            loss = nll.mean()
        elif reduction == "none":
            loss = nll
        else:
            raise ValueError(f"Unsupported reduction: {reduction}")
        return {
            "loss": loss,
            "nll": nll,
            "mean": mean,
            "log_std": log_std,
            "mae": (mean - target_action).abs().mean(dim=-1),
            "rmse": (mean - target_action).square().mean(dim=-1).sqrt(),
        }

    def _coerce_action_history(
        self,
        action: torch.Tensor | None,
        *,
        batch_size: int,
        seq_len: int,
        action_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if action is None:
            raise ValueError(
                "Action history is required; initialize reset history as raw "
                "zero before applying the action normalizer"
            )
        action = action.to(device=device, dtype=dtype)
        if not torch.isfinite(action).all():
            raise ValueError(
                "Action history contains non-finite values; initialize missing "
                "history as raw zero before action normalization"
            )
        if action.ndim == 2:
            action = action[:, None]
        if action.ndim != 3:
            raise ValueError(f"Expected action history [B,T,D], got {action.shape}")
        if action.size(-1) == action_dim:
            return action
        if action_dim % action.size(-1) == 0:
            return action.repeat(1, 1, action_dim // action.size(-1))
        raise ValueError(
            f"Cannot coerce action dimension {action.size(-1)} to {action_dim}"
        )

    def rollout_one_step(
        self,
        embedding_history: torch.Tensor,
        action: torch.Tensor,
        action_history: torch.Tensor,
        history_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        size = min(history_size, embedding_history.size(1), action_history.size(1))
        embedding_context = embedding_history[:, -size:]
        action_context = action_history[:, -size:].clone()
        action_context[:, -1] = action
        prediction = self.predict(
            embedding_context, self.action_encoder(action_context)
        )[:, -1:]
        return (
            torch.cat([embedding_history, prediction], dim=1),
            torch.cat([action_history, action[:, None]], dim=1),
        )

    def _encode_goal(self, info: dict) -> torch.Tensor:
        goal = {key: value for key, value in info.items() if torch.is_tensor(value)}
        goal["pixels"] = goal["goal"]
        for key in list(goal):
            if key.startswith("goal_"):
                goal[key.removeprefix("goal_")] = goal.pop(key)
        goal.pop("action", None)
        return self.encode(goal)["emb"][:, -1]

    @torch.inference_mode()
    def get_action(
        self,
        info: dict,
        horizon: int = 1,
        prefix_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate a deterministic Direct plan with zero candidate search."""
        if horizon < 1:
            raise ValueError("horizon must be positive")
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        info = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in info.items()
        }
        batch_size = info["pixels"].size(0)
        fallback = None
        if prefix_actions is not None:
            fallback = prefix_actions.size(-1)
        elif torch.is_tensor(info.get("action")):
            fallback = info["action"].size(-1)
        action_dim = self.get_action_dim(fallback)

        if not self.actor_warmstart or self.intent_actor is None:
            return torch.zeros(
                batch_size, horizon, action_dim, device=device, dtype=dtype
            )

        initial = {key: value for key, value in info.items() if torch.is_tensor(value)}
        initial["action"] = self._coerce_action_history(
            initial.get("action"),
            batch_size=batch_size,
            seq_len=info["pixels"].size(1),
            action_dim=action_dim,
            device=device,
            dtype=dtype,
        )
        initial = self.encode(initial)
        embeddings = initial["emb"]
        actions_so_far = initial["action"]
        history_size = self.predictor.pos_embedding.size(1)
        goal = self._encode_goal(info)

        if prefix_actions is not None and prefix_actions.numel() > 0:
            prefix_actions = self._coerce_action_history(
                prefix_actions,
                batch_size=batch_size,
                seq_len=prefix_actions.size(1),
                action_dim=action_dim,
                device=device,
                dtype=actions_so_far.dtype,
            )
            for step in range(prefix_actions.size(1)):
                embeddings, actions_so_far = self.rollout_one_step(
                    embeddings,
                    prefix_actions[:, step],
                    actions_so_far,
                    history_size,
                )

        plan = []
        intent_norms = []
        for step in range(horizon):
            current = embeddings[:, -1]
            intent = self.goal_intent(current, goal, horizon - step)
            previous_embedding = self.action_encoder(actions_so_far[:, -1:])[:, -1]
            action = self.intent_actor.action_mean(
                current, intent, previous_embedding
            )
            plan.append(action)
            intent_norms.append(intent.norm(dim=-1).mean())
            embeddings, actions_so_far = self.rollout_one_step(
                embeddings, action, actions_so_far, history_size
            )

        self.last_direct_diagnostics = {
            "intent_norm": float(torch.stack(intent_norms).mean()),
            "terminal_latent_error": float(
                (embeddings[:, -1] - goal).norm(dim=-1).mean()
            ),
            "forward_calls": float(horizon),
            "candidate_sequences": 0.0,
        }
        return torch.stack(plan, dim=1)

    def rollout(
        self, info: dict, action_sequence: torch.Tensor, history_size: int | None = None
    ) -> dict:
        """Roll out action candidates for optional LeWM-compatible CEM checks."""
        batch_size, samples, horizon = action_sequence.shape[:3]
        observed_steps = info["pixels"].size(2)
        initial_actions, future_actions = torch.split(
            action_sequence, [observed_steps, horizon - observed_steps], dim=2
        )
        initial = {
            key: value[:, 0]
            for key, value in info.items()
            if torch.is_tensor(value)
        }
        initial["action"] = initial_actions[:, 0]
        initial = self.encode(initial)
        embeddings = initial["emb"].unsqueeze(1).expand(
            batch_size, samples, -1, -1
        )
        embeddings = rearrange(embeddings, "b s t d -> (b s) t d").clone()
        actions = rearrange(initial_actions, "b s t d -> (b s) t d")
        future_actions = rearrange(future_actions, "b s t d -> (b s) t d")
        context = history_size or self.predictor.pos_embedding.size(1)

        for step in range(horizon - observed_steps):
            prediction = self.predict(
                embeddings[:, -context:],
                self.action_encoder(actions)[:, -context:],
            )[:, -1:]
            embeddings = torch.cat([embeddings, prediction], dim=1)
            actions = torch.cat(
                [actions, future_actions[:, step : step + 1]], dim=1
            )

        final_prediction = self.predict(
            embeddings[:, -context:], self.action_encoder(actions)[:, -context:]
        )[:, -1:]
        embeddings = torch.cat([embeddings, final_prediction], dim=1)
        info["predicted_emb"] = rearrange(
            embeddings, "(b s) t d -> b s t d", b=batch_size, s=samples
        )
        return info

    def criterion(self, info: dict) -> torch.Tensor:
        predicted = info["predicted_emb"][:, :, -1]
        goal = info["goal_emb"][:, None].expand_as(predicted)
        return F.mse_loss(predicted, goal.detach(), reduction="none").sum(dim=-1)

    def get_cost(self, info: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        info = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in info.items()
        }
        collapsed = {
            key: value[:, 0]
            for key, value in info.items()
            if torch.is_tensor(value)
        }
        info["goal_emb"] = self._encode_goal(collapsed)
        return self.criterion(self.rollout(info, action_candidates))
