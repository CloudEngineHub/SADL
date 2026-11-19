"""INTACT / LeWM model, dataset and loss components shared by shared-BN model training."""

from __future__ import annotations

import json
import os
from functools import partial
from pathlib import Path

# This must be set before CUDA creates a cuBLAS handle.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from collapse_study.sdpa import configure_training_sdpa
from module import SIGReg
from single_baseline import (
    build_loaders as build_single_loaders,
    SingleBaselineCheckpoint,
    configure as configure_single_baseline,
    enabled as single_baseline_enabled,
    endpoint_regularizer,
    joint_regularizer,
)
from utils import (
    PreviousActionDataset,
    SaveCkptCallback,
    get_column_normalizer,
    get_column_stats,
    get_img_preprocessor,
)


def configure_training_runtime(
    seed: int, global_sdpa: str = "math", deterministic: bool = True
) -> None:
    """Configure training controls and the non-routed attention backend.

    ``deterministic=True`` is the release/reference contract.  Performance
    runs may opt into cuDNN autotuning and nondeterministic kernels explicitly;
    the launcher records that choice so results are not mixed silently.
    """
    configure_training_sdpa(global_sdpa)
    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    pl.seed_everything(seed, workers=True)


def configure_deterministic_math(seed: int) -> None:
    """Use MATH attention for every encoder and predictor call."""
    configure_training_runtime(seed, global_sdpa="math")


def predict_adjacent_latents(model, embeddings, action_embeddings, history_size):
    """Predict every adjacent transition with a bounded history window."""
    transitions = embeddings.size(1) - 1
    if transitions < 1:
        raise ValueError("A training window must contain at least two frames")

    prefix = min(history_size, transitions)
    predictions = [
        model.predict(embeddings[:, :prefix], action_embeddings[:, :prefix])
    ]
    for index in range(prefix, transitions):
        start = max(0, index - history_size + 1)
        prediction = model.predict(
            embeddings[:, start : index + 1],
            action_embeddings[:, start : index + 1],
        )[:, -1:]
        predictions.append(prediction)
    return torch.cat(predictions, dim=1)


def construct_intents(
    embeddings: torch.Tensor,
    intent_mode: str,
    local_start: int = 0,
    goal_start: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct attached physical intents and deployable terminal-goal intents."""
    max_start = embeddings.size(1) - 2
    if local_start < 0 or local_start > max_start:
        raise ValueError(
            f"local_start must be in [0, T-2], got {local_start} for T={embeddings.size(1)}"
        )
    if goal_start < 0 or goal_start > max_start:
        raise ValueError(
            f"goal_start must be in [0, T-2], got {goal_start} for T={embeddings.size(1)}"
        )
    current = embeddings[:, local_start:-1]
    successor = embeddings[:, local_start + 1 :]
    local = successor - current
    deployment_current = embeddings[:, goal_start:-1]
    goal = embeddings[:, -1:].detach().expand_as(deployment_current)
    displacement = goal - deployment_current
    if intent_mode == "waypoint":
        remaining = torch.arange(
            deployment_current.size(1),
            0,
            -1,
            device=embeddings.device,
            dtype=embeddings.dtype,
        ).view(1, -1, 1)
        deployment = displacement / remaining
    elif intent_mode == "goal_displacement":
        deployment = displacement
    else:
        raise ValueError(f"Unknown intent mode: {intent_mode}")
    return local, deployment


def _action_encoder_is_reusable(action_encoder) -> bool:
    """Whether one full-history encoding preserves per-window semantics."""
    for layer in action_encoder.modules():
        if isinstance(layer, torch.nn.Dropout) and layer.p > 0:
            return False
        if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
            return False
    return True


def paired_intent_action_loss(model, embeddings, actions, previous_actions, cfg):
    """Apply physical and deployment NLLs through the same action operator."""
    local_weight = float(cfg.loss.intent.local_weight)
    goal_weight = float(cfg.loss.intent.goal_weight)
    local_start = int(cfg.loss.intent.local_start)
    goal_start = int(cfg.loss.intent.goal_start)
    if local_weight <= 0 and goal_weight <= 0:
        zero = embeddings.new_zeros(())
        return {
            "action_loss": zero,
            "local_nll": zero,
            "goal_nll": zero,
            "local_mae": zero,
            "goal_mae": zero,
        }

    local_intent, goal_intent = construct_intents(
        embeddings,
        model.intent_mode,
        local_start=local_start,
        goal_start=goal_start,
    )
    zero = embeddings.new_zeros(())
    # The action embedder is deterministic in this config (no dropout
    # or running-stat layers). Encode the full boundary-aware history once and
    # slice it for local/goal windows instead of recomputing it twice.
    previous_embeddings = (
        model.action_encoder(previous_actions)
        if (
            os.environ.get("INTACT_REUSE_ACTION_EMBEDDING", "0") == "1"
            and _action_encoder_is_reusable(model.action_encoder)
        )
        else None
    )
    local_stats = {
        "loss": zero,
        "mae": zero[None],
        "mean": zero[None],
        "log_std": zero[None],
    }
    if local_weight > 0:
        local_action = actions[:, local_start : embeddings.size(1) - 1]
        local_previous = previous_actions[:, local_start : embeddings.size(1) - 1]
        local_stats = model.action_nll(
            z=embeddings[:, local_start:-1],
            intent=local_intent,
            previous_action=local_previous,
            target_action=local_action,
            previous_embedding=(
                None
                if previous_embeddings is None
                else previous_embeddings[:, local_start : embeddings.size(1) - 1]
            ),
        )

    goal_stats = {
        "loss": zero,
        "mae": zero[None],
        "mean": zero[None],
        "log_std": zero[None],
    }
    if goal_weight > 0:
        goal_action = actions[:, goal_start : embeddings.size(1) - 1]
        goal_previous = previous_actions[:, goal_start : embeddings.size(1) - 1]
        goal_stats = model.action_nll(
            z=embeddings[:, goal_start:-1],
            intent=goal_intent,
            previous_action=goal_previous,
            target_action=goal_action,
            previous_embedding=(
                None
                if previous_embeddings is None
                else previous_embeddings[:, goal_start : embeddings.size(1) - 1]
            ),
        )

    return {
        "action_loss": local_weight * local_stats["loss"]
        + goal_weight * goal_stats["loss"],
        "local_nll": local_stats["loss"],
        "goal_nll": goal_stats["loss"],
        "local_mae": local_stats["mae"].mean().detach(),
        "goal_mae": goal_stats["mae"].mean().detach(),
        "local_mean_rms": local_stats["mean"].square().mean().sqrt().detach(),
        "goal_mean_rms": goal_stats["mean"].square().mean().sqrt().detach(),
        "local_log_std": local_stats["log_std"].mean().detach(),
        "goal_log_std": goal_stats["log_std"].mean().detach(),
    }


def intact_forward(self, batch, stage, cfg, capture_diagnostics=None):
    if "previous_action" not in batch:
        raise KeyError("Boundary-aware previous_action is required")
    if os.environ.get("INTACT_VALIDATE_BATCH", "1") == "1":
        if not torch.isfinite(batch["action"]).all():
            raise ValueError("Target actions must be finite")
        if not torch.isfinite(batch["previous_action"]).all():
            raise ValueError("Previous actions must be finite")
    layer_flow_regularizer = getattr(self, "layer_flow_regularizer", None)
    layer_flow_weight = float(getattr(self, "layer_flow_weight", 0.0))
    layer_flow_enabled = layer_flow_regularizer is not None and layer_flow_weight > 0
    if layer_flow_enabled:
        self.model.set_diagnostic_capture(True)
    output = self.model.encode(batch)
    embeddings = output["emb"]
    h_cls = output["h_cls"]
    h_repr = output.get("h_repr", h_cls)
    predictions = predict_adjacent_latents(
        self.model, embeddings, output["act_emb"], cfg.history_size
    )
    targets = embeddings[:, 1:]

    pred_loss = (predictions - targets).square().mean()
    projected_latents = embeddings.transpose(0, 1)
    joint_regularizer = getattr(self, "joint_representation_regularizer", None)
    joint_weight = float(getattr(self, "joint_representation_weight", 0.0))
    h_collapse_weight = float(getattr(self, "h_collapse_weight", 0.0))
    h_regularizer = getattr(self, "h_collapse", None)
    if layer_flow_enabled:
        block_outputs = output.get("encoder_block_cls", [])
        expected_blocks = len(self.model._encoder_sdpa_router.layers)
        if len(block_outputs) != expected_blocks:
            raise RuntimeError(
                "C-RVF requires one CLS representation per encoder block: "
                f"expected {expected_blocks}, got {len(block_outputs)}"
            )
        layer_flow = layer_flow_regularizer(block_outputs, embeddings)
        zero = embeddings.new_zeros(())
        sigreg_loss = zero
        sigreg_potential = zero.detach()
        h_collapse = {
            "loss": zero,
            "mean_std": zero.detach(),
            "top_eigen_share": zero.detach(),
            "std_penalty": zero.detach(),
            "spectral_penalty": zero.detach(),
        }
        joint_loss = zero
        joint_health = zero.detach()
        joint_diagnostics = {}
    elif joint_regularizer is not None and joint_weight > 0:
        if getattr(joint_regularizer, "requires_control", False):
            # AC-SFE uses the action observed on each current latent frame as
            # the task-relevance signal; the regularizer handles the final
            # transition alignment explicitly.
            joint = joint_regularizer(
                h_repr, projected_latents, batch["action"]
            )
        else:
            joint = joint_regularizer(h_repr, projected_latents)
        sigreg_loss = joint["sigreg_loss"]
        sigreg_potential = joint["sigreg_potential"]
        h_collapse = {
            "loss": joint["rvp_loss"],
            "mean_std": joint["mean_std"],
            "top_eigen_share": joint["top_eigen_share"],
            "std_penalty": joint["std_penalty"],
            "spectral_penalty": joint["spectral_penalty"],
        }
        joint_loss = joint["loss"]
        joint_health = joint["joint_health"]
        standard_joint_keys = {
            "loss",
            "rvp_loss",
            "sigreg_loss",
            "sigreg_potential",
            "joint_health",
            "mean_std",
            "top_eigen_share",
            "std_penalty",
            "spectral_penalty",
        }
        joint_diagnostics = {
            f"joint_{key}": value
            for key, value in joint.items()
            if key not in standard_joint_keys
            and torch.is_tensor(value)
            and value.numel() == 1
        }
    elif h_regularizer is not None and h_collapse_weight > 0:
        sigreg_loss = self.sigreg(projected_latents)
        sigreg_loss = route_regularizer_gradient(
            sigreg_loss,
            h_repr,
            self.model.projector,
            getattr(self, "sigreg_gradient_route", "normal"),
        )
        sigreg_potential = torch.log1p(sigreg_loss.float()).detach()
        h_collapse = h_regularizer(h_repr)
        joint_loss = embeddings.new_zeros(())
        joint_health = embeddings.new_zeros(())
        joint_diagnostics = {}
    else:
        sigreg_loss = self.sigreg(projected_latents)
        sigreg_loss = route_regularizer_gradient(
            sigreg_loss,
            h_repr,
            self.model.projector,
            getattr(self, "sigreg_gradient_route", "normal"),
        )
        sigreg_potential = torch.log1p(sigreg_loss.float()).detach()
        zero = embeddings.new_zeros(())
        h_collapse = {
            "loss": zero,
            "mean_std": zero.detach(),
            "top_eigen_share": zero.detach(),
            "std_penalty": zero.detach(),
            "spectral_penalty": zero.detach(),
        }
        joint_loss = zero
        joint_health = zero.detach()
        joint_diagnostics = {}
    if not layer_flow_enabled:
        layer_flow = {
            "loss": embeddings.new_zeros(()),
            "radial_loss": embeddings.new_zeros(()),
            "directional_loss": embeddings.new_zeros(()),
            "radial_active_fraction": embeddings.new_zeros(()),
            "directional_active_fraction": embeddings.new_zeros(()),
            "max_radial_drop": embeddings.new_zeros(()),
            "max_directional_drop": embeddings.new_zeros(()),
        }
    intent = paired_intent_action_loss(
        self.model,
        embeddings,
        batch["action"],
        batch["previous_action"],
        cfg,
    )
    loss = (
        float(cfg.loss.forward_weight) * pred_loss
        + (
            joint_weight * joint_loss
            if joint_regularizer is not None and joint_weight > 0
            else float(cfg.loss.sigreg.weight) * sigreg_loss
            + h_collapse_weight * h_collapse["loss"]
        )
        + layer_flow_weight * layer_flow["loss"]
        + intent["action_loss"]
    )

    metrics = {
        "loss": loss,
        "pred_loss": pred_loss,
        "sigreg_loss": sigreg_loss,
        "sigreg_potential": sigreg_potential,
        "h_collapse_loss": h_collapse["loss"],
        "joint_representation_loss": joint_loss,
        "joint_representation_health": joint_health,
        "layer_flow_loss": layer_flow["loss"],
        "layer_flow_radial_loss": layer_flow["radial_loss"],
        "layer_flow_directional_loss": layer_flow["directional_loss"],
        "layer_flow_radial_active_fraction": layer_flow["radial_active_fraction"],
        "layer_flow_directional_active_fraction": layer_flow[
            "directional_active_fraction"
        ],
        "layer_flow_max_radial_drop": layer_flow["max_radial_drop"],
        "layer_flow_max_directional_drop": layer_flow["max_directional_drop"],
        "h_mean_std": h_collapse["mean_std"],
        "h_top_eigen_share": h_collapse["top_eigen_share"],
        **joint_diagnostics,
        **intent,
    }
    self.log_dict(
        {f"{stage}/{key}": value for key, value in metrics.items()},
        on_step=stage == "train",
        on_epoch=True,
        prog_bar=key_in_progress_bar(metrics),
        sync_dist=True,
    )
    # Multitask throughput runs request no diagnostics. Avoid retaining the
    # large intermediate representation tensors until after backward; callers
    # using the three-argument API keep their behavior.
    if capture_diagnostics is None or capture_diagnostics:
        metrics["diagnostic_tensors"] = {
            "h_cls": h_cls,
            "h_patch_mean": output.get("h_patch_mean", h_cls),
            "h_repr": h_repr,
            "z": embeddings,
            "prediction": predictions,
            **{
                f"encoder_block_{index + 1}": value
                for index, value in enumerate(output.get("encoder_block_cls", []))
            },
        }
        if "h_patch_tokens" in output:
            metrics["diagnostic_tensors"]["h_patch_tokens"] = output[
                "h_patch_tokens"
            ]
    return metrics


def route_regularizer_gradient(
    loss: torch.Tensor,
    h_repr: torch.Tensor,
    projector: nn.Module,
    route: str,
) -> torch.Tensor:
    """Keep a regularizer value fixed while selecting its backward recipients.

    The projector-only and ViT-only variants use the exact local vector-Jacobian
    products from the original forward. They therefore do not run BatchNorm a
    second time or alter its running statistics.
    """

    if route == "normal":
        return loss
    if route == "projector_only":
        parameters = [value for value in projector.parameters() if value.requires_grad]
        gradients = torch.autograd.grad(
            loss, parameters, retain_graph=True, allow_unused=True
        )
        surrogate = loss.new_zeros((), dtype=torch.float32)
        for parameter, gradient in zip(parameters, gradients):
            if gradient is not None:
                surrogate = surrogate + (
                    parameter.float() * gradient.detach().float()
                ).sum()
    elif route == "vit_only":
        gradient = torch.autograd.grad(
            loss, h_repr, retain_graph=True, allow_unused=True
        )[0]
        if gradient is None:
            raise RuntimeError("SIGReg has no differentiable path to h_repr")
        surrogate = (h_repr.float() * gradient.detach().float()).sum()
    else:
        raise ValueError(f"Unknown SIGReg gradient route: {route}")
    return loss.detach() + surrogate - surrogate.detach()


def key_in_progress_bar(metrics):
    # Lightning applies this flag to the whole mapping; keep the mapping compact.
    return "loss" in metrics


def build_dataset(cfg):
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    is_lance = str(dataset_name).endswith(".lance")
    if is_lance:
        # Lance already batches random row access efficiently. Preloading every
        # numeric column duplicates gigabytes once the training stack is imported.
        dataset_cfg["keys_to_cache"] = []
    cache_dir = os.environ.get("LOCAL_DATASET_DIR")
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )

    action_mean, action_std = get_column_stats(dataset, "action")
    transforms = [
        get_img_preprocessor("pixels", "pixels", img_size=cfg.img_size)
    ]
    # The JEPA consumes pixels and actions only. Proprio/state remain available
    # in the batch for diagnostics but do not need full-column normalization.
    transforms.append(
        get_column_normalizer(
            dataset,
            "action",
            "action",
            stats=(action_mean, action_std),
            raw_nan_value=0.0,
        )
    )

    with open_dict(cfg):
        action_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")
        cfg.model.action_encoder.input_dim = action_dim
        if cfg.model.intent_actor is not None:
            cfg.model.intent_actor.action_dim = action_dim
    dataset.transform = spt.data.transforms.Compose(*transforms)
    if is_lance:
        getattr(dataset, "_cache", {}).pop("action", None)
        update_fetch_columns = getattr(dataset, "_update_fetch_columns", None)
        if update_fetch_columns is not None:
            update_fetch_columns()
    return PreviousActionDataset(dataset, action_mean, action_std)


@hydra.main(
    version_base=None, config_path="./config/train", config_name="intact_goal"
)
def run(cfg):
    baseline_enabled = single_baseline_enabled(cfg)
    if baseline_enabled:
        configure_single_baseline(cfg)
        configure_training_runtime(int(cfg.seed), global_sdpa='auto')
    else:
        configure_deterministic_math(int(cfg.seed))
    dataset = build_dataset(cfg)
    generator = torch.Generator().manual_seed(cfg.seed)
    train_set, validation_set = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=generator,
    )
    if baseline_enabled:
        train_loader, validation_loader = build_single_loaders(cfg, train_set, validation_set, generator)
    else:
        train_loader = torch.utils.data.DataLoader(
            train_set, **cfg.loader, shuffle=True, drop_last=True, generator=generator)
        validation_loader = torch.utils.data.DataLoader(
            validation_set, **cfg.loader, shuffle=False, drop_last=False)

    model = hydra.utils.instantiate(cfg.model)
    if baseline_enabled and cfg.baseline.gradient_checkpointing:
        model.encoder.gradient_checkpointing_enable()
    if cfg.init_weights_path:
        state = torch.load(cfg.init_weights_path, map_location="cpu")
        model.load_state_dict(state, strict=True)

    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        }
    }
    module = spt.Module(
        model=model,
        sigreg=endpoint_regularizer(cfg),
        forward=partial(intact_forward, cfg=cfg),
        optim=optimizers,
    )
    if baseline_enabled:
        module.joint_representation_regularizer = joint_regularizer(cfg)
        module.joint_representation_weight = 1.0 if cfg.baseline.sadl else 0.0
    data_module = spt.data.DataModule(
        train=train_loader, val=validation_loader
    )

    checkpoint_root = (Path(cfg.baseline.output_cache) / 'checkpoints' if baseline_enabled else
                       Path(swm.data.utils.get_cache_dir(sub_folder="checkpoints")))
    run_dir = checkpoint_root / cfg.output_model_name
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "train_config.yaml")
    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "method": "LeWM" if cfg.variant == 'lewm' else "INTACT",
                "variant": str(cfg.variant),
                "objective": f"{cfg.variant}_objective",
                "intent_mode": cfg.model.intent_mode,
                "feature_layout": (
                    cfg.model.intent_actor.feature_layout
                    if cfg.model.intent_actor is not None
                    else "none"
                ),
                "seed": int(cfg.seed),
                "dataset": str(cfg.data.dataset.name),
                "goal_start": int(cfg.loss.intent.goal_start),
                "local_start": int(cfg.loss.intent.local_start),
                "loss_weights": {
                    "forward": float(cfg.loss.forward_weight),
                    "sigreg": float(cfg.loss.sigreg.weight),
                    "local": float(cfg.loss.intent.local_weight),
                    "goal": float(cfg.loss.intent.goal_weight),
                },
                "previous_action_contract": "boundary_aware_raw_zero_left_pad_v1",
                "initial_previous_action": "raw_zero_then_zscore",
                "sdpa": str(cfg.baseline.encoder_sdpa) if baseline_enabled else "MATH",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    callbacks = ([SingleBaselineCheckpoint(cfg, generator, len(dataset), len(train_set), len(validation_set))]
                 if baseline_enabled else
                 [SaveCkptCallback(run_name=cfg.output_model_name, cfg=cfg.model, epoch_interval=1)])
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        logger=logger,
        num_sanity_val_steps=0,
        enable_checkpointing=baseline_enabled,
    )
    trainer.fit(model=module, datamodule=data_module,
                ckpt_path=cfg.baseline.resume_checkpoint if baseline_enabled else None)


if __name__ == "__main__":
    run()
