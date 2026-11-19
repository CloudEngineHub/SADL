"""Representation regularizers: SIGReg-compatible endpoint terms and the SADL capacity bridge."""
from __future__ import annotations

import math

import os

import torch

from torch import nn

import torch.nn.functional as F

def _check_positive_slogdet(sign: torch.Tensor, message: str) -> None:
    """Validate a slogdet sign unless the performance profile disables sync checks."""
    if os.environ.get("INTACT_SKIP_SLOGDET_CHECKS", "0") == "1":
        return
    if not bool((sign > 0).detach()):
        raise FloatingPointError(message)

def _capacity_fast_linalg_enabled() -> bool:
    return os.environ.get(
        "INTACT_CAPACITY_FAST_LINALG",
        os.environ.get("INTACT_SKIP_SLOGDET_CHECKS", "0"),
    ) == "1"

def _capacity_reuse_stats_enabled() -> bool:
    """Enable the opt-in fused capacity-statistics path.

    The default runtime keeps the reference path unchanged.  The
    reuse path is intentionally limited to the SADL reference-excitation mode,
    where the same centered covariance is otherwise rebuilt for each detached
    capacity diagnostic.
    """
    return os.environ.get("INTACT_CAPACITY_REUSE_STATS", "0") == "1"

def _capacity_cholesky(matrix: torch.Tensor) -> torch.Tensor:
    if _capacity_fast_linalg_enabled():
        return torch.linalg.cholesky_ex(matrix, check_errors=False)[0]
    return torch.linalg.cholesky(matrix)

class GaussianQQRegularizer(nn.Module):
    """Match random one-dimensional projections to Gaussian quantiles.

    QQWorld writes a sum over the ordered batch samples. For optimization we
    divide that sum by ``N`` and average over timesteps ``T`` and projection
    directions ``K``. Thus the implementation is a mean over ``(T, N, K)``
    and its scalar weight does not change with the batch size. Unlike the
    characteristic-function objective, every ranked sample receives a direct
    transport gradient toward its Gaussian quantile.
    """

    def __init__(self, num_proj: int = 1024) -> None:
        super().__init__()
        if num_proj < 1:
            raise ValueError("num_proj must be positive")
        self.num_proj = int(num_proj)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3:
            raise ValueError(
                f"GaussianQQRegularizer expects (T, B, D), got {tuple(values.shape)}"
            )
        _, batch_size, dimension = values.shape
        if batch_size < 2:
            raise ValueError("QQ regularization requires at least two samples")
        values32 = values.float()
        directions = F.normalize(
            torch.randn(
                dimension,
                self.num_proj,
                device=values.device,
                dtype=torch.float32,
            ),
            dim=0,
            eps=1e-8,
        )
        ordered = (values32 @ directions).sort(dim=1).values
        probabilities = (
            torch.arange(batch_size, device=values.device, dtype=torch.float32)
            + 0.5
        ) / float(batch_size)
        gaussian = (
            torch.erfinv(2.0 * probabilities - 1.0) * math.sqrt(2.0)
        ).view(1, batch_size, 1)
        # The definition sums over ordered samples; use its batch-normalized
        # form in optimization so the published scalar weight is batch-size
        # independent and comparable across memory budgets.
        return (ordered - gaussian).square().mean().to(values.dtype)

class VISRegularizer(nn.Module):
    """Variance/shape/center decomposition from VISReg for (T, B, D)."""

    def __init__(
        self,
        num_proj: int = 1024,
        scale_weight: float = 1.0,
        shape_weight: float = 1.0,
        center_weight: float = 1.0,
        quantile_rule: str = "midpoint",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if num_proj < 1:
            raise ValueError("num_proj must be positive")
        self.num_proj = int(num_proj)
        self.scale_weight = float(scale_weight)
        self.shape_weight = float(shape_weight)
        self.center_weight = float(center_weight)
        if quantile_rule not in {"midpoint", "official_n_over_n_plus_one"}:
            raise ValueError(f"Unknown VISReg quantile rule: {quantile_rule}")
        self.quantile_rule = quantile_rule
        self.eps = float(eps)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3:
            raise ValueError(
                f"VISRegularizer expects (T, B, D), got {tuple(values.shape)}"
            )
        _, batch_size, dimension = values.shape
        if batch_size < 2:
            raise ValueError("VISReg requires at least two samples")
        values32 = values.float()
        mean = values32.mean(dim=1, keepdim=True)
        centered = values32 - mean
        std = centered.square().mean(dim=1).clamp_min(self.eps).sqrt()
        scale = (std - 1.0).square().mean()
        center = mean.square().mean()

        normalized = centered / std.detach().unsqueeze(1).clamp_min(self.eps)
        directions = F.normalize(
            torch.randn(
                dimension,
                self.num_proj,
                device=values.device,
                dtype=torch.float32,
            ),
            dim=0,
            eps=1e-8,
        )
        ordered = (normalized @ directions).sort(dim=1).values
        indices = torch.arange(
            batch_size, device=values.device, dtype=torch.float32
        )
        if self.quantile_rule == "official_n_over_n_plus_one":
            probabilities = (indices + 1.0) / float(batch_size + 1)
        else:
            probabilities = (indices + 0.5) / float(batch_size)
        gaussian = (
            torch.erfinv(2.0 * probabilities - 1.0) * math.sqrt(2.0)
        ).view(1, batch_size, 1)
        shape = (ordered - gaussian).square().mean()
        loss = (
            self.scale_weight * scale
            + self.shape_weight * shape
            + self.center_weight * center
        )
        return loss.to(values.dtype)

def _aligned_joint_samples(
    h_cls: torch.Tensor, projected_latents: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten matching ``(batch, time)`` samples from the h and z sites."""
    if h_cls.ndim != 3 or projected_latents.ndim != 3:
        raise ValueError("Joint h/z regularizers require rank-3 representations")
    # Encoder readouts are [B, T, D], while projected latents are [T, B, D].
    z_batch_first = projected_latents.transpose(0, 1)
    if h_cls.shape[:2] != z_batch_first.shape[:2]:
        raise ValueError(
            "h_cls and projected_latents must describe the same batch/time samples: "
            f"got {tuple(h_cls.shape)} and {tuple(projected_latents.shape)}"
        )
    return (
        h_cls.reshape(-1, h_cls.size(-1)).float(),
        z_batch_first.reshape(-1, z_batch_first.size(-1)).float(),
    )

def _capacity_spectrum_stats(gram: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        powers = torch.linalg.eigvalsh(gram).clamp(0.0, 1.0)
        return powers.mean(), powers[: max(1, gram.size(0) // 4)].mean()

def _canonical_capacity_fraction(
    h_cls: torch.Tensor,
    projected_latents: torch.Tensor,
    *,
    ridge: float,
    snr: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Finite-SNR capacity retained by the aligned ``h -> z`` channel.

    Regularized CCA removes global scale and orthogonal-coordinate rotation
    before measuring the log-determinant channel capacity.  The isotropic
    ridge intentionally does not claim invariance to arbitrary per-coordinate
    rescaling. The denominator counts all coordinate modes, not just the
    active rank. A rank-r lossless channel therefore approaches r/d, and a
    full-rank lossless channel approaches one only as ridge vanishes. This
    second-order score is not the mutual information of the learned channel.
    ``projected_latents`` may be detached by the caller to route
    the capacity gradient exclusively into the encoder.
    """
    if not 0 < ridge < 1 or snr <= 0 or eps <= 0:
        raise ValueError("Invalid canonical-capacity parameters")
    h, z = _aligned_joint_samples(h_cls, projected_latents)
    if h.size(0) < 2:
        raise ValueError("Canonical capacity requires at least two samples")
    h = h - h.mean(dim=0, keepdim=True)
    z = z - z.mean(dim=0, keepdim=True)
    denominator = float(h.size(0))
    covariance_h = h.T @ h / denominator
    covariance_z = z.T @ z / denominator
    covariance_hz = h.T @ z / denominator
    # A relative ridge must participate in autograd.  Detaching its scale
    # leaves the forward CCA score scale-invariant but introduces a false
    # radial gradient that rewards H-norm inflation when this channel is the
    # active min-cut.
    scale_h = covariance_h.diagonal().mean().clamp_min(eps)
    scale_z = covariance_z.diagonal().mean().clamp_min(eps)
    eye_h = torch.eye(h.size(1), device=h.device, dtype=h.dtype)
    eye_z = torch.eye(z.size(1), device=z.device, dtype=z.dtype)
    chol_h = _capacity_cholesky(covariance_h + ridge * scale_h * eye_h)
    chol_z = _capacity_cholesky(covariance_z + ridge * scale_z * eye_z)
    left_whitened = torch.linalg.solve_triangular(
        chol_h, covariance_hz, upper=False
    )
    canonical = torch.linalg.solve_triangular(
        chol_z, left_whitened.T, upper=False
    ).T
    if canonical.size(0) <= canonical.size(1):
        gram = canonical @ canonical.T
    else:
        gram = canonical.T @ canonical
    rank = gram.size(0)
    identity = torch.eye(rank, device=gram.device, dtype=gram.dtype)
    sign, logdet = torch.linalg.slogdet(identity + snr * gram)
    _check_positive_slogdet(
        sign, "Canonical-capacity matrix is not positive definite"
    )
    maximum = rank * math.log1p(snr)
    fraction = (logdet / maximum).clamp(0.0, 1.0)
    mean_power, weakest_quartile = _capacity_spectrum_stats(gram)
    return fraction, mean_power, weakest_quartile

def _capacity_stats_from_flat(flat: torch.Tensor) -> dict[str, torch.Tensor]:
    """Build the shared h/z statistics used by the opt-in SADL fast path.

    ``flat`` is already float32 and sample-major.  The centering and covariance
    operations deliberately match the helpers above, including the
    population denominator ``N`` used by capacity scores.
    """
    if flat.ndim != 2:
        raise ValueError("Capacity statistics require a rank-2 flattened tensor")
    if flat.size(0) < 2:
        raise ValueError("Capacity statistics require at least two samples")
    centered = flat - flat.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / float(flat.size(0))
    return {"flat": flat, "centered": centered, "covariance": covariance}

def _canonical_capacity_fraction_from_stats(
    h_stats: dict[str, torch.Tensor],
    z_stats: dict[str, torch.Tensor],
    *,
    ridge: float,
    snr: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Canonical capacity using stats already built for the same h/z samples."""
    h = h_stats["centered"]
    z = z_stats["centered"]
    covariance_h = h_stats["covariance"]
    covariance_z = z_stats["covariance"]
    if not 0 < ridge < 1 or snr <= 0 or eps <= 0:
        raise ValueError("Invalid canonical-capacity parameters")
    if h.size(0) != z.size(0):
        raise ValueError("Canonical capacity requires aligned h/z samples")
    covariance_hz = h.T @ z / float(h.size(0))
    scale_h = covariance_h.diagonal().mean().clamp_min(eps)
    scale_z = covariance_z.diagonal().mean().clamp_min(eps)
    eye_h = torch.eye(h.size(1), device=h.device, dtype=h.dtype)
    eye_z = torch.eye(z.size(1), device=z.device, dtype=z.dtype)
    chol_h = _capacity_cholesky(covariance_h + ridge * scale_h * eye_h)
    chol_z = _capacity_cholesky(covariance_z + ridge * scale_z * eye_z)
    left_whitened = torch.linalg.solve_triangular(
        chol_h, covariance_hz, upper=False
    )
    canonical = torch.linalg.solve_triangular(
        chol_z, left_whitened.T, upper=False
    ).T
    if canonical.size(0) <= canonical.size(1):
        gram = canonical @ canonical.T
    else:
        gram = canonical.T @ canonical
    rank = gram.size(0)
    identity = torch.eye(rank, device=gram.device, dtype=gram.dtype)
    sign, logdet = torch.linalg.slogdet(identity + snr * gram)
    _check_positive_slogdet(
        sign, "Canonical-capacity matrix is not positive definite"
    )
    maximum = rank * math.log1p(snr)
    fraction = (logdet / maximum).clamp(0.0, 1.0)
    mean_power, weakest_quartile = _capacity_spectrum_stats(gram)
    return fraction, mean_power, weakest_quartile

def _marginal_capacity_fraction(
    projected_latents: torch.Tensor,
    *,
    snr: float,
    eps: float,
) -> torch.Tensor:
    """Scale-free finite-SNR spectral capacity of the projected marginal.

    The covariance is normalized to unit mean eigenvalue.  Concavity of
    ``log(1 + snr * x)`` then bounds the normalized log-determinant by one,
    with equality only for an isotropic covariance.  A zero-variance marginal
    has zero capacity.
    """
    if snr <= 0 or eps <= 0:
        raise ValueError("Invalid marginal-capacity parameters")
    if projected_latents.ndim != 3:
        raise ValueError("Marginal capacity requires a rank-3 representation")
    with torch.autocast(device_type=projected_latents.device.type, enabled=False):
        values = projected_latents.transpose(0, 1).reshape(
            -1, projected_latents.size(-1)
        ).float()
        if values.size(0) < 2:
            raise ValueError("Marginal capacity requires at least two samples")
        centered = values - values.mean(dim=0, keepdim=True)
        covariance = centered.T @ centered / float(values.size(0))
        scale = covariance.diagonal().mean().clamp_min(eps)
        normalized = covariance / scale
        identity = torch.eye(
            normalized.size(0), device=normalized.device, dtype=normalized.dtype
        )
        sign, logdet = torch.linalg.slogdet(identity + snr * normalized)
        _check_positive_slogdet(
            sign, "Marginal-capacity matrix is not positive definite"
        )
        maximum = normalized.size(0) * math.log1p(snr)
        return (logdet / maximum).clamp(0.0, 1.0)

def _marginal_capacity_fraction_from_stats(
    stats: dict[str, torch.Tensor], *, snr: float, eps: float
) -> torch.Tensor:
    """Marginal capacity using a precomputed population covariance."""
    if snr <= 0 or eps <= 0:
        raise ValueError("Invalid marginal-capacity parameters")
    covariance = stats["covariance"]
    with torch.autocast(device_type=covariance.device.type, enabled=False):
        scale = covariance.diagonal().mean().clamp_min(eps)
        normalized = covariance / scale
        identity = torch.eye(
            normalized.size(0), device=normalized.device, dtype=normalized.dtype
        )
        sign, logdet = torch.linalg.slogdet(identity + snr * normalized)
        _check_positive_slogdet(
            sign, "Marginal-capacity matrix is not positive definite"
        )
        maximum = normalized.size(0) * math.log1p(snr)
        return (logdet / maximum).clamp(0.0, 1.0)

def _relative_excitation_capacity_fraction(
    values: torch.Tensor,
    *,
    snr: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Finite-SNR capacity relative to total uncentered feature energy.

    Let ``A = Cov(H) / E[H^2]``. Its mean eigenvalue is the fraction of
    feature energy carried by centered sample variation, so it falls when a
    shared mean consumes the representation energy as well as when covariance
    loses rank. Away from the numerical energy floor, the normalized log
    determinant is globally scale invariant, orthogonally invariant, and
    bounded by one through log concavity.
    """
    if snr <= 0 or eps <= 0:
        raise ValueError("Invalid relative-excitation capacity parameters")
    if values.ndim != 3:
        raise ValueError("Relative-excitation capacity requires rank-3 values")
    with torch.autocast(device_type=values.device.type, enabled=False):
        flat = values.reshape(-1, values.size(-1)).float()
        if flat.size(0) < 2:
            raise ValueError(
                "Relative-excitation capacity requires at least two samples"
            )
        centered = flat - flat.mean(dim=0, keepdim=True)
        covariance = centered.T @ centered / float(flat.size(0))
        total_energy = flat.square().mean().clamp_min(eps)
        relative_covariance = covariance / total_energy
        identity = torch.eye(
            relative_covariance.size(0),
            device=relative_covariance.device,
            dtype=relative_covariance.dtype,
        )
        sign, logdet = torch.linalg.slogdet(
            identity + snr * relative_covariance
        )
        _check_positive_slogdet(
            sign, "Relative-excitation capacity matrix is not positive definite"
        )
        maximum = relative_covariance.size(0) * math.log1p(snr)
        capacity = (logdet / maximum).clamp(0.0, 1.0)
        centered_energy_fraction = (
            covariance.diagonal().mean() / total_energy
        ).clamp(0.0, 1.0)
    return capacity, centered_energy_fraction

def _relative_excitation_capacity_fraction_from_stats(
    stats: dict[str, torch.Tensor], *, snr: float, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Relative-excitation capacity using precomputed covariance statistics."""
    if snr <= 0 or eps <= 0:
        raise ValueError("Invalid relative-excitation capacity parameters")
    covariance = stats["covariance"]
    flat = stats["flat"]
    with torch.autocast(device_type=covariance.device.type, enabled=False):
        total_energy = flat.square().mean().clamp_min(eps)
        relative_covariance = covariance / total_energy
        identity = torch.eye(
            relative_covariance.size(0),
            device=relative_covariance.device,
            dtype=relative_covariance.dtype,
        )
        sign, logdet = torch.linalg.slogdet(identity + snr * relative_covariance)
        _check_positive_slogdet(
            sign, "Relative-excitation capacity matrix is not positive definite"
        )
        maximum = relative_covariance.size(0) * math.log1p(snr)
        capacity = (logdet / maximum).clamp(0.0, 1.0)
        centered_energy_fraction = (
            covariance.diagonal().mean() / total_energy
        ).clamp(0.0, 1.0)
    return capacity, centered_energy_fraction

class CanonicalCapacityMatching(nn.Module):
    """Match internal channel capacity to an output-distribution energy.

    Define ``q_z = exp(-D(z) / tau_D)`` for any non-negative distribution
    energy ``D`` and let ``q_ch`` be the normalized finite-SNR canonical
    capacity of the sample-aligned ``h -> z`` channel. A healthy projected
    marginal must not conceal a less healthy upstream channel, so

        L = alpha D(z) + beta [stopgrad(q_z) - q_ch]_+^2.

    The one-sided constraint is inactive when the channel is already at least
    as healthy as the projected marginal. ``tau_D`` only calibrates the
    dimensionless health score; it does not rescale the output loss gradient.
    In all ``reference_*`` modes, the energy temperature is replaced by the
    regulator's own complete-collapse energy ``D_0 = D(z == 0)``.  The ratio
    ``D(z) / D_0`` is dimensionless and makes the bridge comparable across
    output energies without changing their optimization weight.

    ``reference_excess_free_energy`` instead uses the uniquely endpoint-
    calibrated affine health ``q_z = clip(1 - D(z) / D_0, 0, 1)``.  Its bridge
    is therefore zero at complete output collapse and grows only as the output
    regulator establishes structure.  ``reference_free_energy`` retains the
    stronger exponential recovery bias in additive log units.  In ``absolute``
    mode, ``q_z`` is replaced by one; this is the matched variant for
    unconditional capacity maximization.

    ``reference_excitation_free_energy`` additionally treats the encoder and
    projector as a serial channel. Its upstream cut is normalized by total
    uncentered feature energy, so mean-dominated tiny noise cannot masquerade
    as healthy high-rank variation. The bridge compares the weakest upstream
    cut with the endpoint-calibrated output demand in additive free-energy
    units.

    ``excitation_mincut_free_energy`` removes the surrogate output-health
    calibration altogether. It applies the same excitation capacity to the
    observed projected representation and requires the encoder/projector
    min-cut to support only that realized downstream capacity:

        L_bridge = [log C_exc(z) - log min(C_exc(h), C_tr(h, sg(z)))]_+^2.

    This is a data-processing-inspired proxy constraint, not a data-processing
    theorem: excitation eigenvalues and canonical correlation powers have
    different normalizations. Even a lossless low-rank identity channel can
    activate the bridge. It closes at joint collapse. Detached endpoints block
    direct bridge gradients into z/projector, but encoder updates can still
    change their future outputs.

    ``bregman_mincut_free_energy`` uses the same realized excitation demand
    and min-cut supply, but replaces the squared logarithmic gap by the
    one-sided intensive Bregman free-energy density generated by negative
    entropy,

        B_phi(R, C) = (R + eps) log((R + eps) / (C + eps)) - R + C,

    for ``R > C`` and zero otherwise, where
    ``R = d log(1 + snr) c_req`` and
    ``C = d log(1 + snr) c_cut`` are unnormalized log-det proxy quantities,
    not actual Shannon capacities of the nonlinear model. This is the generalized-KL free
    energy of an excitation demand ``R`` relative to supply ``C``. The
    extensive divergence is divided by the maximum channel capacity
    ``S = d log(1 + snr)`` before weighting. This makes the optimization
    potential intensive for fixed normalized scores (up to the epsilon floor)
    while preserving its minimum and relative-shortage force. Its supply
    derivative is 1-(R+eps)/(C+eps), so it is less singular than a squared
    log gap but is not a bounded restoring force near zero supply.
    """

    def __init__(
        self,
        output_regularizer: nn.Module,
        *,
        output_weight: float = 1.5,
        energy_temperature: float = 1.0,
        bridge_weight: float = 0.09,
        ridge: float = 1e-3,
        snr: float = 9.0,
        mode: str = "matching",
        demand_alpha: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if output_weight <= 0 or bridge_weight < 0 or energy_temperature <= 0:
            raise ValueError(
                "CCM weights must be non-negative and output scale/temperature positive"
            )
        if mode not in {
            "matching",
            "reference_matching",
            "reference_free_energy",
            "reference_excess_free_energy",
            "reference_excitation_free_energy",
            "reference_excitation_only_free_energy",
            "excitation_mincut_free_energy",
            "bregman_mincut_free_energy",
            "absolute",
            "conservation",
            "path_conservation",
            "path_free_energy",
            "mincut_free_energy",
            "free_energy",
            "partition_free_energy",
        }:
            raise ValueError(f"Unknown canonical-capacity mode: {mode}")
        if not 0.0 <= demand_alpha <= 1.0:
            raise ValueError("demand_alpha must lie in [0, 1]")
        self.output_regularizer = output_regularizer
        self.output_weight = float(output_weight)
        self.energy_temperature = float(energy_temperature)
        self.bridge_weight = float(bridge_weight)
        self.ridge = float(ridge)
        self.snr = float(snr)
        self.mode = mode
        self.demand_alpha = float(demand_alpha)
        self.eps = float(eps)
        self.register_buffer(
            "_output_collapse_reference",
            torch.tensor(float("nan"), dtype=torch.float32),
            persistent=False,
        )
        self._collapse_reference_valid = False

    def _apply(self, fn, recurse=True):
        self._collapse_reference_valid = False
        return super()._apply(fn, recurse=recurse)

    def _collapse_reference(self, values: torch.Tensor) -> torch.Tensor:
        """Return ``D(0)`` without perturbing the training RNG stream."""
        cached = self._output_collapse_reference
        fast_linalg = _capacity_fast_linalg_enabled()
        if fast_linalg and self._collapse_reference_valid:
            return cached
        if bool(torch.isfinite(cached).detach()) and bool((cached > self.eps).detach()):
            self._collapse_reference_valid = True
            return cached
        devices = (
            [values.device.index]
            if values.device.type == "cuda" and values.device.index is not None
            else []
        )
        with torch.no_grad(), torch.random.fork_rng(devices=devices, enabled=True):
            collapsed = torch.zeros(
                values.shape,
                device=values.device,
                dtype=torch.float32,
            )
            reference = self.output_regularizer(collapsed)
            if isinstance(reference, dict):
                reference = reference["loss"]
            reference = reference.float().reshape(())
        if not bool(torch.isfinite(reference).detach()) or not bool(
            (reference > self.eps).detach()
        ):
            raise ValueError(
                "reference_matching requires a finite positive complete-collapse "
                f"energy D(0), got {reference.item():.6g}"
            )
        self._output_collapse_reference.copy_(reference.detach())
        self._collapse_reference_valid = True
        return self._output_collapse_reference

    def forward(
        self, h_cls: torch.Tensor, projected_latents: torch.Tensor,
        *, output_loss: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        reuse_stats = (
            _capacity_reuse_stats_enabled()
            and self.mode == "reference_excitation_free_energy"
        )
        h_stats = None
        z_stats = None
        with torch.autocast(device_type=projected_latents.device.type, enabled=False):
            if reuse_stats:
                h_flat, z_flat = _aligned_joint_samples(
                    h_cls.float(), projected_latents.detach().float()
                )
                h_stats = _capacity_stats_from_flat(h_flat)
                z_stats = _capacity_stats_from_flat(z_flat)
                capacity, mean_power, weak_power = (
                    _canonical_capacity_fraction_from_stats(
                        h_stats,
                        z_stats,
                        ridge=self.ridge,
                        snr=self.snr,
                        eps=self.eps,
                    )
                )
            else:
                capacity, mean_power, weak_power = _canonical_capacity_fraction(
                    h_cls.float(),
                    projected_latents.detach().float(),
                    ridge=self.ridge,
                    snr=self.snr,
                    eps=self.eps,
                )
        if output_loss is None:
            output_loss = self.output_regularizer(projected_latents)
        if self.mode in {
            "reference_matching",
            "reference_free_energy",
            "reference_excess_free_energy",
            "reference_excitation_free_energy",
            "reference_excitation_only_free_energy",
        }:
            output_energy_scale = self._collapse_reference(projected_latents)
        else:
            output_energy_scale = output_loss.new_tensor(
                self.energy_temperature, dtype=torch.float32
            )
        output_health = (
            torch.exp(-output_loss.float() / output_energy_scale)
            .detach()
            .clamp(0.0, 1.0)
        )
        with torch.no_grad():
            if reuse_stats:
                output_capacity = _marginal_capacity_fraction_from_stats(
                    z_stats, snr=self.snr, eps=self.eps
                )
            else:
                output_capacity = _marginal_capacity_fraction(
                    projected_latents.detach(), snr=self.snr, eps=self.eps
                )
        if self.mode in {
            "reference_excitation_free_energy",
            "reference_excitation_only_free_energy",
            "excitation_mincut_free_energy",
            "bregman_mincut_free_energy",
        }:
            if reuse_stats:
                input_capacity, input_centered_energy_fraction = (
                    _relative_excitation_capacity_fraction_from_stats(
                        h_stats, snr=self.snr, eps=self.eps
                    )
                )
            else:
                input_capacity, input_centered_energy_fraction = (
                    _relative_excitation_capacity_fraction(
                        h_cls.float(), snr=self.snr, eps=self.eps
                    )
                )
        elif self.mode in {
            "path_conservation",
            "path_free_energy",
            "mincut_free_energy",
        }:
            # End-to-end useful throughput requires both a rich upstream
            # marginal and an aligned h->z channel. A high canonical
            # correlation inside a jointly collapsed subspace is therefore
            # insufficient: either factor can expose the bottleneck.
            input_capacity = _marginal_capacity_fraction(
                h_cls.float(), snr=self.snr, eps=self.eps
            )
        else:
            with torch.no_grad():
                input_capacity = _marginal_capacity_fraction(
                    h_cls.detach(), snr=self.snr, eps=self.eps
                )
        if self.mode not in {
            "reference_excitation_free_energy",
            "reference_excitation_only_free_energy",
            "excitation_mincut_free_energy",
            "bregman_mincut_free_energy",
        }:
            with torch.no_grad():
                _, input_centered_energy_fraction = (
                    _relative_excitation_capacity_fraction(
                        h_cls.detach(), snr=self.snr, eps=self.eps
                    )
                )
        with torch.no_grad():
            # Reduce over [time, batch] in this order.
            # Canonical/marginal z stats are sample-aligned [batch, time], and
            # reusing their covariance would change floating-point summation.
            output_excitation_capacity, output_centered_energy_fraction = (
                _relative_excitation_capacity_fraction(
                    projected_latents.detach(), snr=self.snr, eps=self.eps
                )
            )
        path_capacity = input_capacity * capacity
        path_free_energy = -torch.log(path_capacity.clamp_min(self.eps))
        mincut_capacity = torch.minimum(input_capacity, capacity)
        mincut_free_energy = -torch.log(mincut_capacity.clamp_min(self.eps))
        output_capacity_free_energy = -torch.log(
            output_capacity.clamp_min(self.eps)
        )
        channel_free_energy = -torch.log(capacity.clamp_min(self.eps))
        partition_free_energy = math.log1p(self.snr) * (1.0 - capacity)
        output_free_energy = (output_loss.float() / output_energy_scale).detach()
        bridge_loss_override = None
        bregman_free_energy_extensive = None
        if self.mode in {"matching", "reference_matching"}:
            target_health = output_health
            capacity_gap = F.relu(target_health - capacity)
        elif self.mode == "conservation":
            target_health = output_capacity
            capacity_gap = F.relu(target_health - capacity)
        elif self.mode == "path_conservation":
            target_health = output_capacity
            capacity_gap = F.relu(target_health - path_capacity)
        elif self.mode == "path_free_energy":
            # Serial representation stages add resistance in log-capacity
            # units. This relative deficit retains a strong restoring
            # gradient when both the upstream marginal and channel collapse.
            target_health = output_capacity
            capacity_gap = F.relu(
                path_free_energy - output_capacity_free_energy
            )
        elif self.mode == "mincut_free_energy":
            # This min-cut proxy targets the active second-order bottleneck.
            # Its differently normalized marginal/channel scores do not imply
            # a Shannon data-processing theorem, including for linear models.
            target_health = output_capacity
            capacity_gap = F.relu(
                mincut_free_energy - output_capacity_free_energy
            )
        elif self.mode in {"free_energy", "reference_free_energy"}:
            # C >= exp(-VIS) is exactly -log(C) <= VIS. Penalizing the
            # violation in log space compares channel resistance and output
            # free energy in their additive, dimensionless units. The
            # reference-calibrated form uses D(z)/D(0), so multiplying the
            # output energy by any positive constant leaves this bridge exact.
            target_health = output_health
            capacity_gap = F.relu(channel_free_energy - output_free_energy)
        elif self.mode == "reference_excess_free_energy":
            # Normalize both deficits by their complete-collapse references.
            # The channel excess log-partition energy is 1-C_bar, while the
            # output collapse level is D(z)/D(0). The bridge acts only when the
            # upstream channel is more degraded than the projected marginal.
            target_health = (1.0 - output_free_energy).clamp(0.0, 1.0)
            capacity_gap = F.relu(1.0 - capacity - output_free_energy)
        elif self.mode in {
            "reference_excitation_free_energy",
            "reference_excitation_only_free_energy",
        }:
            # The output regularizer first establishes a nonzero demand. The
            # encoder bridge then opens only if either source excitation or
            # h->z transmission is insufficient. Adding eps symmetrically
            # makes the bridge exactly inactive at complete output collapse.
            external_demand = (1.0 - output_free_energy).clamp(0.0, 1.0)
            attainable_demand = output_excitation_capacity
            target_health = (
                external_demand.pow(1.0 - self.demand_alpha)
                * attainable_demand.pow(self.demand_alpha)
            )
            path_demand_free_energy = -torch.log(target_health + self.eps)
            bridge_supply = (
                input_capacity
                if self.mode == "reference_excitation_only_free_energy"
                else mincut_capacity
            )
            capacity_gap = F.relu(
                -torch.log(bridge_supply + self.eps) - path_demand_free_energy
            )
        elif self.mode == "excitation_mincut_free_energy":
            # The realized endpoint avoids an explicit D(z)->health map.
            # This proxy may still penalize healthy finite-rank identity
            # channels because excitation and CCA use different spectra.
            target_health = output_excitation_capacity
            capacity_gap = F.relu(
                torch.log(output_excitation_capacity + self.eps)
                - torch.log(mincut_capacity + self.eps)
            )
        elif self.mode == "bregman_mincut_free_energy":
            # The negative-entropy Bregman divergence is the generalized KL
            # free energy for non-negative excitation capacities. It is
            # extensive in the realized demand, has its unique minimum at
            # supply=demand, and remains exactly inactive when the min-cut
            # already supports the endpoint. All endpoint terms are detached.
            target_health = output_excitation_capacity
            capacity_gap = F.relu(target_health - mincut_capacity)
            capacity_nats_scale = target_health.new_tensor(
                projected_latents.size(-1) * math.log1p(self.snr)
            )
            demand = capacity_nats_scale * target_health + self.eps
            supply = capacity_nats_scale * mincut_capacity + self.eps
            bregman_free_energy_extensive = (
                demand * torch.log(demand / supply)
                - demand
                + supply
            ).clamp_min(0.0)
            bregman_free_energy_density = (
                bregman_free_energy_extensive / capacity_nats_scale
            )
            bridge_loss_override = torch.where(
                target_health > mincut_capacity,
                bregman_free_energy_density,
                torch.zeros_like(bregman_free_energy_density),
            )
        elif self.mode == "partition_free_energy":
            # This is the per-mode log-partition deficit between an ideal
            # lossless channel and the measured finite-SNR channel:
            #   log(1+snr) - logdet(I+snr RR^T)/r.
            # It is additive across canonical modes and has a fixed zero at
            # perfect transmission, so it can be compared directly with the
            # output distribution energy without an extra score transform.
            target_health = output_health
            capacity_gap = F.relu(partition_free_energy - output_free_energy)
        else:
            target_health = capacity.new_ones(())
            capacity_gap = F.relu(target_health - capacity)
        bridge_loss = (
            capacity_gap.square()
            if bridge_loss_override is None
            else bridge_loss_override
        )
        weighted_output = self.output_weight * output_loss
        weighted_bridge = self.bridge_weight * bridge_loss
        loss = weighted_output + weighted_bridge
        return {
            "loss": loss,
            "rvp_loss": weighted_bridge,
            "sigreg_loss": output_loss,
            "sigreg_potential": weighted_output,
            "joint_health": (-loss.detach()).exp(),
            "mean_std": capacity.detach(),
            "top_eigen_share": mean_power.detach(),
            "std_penalty": capacity_gap.detach(),
            "spectral_penalty": bridge_loss.detach(),
            "canonical_capacity": capacity.detach(),
            "canonical_mean_power": mean_power.detach(),
            "canonical_weak_quartile_power": weak_power.detach(),
            "capacity_gap": capacity_gap.detach(),
            "capacity_bregman_free_energy": (
                bridge_loss.detach()
                if self.mode == "bregman_mincut_free_energy"
                else bridge_loss.detach().new_zeros(())
            ),
            "capacity_bregman_free_energy_extensive": (
                bregman_free_energy_extensive.detach()
                if self.mode == "bregman_mincut_free_energy"
                else bridge_loss.detach().new_zeros(())
            ),
            "capacity_nats_scale": bridge_loss.detach().new_full((),
                projected_latents.size(-1) * math.log1p(self.snr)
            ),
            "bridge_active": (capacity_gap > 0).to(capacity_gap.dtype).detach(),
            "output_health": output_health.detach(),
            "output_capacity": output_capacity.detach(),
            "output_excitation_capacity": output_excitation_capacity.detach(),
            "output_centered_energy_fraction": (
                output_centered_energy_fraction.detach()
            ),
            "input_capacity": input_capacity.detach(),
            "input_centered_energy_fraction": (
                input_centered_energy_fraction.detach()
            ),
            "path_capacity": path_capacity.detach(),
            "path_free_energy": path_free_energy.detach(),
            "mincut_capacity": mincut_capacity.detach(),
            "mincut_free_energy": mincut_free_energy.detach(),
            "output_capacity_free_energy": output_capacity_free_energy.detach(),
            "target_capacity": target_health.detach(),
            "external_demand": (
                (1.0 - output_free_energy).clamp(0.0, 1.0).detach()
            ),
            "attainable_demand": output_excitation_capacity.detach(),
            "demand_alpha": output_loss.new_full((), self.demand_alpha).detach(),
            "channel_free_energy": channel_free_energy.detach(),
            "partition_free_energy": partition_free_energy.detach(),
            "output_free_energy": output_free_energy.detach(),
            "output_regularizer_loss": output_loss.detach(),
            "output_collapse_reference": self._output_collapse_reference.detach(),
            "output_energy_scale": output_energy_scale.detach(),
            # Alias of the bridge term kept for metric compatibility.
            "vis_loss": output_loss.detach(),
            "output_energy_temperature": output_loss.new_full((),
                self.energy_temperature
            ).detach(),
        }

