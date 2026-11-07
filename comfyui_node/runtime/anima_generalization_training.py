"""Train one shared adapter with stable raw-gradient calibration.

Time is O(batch_size * model_cost); audit storage is O(trainable_parameters).
The loss callback supplies paired-target supervision. Audits reuse its graphs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor


DEFAULT_INITIAL_WEIGHT = 0.08
DEFAULT_TARGET_RATIO = 0.15
DEFAULT_AUDIT_INTERVAL = 32
DEFAULT_PROBE_ROWS = 4
DEFAULT_EMA_BETA = 0.9
MINIMUM_WEIGHT = 0.01
MAXIMUM_WEIGHT = 1.0
MINIMUM_WEIGHT_FACTOR = 0.5
MAXIMUM_WEIGHT_FACTOR = 2.0
GRADIENT_CLIP_NORM = 1.0
MEASUREMENT_SCOPE = "uniform_rows_accumulated_raw_gradients"


@dataclass(frozen=True)
class CalibrationSettings:
    """Validated scalar controller settings; construction takes O(1) space/time."""

    initial_weight: float
    target_ratio: float
    maximum_ratio: float
    interval: int
    probe_count: int
    ema_beta: float

    @classmethod
    def from_arguments(cls, arguments: Any) -> CalibrationSettings:
        """Read settings without silently clipping the requested ratio."""
        settings = cls(
            initial_weight=float(getattr(arguments, "style_initial_weight", DEFAULT_INITIAL_WEIGHT)),
            target_ratio=float(getattr(arguments, "style_target_grad_ratio", DEFAULT_TARGET_RATIO))
            * float(getattr(arguments, "style_response_weight", 1.0)),
            maximum_ratio=float(getattr(arguments, "style_max_ratio", DEFAULT_TARGET_RATIO)),
            interval=int(getattr(arguments, "style_grad_interval", DEFAULT_AUDIT_INTERVAL)),
            probe_count=int(getattr(arguments, "style_probe_rows", DEFAULT_PROBE_ROWS)),
            ema_beta=float(getattr(arguments, "style_ema_beta", DEFAULT_EMA_BETA)),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        """Reject settings that would make calibration undefined or misleading."""
        scalars = (self.initial_weight, self.target_ratio, self.maximum_ratio, self.ema_beta)
        if not all(math.isfinite(value) for value in scalars):
            raise ValueError("Style calibration settings must be finite")
        if not MINIMUM_WEIGHT <= self.initial_weight <= MAXIMUM_WEIGHT:
            raise ValueError("style_initial_weight must be between 0.01 and 1.0")
        if not 0 < self.target_ratio <= self.maximum_ratio:
            raise ValueError("Effective style_target_grad_ratio must be positive and <= style_max_ratio")
        if self.interval < 1 or self.probe_count < 1:
            raise ValueError("Style audit interval and probe rows must be positive")
        if not 0 <= self.ema_beta < 1:
            raise ValueError("style_ema_beta must be in [0, 1)")


def select_probe_rows(row_count: int, requested_count: int) -> tuple[int, ...]:
    """Select O(probe_count) uniformly spaced rows independently of microbatch size."""
    if row_count < 1 or requested_count < 1:
        raise ValueError("An update and its probe budget must contain at least one row")
    probe_count = min(row_count, requested_count)
    if probe_count == 1:
        return (row_count // 2,)
    return tuple(round(position * (row_count - 1) / (probe_count - 1)) for position in range(probe_count))


def initialize_controller(context: Any, settings: CalibrationSettings) -> tuple[dict[str, Any], bool]:
    """Restore or initialize O(1) serializable calibration state."""
    existing = getattr(context, "style_controller", None)
    if existing:
        controller = dict(existing)
        weight = float(controller["weight"])
        if not math.isfinite(weight) or not MINIMUM_WEIGHT <= weight <= MAXIMUM_WEIGHT:
            raise ValueError(f"Invalid restored style weight: {weight}")
        context.style_controller = controller
        return controller, False
    controller = {
        "weight": settings.initial_weight,
        "base_grad_norm": 0.0,
        "style_grad_norm": 0.0,
        "grad_cosine": 0.0,
        "audit_step": -1,
        "probe_count": 0,
        "status": "initialized",
    }
    context.style_controller = controller
    return controller, True


def accumulate_gradients(destination: list[Tensor | None], gradients: Sequence[Tensor | None]) -> None:
    """Add detached raw gradients using O(parameter_count) time and storage."""
    for position, gradient in enumerate(gradients):
        if gradient is None:
            continue
        detached = gradient.detach().float()
        if destination[position] is None:
            destination[position] = detached.clone()
        else:
            destination[position].add_(detached)


def measure_gradients(base_gradients: Sequence[Tensor | None], style_gradients: Sequence[Tensor | None]) -> tuple[float, float, float]:
    """Measure norms and cosine of summed gradients without projecting them."""
    first = next((gradient for gradient in (*base_gradients, *style_gradients) if gradient is not None), None)
    if first is None:
        return 0.0, 0.0, 0.0
    sums = torch.zeros(3, device=first.device, dtype=torch.float32)
    for base_gradient, style_gradient in zip(base_gradients, style_gradients):
        if base_gradient is not None:
            sums[0] += base_gradient.square().sum()
        if style_gradient is not None:
            sums[1] += style_gradient.square().sum()
        if base_gradient is not None and style_gradient is not None:
            sums[2] += (base_gradient * style_gradient).sum()
    base_square, style_square, dot_product = sums.cpu().tolist()
    if not all(math.isfinite(value) for value in (base_square, style_square, dot_product)):
        raise FloatingPointError("Nonfinite audited style/base gradients")
    base_norm, style_norm = math.sqrt(base_square), math.sqrt(style_square)
    cosine = dot_product / (base_norm * style_norm) if base_norm and style_norm else 0.0
    return base_norm, style_norm, max(-1.0, min(1.0, cosine))


def update_controller(controller: dict[str, Any], measured: tuple[float, float, float], settings: CalibrationSettings) -> None:
    """Update squared-norm EMA and bound changes to the next update's weight."""
    base_norm, style_norm, cosine = measured
    controller.update(base_grad_norm=base_norm, style_grad_norm=style_norm, grad_cosine=cosine)
    if base_norm == 0.0 or style_norm == 0.0:
        controller["status"] = "zero_base_gradient" if base_norm == 0.0 else "zero_style_gradient"
        return
    beta = settings.ema_beta
    base_square = beta * float(controller.get("base_square_ema", base_norm**2)) + (1 - beta) * base_norm**2
    style_square = beta * float(controller.get("style_square_ema", style_norm**2)) + (1 - beta) * style_norm**2
    controller.update(base_square_ema=base_square, style_square_ema=style_square)
    desired_weight = settings.target_ratio * math.sqrt(base_square / style_square)
    previous_weight = float(controller["weight"])
    lower_bound = max(MINIMUM_WEIGHT, previous_weight * MINIMUM_WEIGHT_FACTOR)
    upper_bound = min(MAXIMUM_WEIGHT, previous_weight * MAXIMUM_WEIGHT_FACTOR)
    controller["weight"] = max(lower_bound, min(upper_bound, desired_weight))
    controller["status"] = "calibrated"


class UpdateAccumulator:
    """Accumulate one update and its optional raw-gradient audit."""

    def __init__(self, context: Any, named: Mapping[str, Tensor], rows: Sequence[dict[str, Any]], config: Mapping[str, int]) -> None:
        """Prepare O(parameter_count + batch_size) update storage."""
        if not rows or int(config["microbatch"]) < 1:
            raise ValueError("Training requires nonempty rows and a positive microbatch")
        self.context, self.parameters = context, list(named.values())
        self.settings = CalibrationSettings.from_arguments(context.args)
        self.controller, fresh = initialize_controller(context, self.settings)
        self.weight_used = float(self.controller["weight"])
        self.step_before = int(getattr(context, "global_step", 0))
        self.audit_requested = fresh or self.step_before % self.settings.interval == 0
        self.probe_rows = select_probe_rows(len(rows), self.settings.probe_count) if self.audit_requested else ()
        self.base_gradients: list[Tensor | None] = [None] * len(self.parameters)
        self.style_gradients: list[Tensor | None] = [None] * len(self.parameters)
        self.base_weight = float(getattr(context.args, "style_loss_weight", 1.0))
        self.row_count = len(rows)
        self.times: list[float] = []
        self.predictions: list[Tensor] = []
        self.base_losses: list[Tensor] = []
        self.style_losses: list[Tensor] = []
        self.center_losses: list[Tensor] = []
        self.margin_losses: list[Tensor] = []

    def audit_batch(self, base_losses: Tensor, style_losses: Tensor, start: int) -> None:
        """Reuse selected rows' graph and accumulate their mean raw gradients."""
        selected = [position - start for position in self.probe_rows if start <= position < start + len(base_losses)]
        if not selected:
            return
        denominator = float(len(self.probe_rows))
        base_objective = base_losses[selected].sum() * self.base_weight / denominator
        style_objective = style_losses[selected].sum() / denominator
        base_probe = torch.autograd.grad(base_objective, self.parameters, retain_graph=True, allow_unused=True)
        style_probe = torch.autograd.grad(style_objective, self.parameters, retain_graph=True, allow_unused=True)
        accumulate_gradients(self.base_gradients, base_probe)
        accumulate_gradients(self.style_gradients, style_probe)

    def process_batch(self, batch_rows: Sequence[dict[str, Any]], start: int, capture: bool) -> None:
        """Run one existing denoiser forward and one ordinary total backward."""
        _, times, prediction, base_loss, style_loss, center_loss, margin_loss = self.context.loss_function(self.context, batch_rows)
        self.audit_batch(base_loss, style_loss, start)
        objective = (self.base_weight * base_loss.sum() + self.weight_used * style_loss.sum()) / self.row_count
        objective.backward()
        self.times.extend(times)
        self.base_losses.append(base_loss.detach())
        self.style_losses.append(style_loss.detach())
        self.center_losses.append(center_loss.detach())
        self.margin_losses.append(margin_loss.detach())
        if capture:
            self.predictions.append(prediction.detach().cpu())

    def finish_audit(self) -> None:
        """Calibrate only the following update after every current backward finishes."""
        if not self.audit_requested:
            return
        measured = measure_gradients(self.base_gradients, self.style_gradients)
        update_controller(self.controller, measured, self.settings)
        self.controller["audit_step"] = self.step_before + 1
        self.controller["probe_count"] = len(self.probe_rows)
        self.base_gradients.clear()
        self.style_gradients.clear()

    def loss_metrics(self) -> dict[str, Any]:
        """Serialize actual losses using the constant weight applied this update."""
        base_tensor = torch.cat(self.base_losses).float()
        style_tensor = torch.cat(self.style_losses).float()
        weighted_tensor = style_tensor * self.weight_used
        total_tensor = self.base_weight * base_tensor + weighted_tensor
        if not torch.isfinite(total_tensor).all():
            raise FloatingPointError("Nonfinite training loss")
        return {
            "base_losses": base_tensor.cpu().tolist(),
            "style_response_losses": style_tensor.cpu().tolist(),
            "style_target_loss": float(style_tensor.mean()),
            "center_losses": torch.cat(self.center_losses).float().cpu().tolist(),
            "margin_losses": torch.cat(self.margin_losses).float().cpu().tolist(),
            "weighted_style_losses": weighted_tensor.cpu().tolist(),
            "losses": total_tensor.cpu().tolist(),
            "times": self.times,
        }

    def gradient_metrics(self) -> dict[str, Any]:
        """Label sparse raw-gradient measurements and compatibility aliases clearly."""
        base_norm = float(self.controller["base_grad_norm"])
        style_norm = float(self.controller["style_grad_norm"])
        weighted_norm = self.weight_used * style_norm
        return {
            "base_grad_norm": base_norm, "style_grad_norm": style_norm,
            "style_grad_norm_projected": style_norm, "projected_norm_field_is_raw_alias": True,
            "weighted_style_grad_norm": weighted_norm,
            "style_grad_ratio": weighted_norm / base_norm if base_norm else 0.0,
            "style_grad_ratio_target": self.settings.target_ratio, "style_grad_ratio_estimated": True,
            "style_grad_ratio_pre_projection": style_norm / base_norm if base_norm else 0.0,
            "grad_cosine": float(self.controller["grad_cosine"]),
            "style_weight_effective": self.weight_used, "style_weight_calibrated": self.weight_used,
            "weight_used": self.weight_used, "weight_next": float(self.controller["weight"]),
            "style_grad_audited": self.audit_requested, "style_grad_projected": False,
            "style_grad_projection_applied": False, "style_grad_audit_only": self.audit_requested,
            "measurement_scope": MEASUREMENT_SCOPE, "probe_rows": list(self.probe_rows),
            "last_audit_step": int(self.controller["audit_step"]),
            "last_audit_probe_count": int(self.controller["probe_count"]),
            "style_controller_status": self.controller["status"],
            "style_controller": dict(self.controller),
        }


def apply_optimizer(named: Mapping[str, Tensor], optimizer: torch.optim.Optimizer, capture: bool) -> tuple[float, dict[str, Tensor] | None]:
    """Check connections and clip once before applying one optimizer step."""
    disconnected = [name for name, parameter in named.items() if parameter.grad is None]
    if disconnected:
        raise RuntimeError(f"Disconnected LoRA gradients: {disconnected[:5]}")
    gradients = {name: parameter.grad.detach().cpu().clone() for name, parameter in named.items()} if capture else None
    norm = torch.nn.utils.clip_grad_norm_(list(named.values()), GRADIENT_CLIP_NORM, error_if_nonfinite=True)
    optimizer.step()
    return float(norm), gradients


def run_training_update(context: Any, named: Mapping[str, Tensor], optimizer: torch.optim.Optimizer, rows: Sequence[dict[str, Any]], config: Mapping[str, int], capture: bool = False) -> dict[str, Any]:
    """Apply one fixed-weight update; sparse audits calibrate the next update only."""
    accumulator = UpdateAccumulator(context, named, rows, config)
    context.route.checkpoint_blocks = config["checkpoint_blocks"]
    context.route.use_checkpoint = True
    optimizer.zero_grad(set_to_none=True)
    calls_before = getattr(context.route, "forward_calls", 0)
    microbatch = int(config["microbatch"])
    for start in range(0, len(rows), microbatch):
        accumulator.process_batch(rows[start:start + microbatch], start, capture)
    calls_after = getattr(context.route, "forward_calls", 0)
    if calls_after - calls_before != math.ceil(len(rows) / microbatch):
        raise RuntimeError("Unexpected denoiser call multiplicity")
    losses = accumulator.loss_metrics()
    accumulator.finish_audit()
    gradient_norm, gradients = apply_optimizer(named, optimizer, capture)
    return {
        **losses, **accumulator.gradient_metrics(), "grad_norm": gradient_norm,
        "predictions": torch.cat(accumulator.predictions) if accumulator.predictions else None,
        "gradients": gradients,
    }
