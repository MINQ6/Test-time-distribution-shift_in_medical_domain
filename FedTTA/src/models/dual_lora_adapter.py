from __future__ import annotations

from collections import OrderedDict
import weakref

import torch
import torch.nn as nn
from torch.nn.modules.module import _IncompatibleKeys

from .lora import LoRA


class DualLoRALinear(nn.Module):
    """Frozen base linear layer 위에 local/global LoRA를 더하는 wrapper."""

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int | None = None,
        alpha: int = 16,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.base_layer = base_layer
        target_device = base_layer.weight.device
        lora_dtype = torch.float32
        self.local_lora = LoRA(
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            input_dim=base_layer.in_features,
            output_dim=base_layer.out_features,
        ).to(device=target_device, dtype=lora_dtype)
        self.global_lora = LoRA(
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            input_dim=base_layer.in_features,
            output_dim=base_layer.out_features,
        ).to(device=target_device, dtype=lora_dtype)

        for param in self.base_layer.parameters():
            param.requires_grad = False
        self.global_adapter_weight: float | torch.Tensor = 1.0
        self.local_adapter_weight: float | torch.Tensor = 1.0

        # FedALT mixer: per-token input-conditioned gate over the 2 branches.
        # Branch index order: 0 = Individual (local_lora), 1 = RoW (global_lora).
        # softmax(gate(x)) -> [alpha_indiv, alpha_row]. Disabled by default to keep
        # backward compatibility with the scalar set_adapter_weights path.
        self.use_mixer: bool = False
        self.mixer = nn.Linear(base_layer.in_features, 2, bias=True).to(
            device=target_device, dtype=lora_dtype
        )
        nn.init.zeros_(self.mixer.weight)
        nn.init.zeros_(self.mixer.bias)
        # Buffer for the most recent per-forward mean of alpha_indiv (diagnostics).
        self._last_alpha_indiv_mean: float | None = None

    def set_use_mixer(self, use_mixer: bool) -> None:
        self.use_mixer = bool(use_mixer)

    def set_adapter_weights(
        self,
        *,
        global_weight: float | torch.Tensor = 1.0,
        local_weight: float | torch.Tensor = 1.0,
    ) -> None:
        self.global_adapter_weight = global_weight
        self.local_adapter_weight = local_weight

    @staticmethod
    def _apply_adapter_weight(output: torch.Tensor, weight: float | torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(weight):
            return output * float(weight)
        weight = weight.to(device=output.device, dtype=output.dtype)
        if output.shape[0] != weight.shape[0]:
            if output.shape[0] % weight.shape[0] != 0:
                raise ValueError(
                    f"Adapter weight batch size {weight.shape[0]} is incompatible with output batch size {output.shape[0]}"
                )
            weight = weight.repeat_interleave(output.shape[0] // weight.shape[0], dim=0)
        view_shape = [weight.shape[0]] + [1] * (output.dim() - 1)
        return output * weight.view(*view_shape)

    def forward(self, x):
        base_out = self.base_layer(x)
        x_for_lora = x.to(
            device=self.local_lora.lora_A.weight.device,
            dtype=self.local_lora.lora_A.weight.dtype,
        )
        if self.use_mixer:
            # FedALT per-token gate. alpha = softmax(mixer(x)) over [Individual, RoW].
            gate_logits = self.mixer(x_for_lora)  # [..., 2]
            alpha = torch.softmax(gate_logits, dim=-1)
            alpha_indiv = alpha[..., 0:1]  # Individual (local)
            alpha_row = alpha[..., 1:2]    # RoW (global)
            individual_out = self.local_lora(x_for_lora) * alpha_indiv
            row_out = self.global_lora(x_for_lora) * alpha_row
            lora_out = individual_out + row_out
            self._last_alpha_indiv_mean = float(alpha_indiv.detach().mean())
            return base_out + lora_out.to(dtype=base_out.dtype)

        local_out = self._apply_adapter_weight(self.local_lora(x_for_lora), self.local_adapter_weight)
        global_out = self._apply_adapter_weight(self.global_lora(x_for_lora), self.global_adapter_weight)
        lora_out = local_out + global_out
        return base_out + lora_out.to(dtype=base_out.dtype)


class DualLoRAAdapter(nn.Module):
    """FedDPA 스타일의 local/global dual LoRA adapter."""

    def __init__(
        self,
        model: nn.Module,
        rank: int | None = None,
        alpha: int = 16,
        dropout: float = 0.05,
        target_module_suffixes: tuple[str, ...] | None = None,
    ):
        super().__init__()
        object.__setattr__(self, "_model_ref", weakref.ref(model))
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.target_module_suffixes = target_module_suffixes or ("q_proj", "v_proj")
        self.target_module_names: list[str] = []
        self._wrapped_modules: dict[str, DualLoRALinear] = {}
        self._attach_to_target_modules()

    def _attach_to_target_modules(self) -> None:
        model = self._model_ref()
        if model is None:
            raise RuntimeError("Base model reference is no longer available.")

        for module_name, module in list(model.named_modules()):
            if not module_name or not isinstance(module, nn.Linear):
                continue
            if not module_name.endswith(self.target_module_suffixes):
                continue

            parent_name, child_name = module_name.rsplit(".", 1)
            parent_module = model.get_submodule(parent_name)
            wrapped = DualLoRALinear(
                module,
                rank=self.rank,
                alpha=self.alpha,
                dropout=self.dropout,
            )
            setattr(parent_module, child_name, wrapped)
            self._wrapped_modules[module_name] = wrapped
            self.target_module_names.append(module_name)

    def configure_training(self):
        for param in self.local_parameters():
            param.requires_grad = True
        for param in self.global_parameters():
            param.requires_grad = True

    def configure_global_training(self):
        for param in self.local_parameters():
            param.requires_grad = False
        for param in self.global_parameters():
            param.requires_grad = True

    def configure_local_training(self):
        for param in self.global_parameters():
            param.requires_grad = False
        for param in self.local_parameters():
            param.requires_grad = True

    def configure_fedalt_training(self):
        """FedALT local update: train Individual(local) + mixer, freeze RoW(global)."""
        for param in self.global_parameters():
            param.requires_grad = False
        for param in self.local_parameters():
            param.requires_grad = True
        for param in self.mixer_parameters():
            param.requires_grad = True

    def set_use_mixer(self, use_mixer: bool) -> None:
        for wrapped in self._wrapped_modules.values():
            wrapped.set_use_mixer(use_mixer)

    def mixer_parameters(self):
        for wrapped in self._wrapped_modules.values():
            yield from wrapped.mixer.parameters()

    def mixer_state_dict(self) -> "OrderedDict[str, torch.Tensor]":
        """Per-client mixer weights (kept local, never aggregated)."""
        state = OrderedDict()
        for module_name, wrapped in self._wrapped_modules.items():
            state[f"{module_name}.mixer.weight"] = wrapped.mixer.weight.detach().clone()
            state[f"{module_name}.mixer.bias"] = wrapped.mixer.bias.detach().clone()
        return state

    def load_mixer_state_dict(self, mixer_state: dict[str, torch.Tensor] | None) -> None:
        if mixer_state is None:
            return
        for key, value in mixer_state.items():
            if ".mixer." not in key:
                continue
            module_name, suffix = key.split(".mixer.", 1)
            wrapped = self._wrapped_modules.get(module_name)
            if wrapped is None:
                continue
            target_param = wrapped.mixer.weight if suffix.startswith("weight") else wrapped.mixer.bias
            incoming = value.to(device=target_param.device, dtype=target_param.dtype)
            target_param.data.copy_(incoming)

    def alpha_indiv_mean(self) -> float | None:
        """Mean alpha_indiv over wrapped layers from the most recent mixer forward."""
        vals = [
            wrapped._last_alpha_indiv_mean
            for wrapped in self._wrapped_modules.values()
            if wrapped._last_alpha_indiv_mean is not None
        ]
        if not vals:
            return None
        return float(sum(vals) / len(vals))

    def global_parameters(self):
        for wrapped in self._wrapped_modules.values():
            yield from wrapped.global_lora.parameters()

    def local_parameters(self):
        for wrapped in self._wrapped_modules.values():
            yield from wrapped.local_lora.parameters()

    def set_adapter_weights(
        self,
        *,
        global_weight: float | torch.Tensor = 1.0,
        local_weight: float | torch.Tensor = 1.0,
    ) -> None:
        for wrapped in self._wrapped_modules.values():
            wrapped.set_adapter_weights(
                global_weight=global_weight,
                local_weight=local_weight,
            )

    def get_upload_payload(self):
        global_lora_payload = {}
        for module_name, wrapped in self._wrapped_modules.items():
            global_lora_payload[module_name] = {
                "A": wrapped.global_lora.lora_A.weight.detach().clone().cpu(),
                "B": wrapped.global_lora.lora_B.weight.detach().clone().cpu(),
            }
        return {
            "global_lora": global_lora_payload
        }

    def load_global_lora(self, global_lora_state: dict[str, torch.Tensor] | None):
        if global_lora_state is None:
            return

        for key, value in global_lora_state.items():
            if ".global_lora." not in key:
                continue
            module_name, suffix = key.split(".global_lora.", 1)
            wrapped = self._wrapped_modules.get(module_name)
            if wrapped is None:
                continue
            target_param = getattr(wrapped.global_lora, suffix.removesuffix(".weight")).weight
            incoming = value.to(device=target_param.device, dtype=target_param.dtype)
            target_param.data.copy_(incoming)

    def state_dict(self, *args, **kwargs):
        state = OrderedDict()
        for module_name, wrapped in self._wrapped_modules.items():
            state[f"{module_name}.local_lora.lora_A.weight"] = wrapped.local_lora.lora_A.weight.detach().clone()
            state[f"{module_name}.local_lora.lora_B.weight"] = wrapped.local_lora.lora_B.weight.detach().clone()
            state[f"{module_name}.global_lora.lora_A.weight"] = wrapped.global_lora.lora_A.weight.detach().clone()
            state[f"{module_name}.global_lora.lora_B.weight"] = wrapped.global_lora.lora_B.weight.detach().clone()
        return state

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        missing_keys: list[str] = []
        unexpected_keys: list[str] = []

        expected_keys = set(self.state_dict().keys())
        for key, value in state_dict.items():
            if key not in expected_keys:
                unexpected_keys.append(key)
                continue

            if ".local_lora." in key:
                module_name = key.split(".local_lora.", 1)[0]
                lora_name = "local_lora"
            else:
                module_name = key.split(".global_lora.", 1)[0]
                lora_name = "global_lora"
            wrapped = self._wrapped_modules[module_name]
            param_name = "lora_A" if key.endswith("lora_A.weight") else "lora_B"
            target_param = getattr(getattr(wrapped, lora_name), param_name).weight
            incoming = value.to(device=target_param.device, dtype=target_param.dtype)
            target_param.data.copy_(incoming)

        if strict:
            incoming_keys = set(state_dict.keys())
            missing_keys = sorted(expected_keys - incoming_keys)

        return _IncompatibleKeys(missing_keys, unexpected_keys)
