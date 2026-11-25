"""Sparse top-2 feed-forward replacement used by Mornork."""

from __future__ import annotations

import copy

import torch
from torch import nn


class Top2MoEMLP(nn.Module):
    """A sparse top-k wrapper around copies of a dense feed-forward module.

    At zero noise every expert is an exact copy of base_mlp, and forward uses
    one exact expert path to preserve dense outputs numerically. Call
    enable_routing after perturbing experts to begin sparse top-k dispatch.
    """

    def __init__(
        self,
        base_mlp: nn.Module,
        hidden_size: int,
        num_experts: int = 4,
        top_k: int = 2,
        noise_std: float = 0.0,
    ) -> None:
        super().__init__()
        if num_experts < 1:
            raise ValueError("num_experts must be at least 1")
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be between 1 and num_experts")
        if noise_std < 0:
            raise ValueError("noise_std cannot be negative")

        reference = next(base_mlp.parameters(), None)
        if reference is None:
            raise ValueError("base_mlp must contain at least one parameter")

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        # The zero-noise artifact must reproduce the dense model exactly. Enable
        # sparse routing when experts are perturbed to begin specialization.
        self.preserve_dense_output = noise_std == 0.0
        self.experts = nn.ModuleList()
        for _ in range(num_experts):
            expert = copy.deepcopy(base_mlp)
            if noise_std:
                with torch.no_grad():
                    for parameter in expert.parameters():
                        parameter.add_(torch.randn_like(parameter) * noise_std)
            self.experts.append(expert)

        self.router = nn.Linear(
            hidden_size,
            num_experts,
            bias=False,
            device=reference.device,
            dtype=reference.dtype,
        )
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)

    def enable_routing(self) -> None:
        """Enable sparse dispatch after expert specialization begins."""
        self.preserve_dense_output = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected final dimension {self.hidden_size}, "
                f"got {hidden_states.shape[-1]}"
            )

        original_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, self.hidden_size)
        if flat.shape[0] == 0:
            return hidden_states
        if self.preserve_dense_output:
            return self.experts[0](hidden_states)

        probabilities = self.router(flat).float().softmax(dim=-1)
        selected_weights, selected_indices = probabilities.topk(
            self.top_k, dim=-1
        )
        selected_weights = selected_weights / selected_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(torch.finfo(selected_weights.dtype).tiny)

        # Keep router weights and the expert sum in FP32. Casting each gate to
        # BF16 before multiplication breaks the zero-noise identity across layers.
        output = torch.zeros_like(flat, dtype=torch.float32)
        for expert_index, expert in enumerate(self.experts):
            selected = selected_indices.eq(expert_index)
            rows = selected.any(dim=-1).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue

            gates = (selected_weights * selected).sum(dim=-1).index_select(0, rows)
            expert_input = flat.index_select(0, rows)
            expert_output = expert(expert_input)
            contribution = expert_output.float() * gates.unsqueeze(-1)
            output = output.index_add(0, rows, contribution)

        return output.to(hidden_states.dtype).reshape(original_shape)
