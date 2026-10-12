"""Input-axis DoRA in the sd-scripts LoRA modules used for FLUX.1 and SDXL (sd-scripts/networks/dora.py)."""

from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

SD_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sd-scripts")
sys.path.insert(0, SD_SCRIPTS)

from musubi_tuner.networks.lora_shared import DORA_EPS  # noqa: E402
from networks import lora as lora_sdxl  # noqa: E402
from networks import lora_flux  # noqa: E402
from networks.dora import dora_class, saved_dtype  # noqa: E402


def _relerr(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _dense(weight, down, up, magnitude, scale):
    delta = (up.reshape(up.shape[0], -1) @ down.reshape(down.shape[0], -1)).reshape(weight.shape)
    merged = weight + scale * delta
    norm = merged.transpose(0, 1).reshape(weight.shape[1], -1).norm(dim=1)
    return merged * (magnitude.reshape(-1) / (norm + DORA_EPS)).view(1, -1, *([1] * (weight.dim() - 2)))


def _wrap(base_class, layer, *args, **kwargs):
    torch.manual_seed(0)
    layer.requires_grad_(False)
    module = dora_class(base_class)("lora_unet_layer", layer, 1.0, *args, **kwargs)
    with torch.no_grad():
        ups = module.lora_up if isinstance(module.lora_up, nn.ModuleList) else [module.lora_up]
        for up in ups:
            up.weight.normal_(std=0.05)
        module.dora_scale.mul_(1.0 + 0.05 * torch.randn_like(module.dora_scale))
    module.apply_to()
    return module


def test_flux_linear_matches_the_dense_definition_with_gradients():
    layer = nn.Linear(32, 24)
    module = _wrap(lora_flux.LoRAModule, layer, 4, 2.0)
    x = torch.randn(2, 5, 32)
    out = layer(x)
    ref_up = module.lora_up.weight.detach().clone().requires_grad_(True)
    ref_mag = module.dora_scale.detach().clone().requires_grad_(True)
    ref = F.linear(x, _dense(layer.weight, module.lora_down.weight.detach(), ref_up, ref_mag, module.scale), layer.bias)
    assert _relerr(out, ref) < 1e-5
    grad = torch.randn_like(out)
    out.backward(grad)
    ref.backward(grad)
    assert _relerr(module.lora_up.weight.grad, ref_up.grad) < 1e-4
    assert _relerr(module.dora_scale.grad, ref_mag.grad) < 1e-4


def test_flux_split_qkv_matches_the_fused_definition():
    layer = nn.Linear(16, 30, bias=False)
    module = _wrap(lora_flux.LoRAModule, layer, 2, 2.0, split_dims=[10, 10, 10])
    x = torch.randn(4, 16)
    down = torch.cat([m.weight for m in module.lora_down]).detach()
    up = torch.zeros(30, 6)
    for index, m in enumerate(module.lora_up):
        up[index * 10 : (index + 1) * 10, index * 2 : (index + 1) * 2] = m.weight.detach()
    ref = F.linear(x, _dense(layer.weight, down, up, module.dora_scale.detach(), module.scale))
    assert _relerr(layer(x), ref) < 1e-5


def test_sdxl_conv_matches_the_dense_definition():
    layer = nn.Conv2d(6, 8, 3, padding=1)
    module = _wrap(lora_sdxl.LoRAModule, layer, 3, 3.0)
    assert module.dora_scale.shape == (1, 6, 1, 1)
    x = torch.randn(2, 6, 7, 7)
    ref = F.conv2d(x, _dense(layer.weight, module.lora_down.weight.detach(), module.lora_up.weight.detach(), module.dora_scale.detach(), module.scale), layer.bias, padding=1)
    assert _relerr(layer(x), ref) < 1e-5


def test_layers_sharing_one_input_keep_it_once():
    layers = [nn.Linear(64, 64).requires_grad_(False) for _ in range(3)]
    for index, layer in enumerate(layers):
        dora_class(lora_flux.LoRAModule)(f"lora_unet_l{index}", layer, 1.0, 4, 4.0).apply_to()
    x = torch.randn(32, 64, requires_grad=True)
    kept = set()

    def pack(tensor):
        if tensor.numel() == x.numel():
            kept.add(tensor.untyped_storage().data_ptr())
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        out = sum(layer(x) for layer in layers)
    out.sum().backward()
    assert len(kept) == 1


def test_dora_scale_is_saved_in_fp32():
    assert saved_dtype("lora_unet_x.dora_scale", torch.ones(1), torch.bfloat16) == torch.float32
    assert saved_dtype("lora_unet_x.lora_up.weight", torch.ones(1), torch.bfloat16) == torch.bfloat16


def test_flux_network_uses_dora_and_keeps_the_scale_of_split_modules():
    class DoubleStreamBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.img_attn = nn.Module()
            self.img_attn.qkv = nn.Linear(16, 9216)

    unet = nn.Module()
    unet.double_blocks = nn.ModuleList([DoubleStreamBlock()])
    unet.requires_grad_(False)
    network = lora_flux.create_network(1.0, 4, 4.0, None, [], unet, use_dora="True", split_qkv="True")
    assert type(network.unet_loras[0]).__name__ == "DoRALoRAModule"
    network.apply_to([], unet, apply_text_encoder=False, apply_unet=True)
    state = network.state_dict()
    assert "lora_unet_double_blocks_0_img_attn_qkv.dora_scale" in state
    assert "lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight" in state
    with pytest.raises(ValueError, match="input axis"):
        lora_flux.create_network(1.0, 4, 4.0, None, [], unet, use_dora="True", dora_axis="output")


def test_compiles_whole_inside_activation_checkpointing():
    """sd-scripts compiles blocks that checkpoint their forward: a DoRA forward that stored attributes would make
    Dynamo run such a block eagerly for good."""

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(32, 32)

        def forward(self, x):
            return torch.utils.checkpoint.checkpoint(self.layer, x, use_reentrant=False)

    block = Block()
    module = _wrap(lora_flux.LoRAModule, block.layer, 4, 2.0)
    compiled = torch.compile(block, fullgraph=True, backend="eager")  # a graph break raises
    for _ in range(2):
        compiled(torch.randn(2, 5, 32, requires_grad=True)).sum().backward()
    assert module.dora_scale.grad is not None and module.lora_down.weight.grad is not None
