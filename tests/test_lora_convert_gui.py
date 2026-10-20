"""LoRA converter tool: the Qwen-Image 2.1 -> ComfyUI / SwarmUI profile always writes the ``transformer.`` Diffusers keys
ComfyUI maps onto Comfy-Org's fused MLP (with the trainer's own keys 2 of the 7 layers per block were skipped)."""

from musubi_tuner_gui import lora_convert_gui as m


def test_qwen21_profile_forces_the_transformer_prefix():
    profile = next(p for p in m.PROFILE_DEFINITIONS if p.key == "qwen21_to_comfy")
    converter = m.LoRAConverter(headless=True, config=None)
    command = converter._build_command(profile, "in.safetensors", "out.safetensors", "")
    assert command[command.index("--target") + 1] == "other"
    assert command[command.index("--diffusers_prefix") + 1] == "transformer"
    # a prefix typed for another profile cannot change it, and the prefix field is hidden for this profile
    command = converter._build_command(profile, "in.safetensors", "out.safetensors", "diffusion_model")
    assert command[command.index("--diffusers_prefix") + 1] == "transformer"
    assert not m._is_diffusers_prefix_supported(profile.label)
    generic = m.PROFILE_BY_LABEL[m.DEFAULT_PROFILE_LABEL]
    assert m._is_diffusers_prefix_supported(generic.label)
    assert "--diffusers_prefix" not in converter._build_command(generic, "in.safetensors", "out.safetensors", "")
