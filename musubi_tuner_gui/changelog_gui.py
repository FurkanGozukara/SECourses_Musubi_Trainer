import gradio as gr

# Each entry: (accordion title, markdown body).
# Newest first. The newest entry is rendered open by default.
CHANGELOG_ENTRIES = [
    (
        "V32.1 — 2 August 2026 — Dataset Repeat-Count Safety + Notifications",
        """
**📁 The dataset TOML generator now tells you when a dataset folder is missing the repeat-count prefix instead of silently counting it once.**

- Dataset folders are expected to be named `N_name` (e.g. `5_ohwx` = 5 repeats of concept `ohwx` per epoch). If you forget the `N_` prefix, the folder still trains: the repeat count **defaults to 1 and the generator now shows an `[INFO]` message** telling you the prefix was forgotten, that the folder was counted as 1 repeat, and how to rename it to set repeats explicitly.
- A repeat count of `0` (e.g. `0_ohwx`) would make the trainer skip that dataset entirely, so it is now **clamped to 1 repeat with a `[WARNING]`** instead of writing `num_repeats = 0` into the TOML.
- Ambiguous number-only folder names (e.g. `5` or `5_`) are detected and counted as 1 repeat with a clear explanation.
- Every generated dataset now reports its final repeat count in the status output (`[OK] Added ... with num_repeats=N`), so you can verify repeats at a glance before training.
- Applies to **every training tab** that generates dataset TOMLs from folder structures (Qwen, WAN, LTX-2, FLUX, FLUX.2, Z Image, Krea 2, Ideogram 4, FLUX Klein, ...). Folder-name caption generation is unchanged and now shares one robust parser everywhere, with new automated tests covering the edge cases.
""",
    ),
    (
        "V32.0 — 1 August 2026 — LTX 2.3 (22B) / LTX-2 (19B) Video Training",
        """
**🎬 Full LTX-2 / LTX-2.3 text-to-video and audio-video LoRA training with INT8 ConvRot quantization.**

- **New "LTX 2.3 Video Training" tab** supporting both LTX-2 (19B) and LTX-2.3 (22B) checkpoints (`--ltx_version` selector with automatic checkpoint version detection).
- **Complete video training pipeline**: dataset TOML generation from folder structures (videos and images, 8k+1 frame rule and 25 FPS resampling handled automatically), VAE latent caching, Gemma 3 12B text-encoder output caching, and LoRA training with `networks.lora_ltx2` — all chained from a single Start Training button.
- **INT8 ConvRot quantized training** for the 22B base model: group-wise Hadamard rotation + per-row MSE-clipped INT8 (≈41 dB weight SQNR vs ≈32 dB for scaled FP8), selectable group size, optional per-layer quality report JSON, and support for pre-quantized INT8 ConvRot checkpoints produced by `ltx2_quantize_int8_convrot.py`.
- **Additional base quantizations**: Scaled FP8 (with keep-blocks list and optional W8A8 activations) and NF4, plus block swap up to 47 blocks, H2D-only block swap, blockwise checkpointing, and low main-RAM loading for large-model training on consumer GPUs.
- **Audio-video (av) and audio-only training modes** with separate audio buckets and selectable audio source (video container or external .wav files).
- **Video sample generation during training** with tiled VAE decoding, CPU offloading, LTX-2.3 sampling presets, and optional audio merge into the preview mp4.
- **Model downloader**: new "LTX 2.3 Training Models" bundle (LTX-2.3 22B dev checkpoint + Gemma 3 12B FP8 single-file text encoder, ~59 GB total, no gated repos - no login needed).
- **Demo preset**: `LTX_2.3_LoRA_Demo.toml` — rank 128 LoRA with INT8 ConvRot, tuned for 24-32 GB GPUs.
- Merged the complete LTX-2 backend into our musubi-tuner fork: 60+ new LTX-2 modules including quantizers, LoRA merge/extract/convert utilities, IC-LoRA and conditioning systems, and audio-video joint training — while keeping every existing model family (Qwen, Wan, FLUX, Z Image, Ideogram 4, Krea 2, Hunyuan, FramePack...) unchanged and fully working.
""",
    ),
    (
        "V31.0 — 1 August 2026 — Krea 2 INT8 ConvRot Training + Fixes",
        """
**⚡ A faster, more accurate alternative to Scaled FP8 for Krea 2 LoRA training — and it finally makes Krea 2 fast on RTX 30 series and older GPUs.**

- **Krea 2 INT8 ConvRot training implemented** (integrated musubi-tuner pull request #1008, [arXiv:2512.03673](https://arxiv.org/abs/2512.03673)).
  - It adds a **third base-weight mode** for Krea 2 LoRA training, alongside BF16 and Scaled FP8. The frozen DiT base weights are quantized to INT8 *after a block-diagonal Hadamard rotation that makes them easy to quantize*, and the forward pass then runs a real fused Triton INT8 matmul instead of dequantizing back to BF16.
- **New "ConvRot INT8" checkbox** in the Krea 2 LoRA tab's Model Settings. It replaces FP8 Base + Scaled FP8 (the two cannot be combined — the GUI rejects the combination).
- **New "ConvRot INT8 Backward" dropdown**: `bf16` (default, most accurate) or `int8` (faster gradients, slightly quantized).
- **Same VRAM as FP8** (1 byte per base weight, half of BF16) but roughly **2.5x faster Linear forward than BF16 and 2.7x faster than Scaled FP8** — up to 1.5x faster overall training with no quality loss.
- **Better numerical accuracy than block-wise Scaled FP8** — measured output relative error against the BF16 reference is about 1.4e-2 for ConvRot INT8 vs about 2.5e-2 for Scaled FP8. According to our tests INT8 ConvRot tracks BF16 training more closely than FP8 Scaled does (all runs correlate with the reference at r ≥ 0.9997).
- Works with block swap, gradient checkpointing, and torch.compile. Requires `triton` on Linux / `triton-windows` on Windows — both already installed by this application's requirements.
- Demo configs updated / added for the new INT8 ConvRot training: `Krea_2_LoRA_Demo.toml` (updated) and `Krea_2_LoRA_Demo_High_VRAM.toml` (new).
- Krea 2 made its Hugging Face repo gated after it was added to the downloader, which broke downloads — **fixed** in the latest zip file.
- Some minor bug fixes, improved CMD output to prevent confusion, and improved error reporting.
""",
    ),
    (
        "V30.4 — 27 July 2026 — Faster Startup + Torch Compile Robustness",
        """
- **Improved robustness and startup speed of Torch Compile.**
  - The system probing of CUDA and C++ toolchain elements used to run at every training start, slowing down initialization. It is now **cached for 30 days** (a fresh install into a new folder re-caches).
  - If Torch Compile fails for any reason, training still works via automatic fallback.
- Fixed rare errors when saving and loading configs.
- Fixed a rare Z-Image Base DreamBooth training failure.
- To update: same zip file, just run `Windows_Install_and_Update.bat`.
""",
    ),
    (
        "V30.2 — 18 July 2026 — LTX 2.3 INT8 Row ConvRot HQ Quantization Preset",
        """
- Added **LTX 2.3 INT8 Row ConvRot HQ** quantization preset to the Model Quantizer tab.
- INT8 Row ConvRot HQ is better than GGUF Q8 quality and **more than 100% faster on RTX 3000, 4000 and 5000 series**.
- Krea 2 measurements: INT8 Row ConvRot is **96.2% similar to BF16** while GGUF Q8 is only 90.0%, FP8 Scaled 82.2% and NVFP4 63.7%.
  - INT8 Row ConvRot generates in 3.05 s → **1.82x faster than BF16** (5.56 s). NVFP4 takes 3.8 s (1.46x faster), GGUF Q8 takes 6.06 s (~8.3% slower than BF16).
- LTX 2.3: INT8 Row ConvRot HQ is **100% faster than FP8 Scaled and 50% faster than BF16** on RTX 5090, with quality almost identical to BF16.
- Pre-quantized checkpoints are downloadable with our SwarmUI model downloader app (bundles include them).
- To update from V30, just run `Windows_Install_and_Update.bat`.
""",
    ),
    (
        "V30.0 — 14 July 2026 — Massive Performance Update + Full Fine-Tuning for All Image Models",
        """
**This is a major update with lots of new features and massive performance improvements — Windows finally catches Linux training speeds.**

- Measured speed-ups vs default Musubi Trainer (1024x1024, batch size 1, rank 128 LoRA, zero quality tradeoff):
  - **Krea 2: 36% · FLUX 2 Klein 9B: 46% · Z-Image Base/Turbo: 53% · Ideogram 4: 14% · Qwen 2512: 24.5% · Qwen 2511: 40% · Wan 2.1: 22% · Wan 2.2: 15% · FLUX 2 Dev: 15.6% faster**
- **Full Fine-Tuning / DreamBooth now supported** for FLUX.2 dev, FLUX Klein 4B and 9B, Ideogram 4, Krea 2 Raw and Turbo (integrated and extended musubi-tuner pull request #997). Demo presets updated accordingly.
- **New shared full-model training engine:**
  - Full FP32 or BF16 DiT training and checkpoint export.
  - Exact training resume, including mid-epoch resumes and epoch boundaries.
  - Training-time sampling, checkpoint retention, metadata, Hugging Face upload, and memory-efficient saving.
  - Single-GPU block swapping and fused-backward memory optimizations.
  - Ordinary multi-GPU DDP support with automatic validation of incompatible options.
  - FLUX.1 Kontext full-DiT training also available through the Musubi backend.
- **Three new experimental Automagic optimizers** implemented from Ostris AI Toolkit: Automagic, Automagic2 and Automagic3, available across all training tabs.
  - Adaptive learning rates with per-element, per-tensor, and per-group strategies.
  - Support LoRA and full Fine-Tuning / DreamBooth, block swapping, checkpoint resume, and low-precision training.
  - Automatic fused/non-fused selection where supported; unsafe combinations are rejected before training; detailed optimizer-specific guidance in the GUI.
- **Guarded Windows SDPA acceleration:** native PyTorch fused SDPA is preferred; external FlashAttention is used only after passing real CUDA forward/backward compatibility tests; automatic fallback to working PyTorch SDPA. Every training tab has a "Use Legacy PyTorch SDPA" compatibility switch.
- Smarter torch.compile with block swapping.
- Improved handling of empty or missing Accelerate launch values; hardened Qwen and Z-Image full-model configuration previews and runtime exports; correct full-fine-tuning metadata written to Qwen and Z-Image checkpoints.
- New gradient statistics (including maximum absolute gradient) measured before clipping and sent to TensorBoard / Weights & Biases.
- Added CUDA 13.2 / PyTorch packaging support; improved Ninja and compiler discovery for virtual environments and Visual Studio.
""",
    ),
    (
        "V29.0 — 12 July 2026 — Torch 2.13 + CUDA 13.1, Krea 2 & Ideogram 4 Training Added",
        """
**A pretty massive update — please make a separate fresh install for this one.**

- **Fully moved to Torch 2.13.0 and CUDA 13.1** with pre-compiled libraries. Preferred Python is now 3.12.10 (3.10/3.11/3.13 also work).
- Pre-compiled the following wheels with all CUDA 13 features and all CUDA architectures: **mslk, xformers, flash_attn, sageattention, torchao** — all abi3, so they work on Python 3.10–3.13, Windows and Linux, consumer and cloud GPUs.
- **Full Krea 2 training added** (LoRA at release; fine-tuning arrived in V30).
- **Full Ideogram 4 training added** (LoRA at release; fine-tuning arrived in V30).
- Demo presets in `Demo_Training_Configs_FLUX-2_Z-Image_FLUX-Klein_WAN-21_Krea2_Ideogram4` folder.
- **Model downloader updated and improved** — more robust and faster.
- **Auto model-path detection:** when you load a config, the app checks whether it has valid model paths; if not, it scans the default model-downloader paths and auto-fills them. Works on Windows, Linux and Cloud.
- Download multiple models at once with comma separation or ranges, e.g. `1,2,3` or `1-3,5`.
- Model Quantizer tab completely improved with newer presets. INT8 ConvRot Learned (Best Quality / Slow) preset recommended — better than GGUF Q8 quality and 2x faster on all GPUs. ComfyUI and SwarmUI fully support INT8 Row ConvRot.
- Warnings and errors are now displayed in the Gradio UI with notice bubbles.
- Torch Compile works even better; separate C++ build tools no longer needed — only Visual Studio Community Edition with C++ options.
- Accurate training step speed shown almost immediately after training starts (fixed in our maintained fork).
- RunPod, SimplePod and Massed Compute installers auto-install Python 3.12.
""",
    ),
    (
        "V28.3 — 4 June 2026 — Image Captioning: Append Mode",
        """
- New requested feature in Image Captioning: when Overwrite is off, **append new captions to existing text files** instead of skipping them ("Append Existing Captions").
- Just run `Windows_Install_and_Update.bat` to update.
""",
    ),
    (
        "V28.2 — 20 April 2026 — Quantizer Prodigy Upgrade + New Presets",
        """
- Fixed torchaudio version bug.
- Quantizer application updated: it now uses prodigy to generate FP8 Scaled quants or NVFP4 — faster and better quality.
- **FLUX 1 DEV, FLUX 2 Klein and ERNIE presets added**; all presets updated to the new prodigy pipeline.
- Get the latest zip file, extract and overwrite all, then run `Windows_Install_and_Update.bat` to update.
""",
    ),
    (
        "V27.6 — 8 March 2026 — Model Quantizer Significantly Updated",
        """
- Model quantizer app significantly updated; presets updated and made much better.
- FP8 Scaled generation is much better than default FP8.
- Zip file unchanged — just run `Windows_Install_and_Update.bat` to update.
""",
    ),
    (
        "V27.5 — 10 February 2026 — Sample Generation Improvements",
        """
- Sample generation section and backend for Qwen, Wan, FLUX and Z-Image improved and fixed.
""",
    ),
    (
        "V27.2 — 7 February 2026 — FLUX Training Tab + Z Image Training Tab + Model Quantizer Page",
        """
**A pretty big update:**

- **New Model Quantizer page** supporting FP8 Scaled, NVFP4 and more quantizations.
- **FLUX Training tab added:**
  - FLUX 2 Dev LoRA training with Torch Compile — min 18 GB VRAM at rank 128 / 1024px (reduce rank or resolution for less).
  - FLUX Klein 9B LoRA training — min 9.6 GB VRAM at rank 128 / 1024px.
  - FLUX Klein 4B LoRA training — min 5.6 GB VRAM at rank 128 / 1024px.
- **Z Image Training tab added:**
  - Z-Image Base Fine-Tuning with Torch Compile — min 6 GB VRAM at 1024px.
  - Z-Image Base LoRA training — min 6.1 GB VRAM at rank 128 / 1024px.
  - Z-Image Turbo LoRA training — min 8.6 GB VRAM at rank 128 / 1536px.
- New demo configs in `Demo_Training_Configs_FLUX-2_Z-Image_FLUX-Klein_WAN-21`.
- Configs are set for lowest VRAM — with more VRAM you can reduce block swap count for more speed.
""",
    ),
    (
        "V26.2 — 16 January 2026 — SimplePod Support (Cheaper Than RunPod)",
        """
- RunPod template link updated, and we now fully support **SimplePod** — much faster and cheaper than RunPod (e.g. RTX 5090: $0.45/hr vs $0.89/hr; RTX PRO 6000: $0.79/hr vs $1.84/hr).
- Register: [https://simplepod.ai/ref?user=secourses](https://simplepod.ai/ref?user=secourses) and use our ready template.
- As usual follow `Massed_Compute_Instructions_READ.txt` and `Runpod_SimplePod_Musubi_Trainer_Instructions.txt`.
""",
    ),
    (
        "V26.2 — 4 January 2026 — Pinned RAM for FP8 Converter",
        """
- FP8 Model Converter (convert_to_quant learned rounding method): new option **Use pinned RAM for faster GPU transfers** (page-locked memory).
""",
    ),
    (
        "V26.0 — 1 January 2026 — Qwen Image 2512 + Learned Rounding Converter",
        """
- **Qwen Image 2512 BF16** added to the downloader app — train it exactly like Qwen Image, zero difference.
- New highly experimental **Improved (convert_to_quant learned rounding)** mode added to the FP8 Converter tool (based on [convert_to_quant](https://github.com/silveroxides/convert_to_quant)).
""",
    ),
    (
        "V25.2 — 31 December 2025 — Wan Target Frames Controls",
        """
- **Target frames** can now be set in the interface — this matters for video training quality.
- New **Auto Normalize Target Frames** option (recommended and auto-enabled).
- Both options are in the Wan Training dataset preparation tab.
""",
    ),
    (
        "V25.1 — 29 December 2025 — uv Installers (~100x Faster) + Qwen Image Edit 2511",
        """
- **Installers upgraded to uv pip installation** — installation on RunPod is now literally ~100x faster, and many times faster on Windows.
- Long-awaited **Wan 2.2 training tutorial** published: [https://youtu.be/ocEkhAsPOs4](https://youtu.be/ocEkhAsPOs4)
- **Qwen Image Edit 25-11 model** added to the training models downloader and training support implemented — you can train it without control images just like a regular text-to-image model.
- Model version is now selected from a dropdown.
""",
    ),
    (
        "V23 — 17 December 2025 — Official Wan 2.2 Training Configs (64+ Trainings Research)",
        """
**Massive update — long-awaited Wan 2.2 official training configs published.**

- Wan 2.2 fully researched with literally **over 64 unique trainings**, analyzed on an 8x B200 cloud machine.
- After watching the Qwen tutorial you can train Wan 2.2 Text-to-Image, Text-to-Video or Image-to-Video the exact same way. Static image datasets work perfectly.
- **Configs for literally every GPU tier** included.
- Fixed Qwen Image Fine Tuning Tier1_84000_MB.toml config (Fused Backward pass was off due to an upstream Kohya Musubi bug).
- Fixed several serious bugs in Wan 2.2 model training.
""",
    ),
    (
        "V21 — 29 November 2025 — Torch Compile Arrives",
        """
- **Torch Compile support for Qwen and Wan** — all Qwen training configs modified to auto-enable Torch Compile.
  - Around 7.5–15% speed-up at the time and lower VRAM usage with absolutely 0% quality loss — total win.
- FP8 Model Converter now supports **Z Image Turbo** — that is how the FP8 Scaled Z Image Turbo model was published first in the community, with quality almost identical to BF16.
- If Visual Studio Build Tools MSVC is properly installed, training auto-starts with a cl.exe-enabled environment for more complex Torch Compile options.
""",
    ),
    (
        "V20 — 14 November 2025 — LoRA Extractor + LoRA Merger Tabs",
        """
- **New tab: LoRA Extractor** — extract a LoRA from a fine-tuned Qwen model. Works amazingly well.
- **New tab: LoRA Merger** — merge LoRAs for Qwen, SkyReels, Wan; merge LoRA into LoRA, or merge LoRA + base model into a new base model.
- Main training tutorial published: [https://youtu.be/DPX3eBTuO_Y](https://youtu.be/DPX3eBTuO_Y) and realism tutorial: [https://youtu.be/XWzZ2wnzNuQ](https://youtu.be/XWzZ2wnzNuQ)
""",
    ),
    (
        "V19 — 30 October 2025 — Tutorial-Ready Configs + SwarmUI Metadata",
        """
- All Qwen configs updated (LoRA and Fine-Tuning tiers) — fully ready for the Qwen Image tutorial.
- New option: **Faster Model Loading** (uses more RAM but speeds up model loading — enable on RunPod).
- Enabling Qwen-Image-Edit-2509 mode now auto-sets metadata so SwarmUI auto-recognizes checkpoints (needed for fine-tuning).
- All Fine-Tuning configs now set 1328x1328 metadata so SwarmUI auto-recognizes resolution accurately.
""",
    ),
    (
        "V18.7 — 28 October 2025 — Image Preprocessing Tab + FP8 Model Converter",
        """
- **New Image Preprocessing tool:** preprocesses your dataset folder with the actual Kohya Musubi Tuner training code — see exactly the resolution, aspect ratio and bucketing the trainer will use.
  - Kohya doesn't apply EXIF orientation, so this tool also reveals (and lets you fix) mis-oriented image surprises.
  - You can use the pre-processed dataset as your training dataset to save time and resources.
- **Scaled FP8 Model Converter:** batch-convert fine-tuned BF16 Qwen models to Scaled FP8 — half the size, almost same quality, extremely useful below 60 GB VRAM.
- Detailed debug options added to latent caching so you can inspect processed images one by one.
""",
    ),
    (
        "V18.0 — 21 October 2025 — Qwen Image Edit Plus (2509) Support",
        """
**🎨 Complete integration of Qwen Image Edit Plus (2509) for advanced image editing training.**

- **Multiple control images** — train with up to 3 control images simultaneously (numbered image_0.png, image_1.png, image_2.png convention).
- **Auto-Generate Control Images** — one click creates black PNG control images matched to each training image, with configurable dimensions, for easy dataset preparation.
- Smart mutual-exclusion logic between Edit and Edit-Plus modes; automatic `--edit_plus` flag management; full config persistence and validation.
- Qwen Image Edit Plus (2509) model added to the model downloader on all platforms.
- Qwen Image Fine-Tuning configs completed.
- Fixed "Invalid digits suffix" dataset filename error; improved dataset processing robustness.
""",
    ),
    (
        "V17.9 — 21 October 2025 — Qwen Fine-Tuning / DreamBooth Research Completed",
        """
- Qwen Image Full Fine-Tuning / DreamBooth research completed.
- Configs prepared from **5750 MB (6 GB GPUs) up to 84000 MB (96 GB GPUs)**.
- Three config sets: 200_epoch (best quality), 150_epoch (good quality), 75_epoch (faster training).
- All Qwen configs organized into `Qwen_Image_Training_Configs` with LoRA_Training and Fine_Tuning_Training-DreamBooth subfolders.
- Sub-20 GB configs use CPU BF16 T5 text-encoder caching — still really fast.
""",
    ),
    (
        "V17.8 — 20 October 2025 — Wan 2.2 BF16 Models in Downloader",
        """
- Model downloader now supports **Wan 2.2 Text-to-Video and Image-to-Video** models.
- Official FP32 Wan weights converted to BF16 — better quality at training and often at inference too.
- Example Wan 2.2 LoRA configs added (10250 MB and 12000 MB VRAM variants).
""",
    ),
    (
        "V17.5 — 3 October 2025 — Fixes",
        """
- Bug fix for Wan training on Linux systems (RunPod & Massed Compute).
- Low-memory branch switch file removed (merged upstream).
- Improved Wan models training directory explanation.
""",
    ),
    (
        "V17.0 — 29 September 2025 — Wan Video Training Support",
        """
**🎬 Complete WAN video models training support.**

- **Text-to-Video, Image-to-Video, Text-to-Image, FramePack, Fun-Control and Wan 2.2 dual-model training** — everything Musubi Tuner supports.
- Smart dataset setup auto-detects and organizes images and videos; one-click resolution configs; flow-matching training.
- Multi-architecture support (1.3B / 14B), dual-model Wan 2.2 training with smart CPU/GPU offloading.
- Full memory toolkit: block swapping, FP8 precision, VAE tiling, CPU offloading.
- Model downloader updated for Wan 2.1 models on Windows, RunPod and Massed Compute.
""",
    ),
    (
        "V16.0 — 23-27 September 2025 — Qwen LoRA Research Complete + Advanced Network Architectures",
        """
- **Qwen Image LoRA research completed — over 50 full trainings** to find the best parameters and workflow; all configs updated (6e-05 learning rate finding).
- App fully supports **Qwen Image Full Fine-Tuning / DreamBooth** (demo config included, 5 GB VRAM path documented).
- **Advanced LoRA architecture selection:** standard `networks.lora_qwen_image`, `networks.dylora`, `networks.lora_fa`, or any custom module.
- **Post-training format conversions:** auto-convert to Diffusers format or alternative SafeTensors key naming.
- **Real-time FP8 validation** (warns when fp8_base is enabled without fp8_scaled), dtype conflict detection, attention mechanism priority system (SDPA → FlashAttention → SageAttention → xformers), SageAttention and FlashAttention 3 options.
- Intelligent auto-detection for DiT layer count and network dimension.
- **Critical fix:** DreamBooth Fine-Tuning mode now correctly uses `qwen_image_train.py` instead of the LoRA script.
- Installers updated to Torch 2.8, CUDA 12.9, Flash Attention 2.8.3, Sage Attention 2.2, xFormers 0.0.33 with pre-compiled wheels for Windows and Linux.
""",
    ),
    (
        "V15.0 — 12 September 2025 — Intelligent Logging Directory Management",
        """
- **Fixed PermissionError** when creating logging directories at filesystem root.
- Smart `logging_dir` handling: empty → auto-creates `{output_dir}/logs/session_{timestamp}`; relative → `{output_dir}/mylogs`; absolute → used as-is.
- Cross-platform path normalization with forward-slash standardization for TOML compatibility.
""",
    ),
    (
        "V14.0 — 5 September 2025 — Sample Prompt Enhancement Control + Configs_v1",
        """
- New **Disable Automatic Prompt Enhancement** checkbox — use your prompt txt files exactly as written.
- Configs_v1 released; tier system introduced (Tier 1 best → Tier 3), supporting GPUs as low as **6 GB VRAM**.
- Paths containing spaces now work (still recommended to avoid them).
""",
    ),
    (
        "V13.0 — 4 September 2025 — Cross-Platform Path Handling",
        """
- **Fixed dataset paths containing spaces** — now generate valid TOML on Windows and Linux.
- Added normalize_path(), validate_path_for_toml() and is_path_safe() utilities; all file dialogs normalize paths consistently.
""",
    ),
    (
        "V12.0 — 4 September 2025 — Text Encoder Caching Config Fix",
        """
- **Fixed Text Encoder Caching parameters not saving** — all caching_teo_* settings (text encoder path, device, FP8, batch size, workers, skip existing, keep cache) now persist correctly across save/load.
""",
    ),
    (
        "V11.0 — 31 August 2025 — Stop Button for Batch Captioning + Critical Config Fixes",
        """
- **Professional Stop button for batch image captioning** — safe interruption that finishes the current image, preserves completed captions and restores button states.
- Fixed caption prefix/suffix timing — now applied after word replacement.
- Critical config save/load bugs fixed (including GPU ID defaulting to 0 — important when you have an iGPU + discrete GPU).
- UI significantly improved; downloader now shows single-line per-model progress.
- Comprehensive parameter support with 100% coverage of Musubi Tuner parameters; **integrated search bar** to find any setting instantly.
""",
    ),
    (
        "V10.0 — 31 August 2025 — Critical Checkpoint Management Fix",
        """
- **Fixed critical checkpoint removal bug** — checkpoints were deleted immediately after saving when save_last_n_epochs=0.
- Proper 0→None translation for all save_last_n_* parameters; clearer "Keep Last N Checkpoints/State Files" labels.
""",
    ),
    (
        "V9.0 — 31 August 2025 — Config Save/Load Fixes",
        """
- Fixed ddp_timeout and save_last_n_epochs being forced to minimum value 1.
- Fixed GPU IDs not being saved to TOML config files.
- Resolved configuration double-save corruption issues.
""",
    ),
    (
        "V8.0 — 31 August 2025 — 100% Parameter Coverage + Search Bar",
        """
- Comprehensive parameter support for Qwen Image training with **100% coverage of Musubi Tuner parameters**.
- **New integrated search bar** — quickly find any setting without opening all panels.
- Tab renamed from "Qwen Image LoRA" to "Qwen Image Training" (LoRA + Fine-Tuning).
- Qwen-Image-Edit mode support for control image training; control image resolution settings; advanced flow matching parameters (logit_mean, logit_std, mode_scale); complete VAE optimization settings.
""",
    ),
    (
        "V7.0 — 30 August 2025 — Smart Sample Prompt Enhancement",
        """
- Intelligent sample prompt enhancement system — automatically applies optimal Qwen Image resolution (1328x1328) and GUI defaults to sample prompts; per-prompt overrides via --w, --h, --s, --g, --d flags.
- Dedicated Sample Generation Settings section; enhanced prompt files saved to output directory for transparency.
""",
    ),
    (
        "V6.0 — 30 August 2025 — Optimizer/Scheduler Args Fix",
        """
- Fixed broken config save/load for Optimizer Arguments and Scheduler Arguments.
- Stop Training button now appears much earlier (when Text Encoder caching starts).
""",
    ),
    (
        "V5.0 — 29 August 2025 — Captioning Skip Logic Fix",
        """
- Fixed skip-existing-captions logic (checks before processing instead of after generation).
- Full batch captioning status display with progress tracking and ETA.
""",
    ),
    (
        "V4.0 — 29 August 2025 — Sample Prompts File Selector Fix",
        """
- Fixed TypeError in the sample prompts file selector folder icon.
""",
    ),
    (
        "V3.0 — 29 August 2025 — SHA256-Verified Model Downloader",
        """
- Model downloader made more robust with **SHA256 verification** — a corrupted-but-same-size model file was the cause of captioning failures, now impossible.
- Qwen2.5-VL image captioning works perfectly on Windows and Linux.
""",
    ),
    (
        "V2.0 — 29 August 2025 — Dataset Generation Fix",
        """
- Fixed dataset generation error in generate_dataset_config_from_folders().
- Added example filenames to model path descriptions; minor GUI description improvements.
""",
    ),
    (
        "V1.0 — 28 August 2025 — Initial Release",
        """
**Initial app release.**

- Full support for Qwen Image LoRA training.
- Qwen2.5-VL based image captioning.
- Intuitive GUI for training configuration with batch processing capabilities.
""",
    ),
]


def version_history_tab(headless=False, config=None):
    """
    Create the Version History / Changelog tab.

    Every release lives in its own accordion so users can quickly scan the
    version list and expand only the releases they care about.
    """
    with gr.Column():
        with gr.Row(equal_height=True):
            with gr.Column(scale=5):
                gr.Markdown("## 📜 Version History / Changelog")
            with gr.Column(scale=1):
                toggle_sections_btn = gr.Button(
                    value="Open All Sections",
                    variant="secondary",
                    size="lg",
                    elem_id="toggle-all-changelog-btn",
                    elem_classes=["mbtn", "mbtn-indigo"],
                )
                sections_state = gr.State(value="closed")
        gr.Markdown(
            """
Complete release history of **SECourses Musubi Trainer** — newest release first.
Click any version below to expand its full changelog.

**Latest zip file, tutorials and support:** [https://www.patreon.com/posts/137551634](https://www.patreon.com/posts/137551634)
"""
        )

        accordions = []
        for index, (title, body) in enumerate(CHANGELOG_ENTRIES):
            with gr.Accordion(title, open=(index == 0)) as section_accordion:
                gr.Markdown(body)
            accordions.append(section_accordion)

        def toggle_all_sections(current_state):
            if current_state == "closed":
                new_state = "open"
                new_text = "Close All Sections"
                accordion_states = [gr.Accordion(open=True) for _ in accordions]
            else:
                new_state = "closed"
                new_text = "Open All Sections"
                accordion_states = [gr.Accordion(open=False) for _ in accordions]
            return [new_state, gr.Button(value=new_text)] + accordion_states

        toggle_sections_btn.click(
            toggle_all_sections,
            inputs=[sections_state],
            outputs=[sections_state, toggle_sections_btn] + accordions,
            show_progress=False,
        )
