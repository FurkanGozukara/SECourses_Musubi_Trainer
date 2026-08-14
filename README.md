# SECourses Musubi Trainer V34 — The Only App You Need to Train Every Major Open Model: Qwen Image, Wan 2.1 / 2.2, FLUX 2, FLUX Klein, Z-Image, Ideogram 4, Krea 2, LTX 2.3 and MiniMax H3 Video + Audio

**One app. Fifteen tabs. State-of-the-art open image and video training, including joint audio — Full Fine-Tuning / DreamBooth for supported image families and LoRA across image and video families — with 1-click installers, researched ready-to-use presets, and speed the stock trainer simply cannot reach.**

- **App Download Link (Patreon) : https://www.patreon.com/posts/secourses-musubi-137551634**
- **Latest Zip File : [SECourses_Musubi_Trainer_v34.zip](https://www.patreon.com/file?h=137551634&m=717993383)**
- **[Click here to choose a membership and Join to download the zip file](https://www.patreon.com/cw/SECourses/membership)**
- **Main training tutorial — mandatory to watch : https://youtu.be/DPX3eBTuO_Y**
  - Wan 2.2 training tutorial : https://youtu.be/ocEkhAsPOs4
  - SwarmUI (for inference of your trained models) : https://www.patreon.com/posts/114517862
  - ComfyUI : https://www.patreon.com/posts/105023709

---

## 14 August 2026 Version 34.0 Update

- **Full MiniMax H3 video + audio LoRA training added**
  - Brand-new **MiniMax H3 Video Training** tab for text-to-video-with-audio (T2VA), first/last-frame FL2VA and reference-to-video Ref2VA training.
  - One Start Training button chains dataset generation → video/audio latent caching → Qwen3-VL text-embedding caching → training.
  - Joint audio supervision uses embedded audio or matching `.wav` files; silent clips are not incorrectly supervised toward silence.
  - Dataset rules are handled automatically: batch size 1, valid `17n+5` frame counts, 24 FPS normalization and 32-pixel resolution steps.
  - Guidance-distillation protection is enabled by default to preserve H3's CFG-distilled behavior.
  - Advanced teacher matching supports both the `first,last` endpoint teacher and the new `ref` self-reference teacher, with stronger teaching signals and real audio targets.
  - Sigma protection, timestep focus, DC/magnitude shaping, audio-loss weighting and video-only controls are exposed in the GUI.
- **Consumer-GPU memory support**
  - Automatically loads the approximately 21 GB pruned ConvRot INT8 H3 transformer.
  - Supports the NVFP4+AWQ Qwen3-VL text encoder with layer-by-layer CPU streaming.
  - Swaps up to 48 of H3's 50 transformer blocks with H2D-only block swapping.
  - Full BF16 checkpoints can use on-the-fly ConvRot INT8 and AdaLN pruning.
  - Compatible ComfyUI pre-quantized ConvRot INT8 checkpoints load directly; unsupported H3 FP8 configurations are blocked with a clear error.
- **Models and ready-to-use presets**
  - New **MiniMax H3 Training Models — Low VRAM** downloader bundle, plus optional higher-quality ConvRot INT8 text-encoder and Ref2VA transformer downloads.
  - `MiniMax_H3_LoRA_Demo_24GB.toml`: 384px, 124 frames, measured at 22.5 GB peak on an RTX 3090.
  - `MiniMax_H3_LoRA_Demo_Lowest_VRAM.toml`: 256px, 124 frames with heavy block swapping, measured at approximately 8.5 seconds/iteration on an RTX 5090.
- **LTX 2.5 quantization**
  - New Model Quantizer preset converts an original BF16 dev or distilled model into an approximately 22 GB ComfyUI-ready mixed BF16/INT8 ConvRot checkpoint.
  - The workflow detects the model variant, applies the benchmark-validated video layer plan, selects the ConvRot group size per layer, validates the architecture and writes a detailed JSON quality report.
  - Older V32.4 LTX 2.5 workflow configurations migrate automatically.
- **Model Quantizer improvements**
  - Added or improved profiles for Qwen VLM/Qwen3.5+, Chroma/Distilled, Mistral, NeRF, Radiance, visual encoders and other architectures.
  - Qwen VLM protection now handles the final language layer, embeddings, MTP weights and visual stack dynamically.
  - Added BF16/FP16 output selection and regex controls for layers that must preserve their source dtype.
  - Quantizer panels now appear only when relevant; preset/workflow switching, pre-quantized ConvRot validation and mixed group-size handling are more consistent.
- **Critical Full Fine-Tuning safety fix**
  - Runtime TOMLs now preserve LoRA versus Full Fine-Tuning / DreamBooth mode across FLUX, FLUX.2, FLUX Klein, Qwen Image, Z-Image, Ideogram 4 and Krea 2.
  - Older configurations infer the correct mode from their network settings.
  - Loading a saved or failed full-finetune run no longer silently changes it to LoRA, and the matching GUI panels update correctly after loading.
- **Reliability improvements**
  - Fixed H3 FlashAttention 2 dtype handling, CPU-offloaded activations before the final layer and condition-noise random-number aliasing.
  - Improved audio timestamp-jitter recovery and JSONL relative-path resolution.
  - Added official MiniMax H3 processor/configuration handling.
  - Fixed end-to-end H3 dataset validation and `flash_auto` launch support.
  - Expanded regression coverage across configuration round-trips, caching, quantization, teacher matching and dataset generation.
- New training tab interface below

<img height="600" alt="image" src="https://github.com/user-attachments/assets/5f1159ac-ce53-4f5e-8e27-ff458110a19e" />

- Updated model downloader bat file interface below

<img  height="600" alt="image" src="https://github.com/user-attachments/assets/f2726961-022e-4cfd-ae49-95e1d15a6309" />

- Demo presets folder screenshot below

<img height="600" alt="image" src="https://github.com/user-attachments/assets/cd38f0cc-7e17-4e20-a5de-b9b5fc18a46a" />

- To update, download the latest zip, extract and overwrite the existing files, run `Windows_Install_and_Update.bat`, and use the newest preset TOMLs.

## 13 August 2026 Version 32.3 Update

- Fixed a FLUX 2 Klein training bug and updated the related demo presets.
- Extract and overwrite the existing files, run `Windows_Install_and_Update.bat`, and use the newest preset TOMLs.

## 3 August 2026 Version 32.1 Update

- **Interface fully modernized and upgraded** — every screenshot in this post shows the new look.
- **LTX 2.3 full training capability implemented** with demo preset — even higher quality and faster INT8 ConvRot base training supported, same as Krea 2.
- Now even if you forget to write 1_folder_name (the repeating number) in dataset preparation, the app counts it automatically as 1 repeat and tells you exactly how to set it explicitly.
- **Automagic optimizers now fully working** — ported from the famous Ostris AI Toolkit (details below).
- Full version history is now inside the Gradio app — Version History tab.
- The post is fully updated so please read all. To update: get the latest zip file, overwrite previous files and run Windows_Install_and_Update.bat.

LTX 2.3 is unusual among supported families — a single checkpoint carries the DiT transformer, video VAE, audio VAE, vocoder and text-encoder connector weights. The app handles the whole pipeline for you, including the 8k+1 valid frame rule and FPS resampling — here is the built-in explanation of how LTX 2.3 training works:

![How LTX 2.3 training works — built-in guide](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/A_LTPRdRJptTTr2Dh_Agt.png)

**Automagic optimizers — which one should you pick?** Automagic is an experimental family of adaptive optimizers originally from Ostris AI Toolkit, integrated into this trainer — and not limited to Krea 2: it is available for Qwen Image, Wan, FLUX, Z-Image, Ideogram 4, Krea 2 and LTX 2.3 training.

- **Automagic3 — the recommended starting point.** It automatically selects the correct operating mode: fused updates when your configuration is compatible, safe non-fused updates when you use gradient accumulation / clipping or other incompatible features (with a warning), and a compact sign-history state for its adaptive learning rate. Choose Automagic3 when unsure.
- **Automagic2 — lowest gradient memory, strict requirements.** It updates parameters during backward propagation, so it needs: one GPU and one process, Gradient Accumulation Steps = 1, Max Gradient Norm = 0, BF16 or no mixed precision (FP16 unsupported), Fused Backward Pass disabled (it manages its own) and Patch Optimizer for Block Swap disabled. If your configuration is incompatible, the trainer **stops before loading the model** and explains exactly which settings to change.
- **Automagic v1 — the most detailed adaptive state.** Adafactor-style second moment plus an adaptive per-element learning-rate mask; supports normal stepping, gradient accumulation, clipping and block swapping, at somewhat higher optimizer memory.
- **Learning-rate behavior:** for all Automagic versions the Learning Rate you set is only the *starting* rate — Automagic adapts it during training and bypasses external schedulers. Start from the model preset values, don't copy AdamW rates.
- Tested successfully with ConvRot (BF16 and INT8 backward) and Scaled FP8 base quantization.

---

## SECourses Musubi Trainer Premium App Full Features

## Why This App Exists

Kohya's Musubi Tuner is the best training backend in the open-source world — but out of the box it is a command-line tool: hand-written TOML files, manual model downloads, manual wheel compilation, and no Windows Torch Compile. I maintain my own fork of Musubi Tuner and wrapped it in a complete product: a full GUI, 1-click installers for Windows and every major cloud, an ultra-fast verified model downloader, researched configs for every GPU tier from 6 GB to 96 GB, and performance work that makes trainings up to 53% faster at identical quality.

This is not a thin wrapper. Every feature of Musubi Tuner is exposed — **100% parameter coverage** — plus dozens of features that exist only here: INT8 ConvRot quantized training, LTX 2.3 and MiniMax H3 video + audio training, LTX 2.5 quantization, guarded Windows SDPA acceleration, a model quantizer that beats GGUF Q8 quality, Automagic optimizers, and much more. It is updated constantly: **44 releases shipped in 11 months.**

![SECourses Musubi Trainer vs stock Musubi Tuner comparison](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/s2x-IOZChpAQ1IZNSUDHa.png)

## Up to 53% Faster Than Stock Musubi Tuner — Zero Quality Loss

I pre-compiled the entire acceleration stack myself (FlashAttention, SageAttention, xFormers, TorchAO, mslk — for Torch 2.13 + CUDA 13.1, abi3 wheels working on Python 3.10–3.13, on consumer and datacenter GPUs), and built an automatic Torch Compile toolchain that discovers and validates your MSVC/CUDA environment on Windows, caches the result for 30 days, and falls back safely if anything fails. The result — measured model by model against default Musubi Trainer settings, same exact output quality:

![Training speed-up benchmark chart — up to 53% faster](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/kXglvnJvZR2FuG0tKmBML.png)

- **Z-Image Base / Turbo : 53% faster · FLUX Klein 9B : 46% · Qwen 2511 Edit : 40% · Krea 2 : 36% · Qwen 2512 : 24.5% · Wan 2.1 : 22% · FLUX 2 Dev : 15.6% · Wan 2.2 : 15% · Ideogram 4 : 14%**
- All of it works out of the box on Windows and Linux — no compiler setup, no environment variables, no praying.
- On Krea 2 and LTX 2.3 you can stack the exclusive **INT8 ConvRot quantized training** on top — the measured speed table is in the Krea 2 section below.

## Every Model. One App.

Full Fine-Tuning / DreamBooth **and** LoRA for all major image models, plus LoRA for every Wan 2.1 / 2.2 variant, LTX 2.3 with audio, and MiniMax H3 with audio, first-frame, last-frame and reference conditioning — every family with fully working Torch Compile:

![Supported models and training modes matrix](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/NU3YnMGs8iT6lwqzZ7gUd.png)

## The Presets We have in the Zip File and the Automatic Installers

Everything in this post ships as one zip file. Extract it and you get the 1-click installers for Windows, RunPod, SimplePod and Massed Compute, the model downloader (Windows_Download_Training_Model_Files.bat / Download_Train_Models.py), the app starter, the instruction files for every cloud — and every researched preset folder shown below:

![Zip file contents — installers, downloader and preset folders](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/OnFcR-bKDOwKK1Vrjyp-U.png)

**Ready demo presets for every model family** — FLUX 2 (LoRA + Fine-Tuning), FLUX Klein 4B / 9B, Ideogram 4, Krea 2 (including the High-VRAM INT8 ConvRot variant), LTX 2.3, MiniMax H3 (24 GB and lowest-VRAM variants), Wan 2.1 and Z-Image Base / Turbo. Load one, point it at your dataset, press Start:

![V34 demo training config presets, including MiniMax H3](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/LVtXuIFGJWQAF998Vd-3d.png)

**The full Qwen research config library** — separate LoRA_Training and Fine_Tuning_Training-DreamBooth tiers for every GPU class, plus complete example datasets for regular training and for Qwen Image Edit control-image training, and a Tutorial_Workflow.html reference:

![Qwen Training Configs folder — LoRA and Fine-Tuning tiers with example datasets](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/7AnBcG8rLfcTtYGOWDlde.png)

**A ready-made prompt library for testing your trainings** — demo prompt sets for style / character / product trainings, Gemini prompt-generator templates, and Grid_Find_Best_Checkpoint prompt files you use to X/Y-grid your saved checkpoints and find the best epoch:

![Qwen Training Tutorial Prompts folder — ready prompt libraries](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/DPC7lus0Xh9GoH_bLwyQe.png)

**The official Wan 2.2 config sets** — Text-to-Video, Image-to-Video and Text-to-Image folders distilled from my 64+ research trainings, with a Configs_Explanation guide image:

![Wan 2.2 Training Configs folder — T2V, I2V and T2I sets](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/1xLKR3ObEGjrj2SE4-spZ.png)

---

## The Grand Tour — All 15 Tabs, With Real 4K Screenshots

### 1 — Qwen Image Training

The flagship tab. Trains Qwen Image, the new Qwen 2512, and Qwen Image Edit Plus 2509 / 2511 (with up to 3 control images and one-click auto-generated control images). LoRA and Full Fine-Tuning / DreamBooth, backed by my research of **over 50 complete trainings** — the resulting configs cover every GPU from 6 GB to 96 GB in quality tiers, so you load a preset matching your VRAM and press Start.

![Qwen Image Training tab](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/aByv9ZtT7zxyTU7yQoPC6.png)

And this is what one single tab actually contains — every Musubi Tuner parameter, organized, documented with tooltips, and searchable. You never touch a TOML file by hand again. Here is the Qwen tab, section by section:

**Configs, Accelerate & checkpointing** — preset save / load, the settings search bar, Accelerate launch options (multi-GPU, mixed precision) and complete checkpoint save / resume / retention management:

![Qwen tab 1/6 — configs, Accelerate and checkpointing](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/Dc7-jeLQuevjqoExnjayL.png)

**Dataset preparation** — point it at a Kohya-style folder (e.g. `1_ohwx man`) and the full dataset TOML is generated for you, including Qwen Edit control-image modes and automatic caption file creation. And if you forget the repeat-count prefix on a folder, the app does not silently guess: it counts the folder as 1 repeat and tells you exactly how to name it to set repeats explicitly:

![Qwen tab 2/6 — dataset preparation](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/99MyoxmTFdMPOykPkI0NX.png)

**Model settings, Torch Compile & caching** — BF16 model paths with auto-fill, FP8 base + Scaled FP8, Edit / Edit-Plus modes, Torch Compile controls, latent and text-encoder output caching:

![Qwen tab 3/6 — model settings, Torch Compile and caching](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/Aj2Y1cJ_56ywXHYLTggBN.png)

**Optimizers, scheduler & LoRA** — the full optimizer catalog including Automagic 1/2/3 with per-optimizer guidance, learning-rate schedulers, and every LoRA architecture option (DyLoRA, LoRA-FA, custom modules):

![Qwen tab 4/6 — optimizers, scheduler and LoRA](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/UFXuVihUYQYnE34JO5jUA.png)

**Training settings & sample generation** — epochs / steps, attention backends, block swap, gradient checkpointing, and live sample image generation during training:

![Qwen tab 5/6 — training settings and sample generation](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/bK1EvrIqQ97Hek_MYMRae.png)

**Advanced, metadata & Hugging Face** — flow matching and timestep sampling, SwarmUI auto-detection metadata, direct Hugging Face checkpoint upload, and the one-click training start:

![Qwen tab 6/6 — advanced, metadata and Hugging Face](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/7hurmaGbZYO7DzHPoJNbz.png)

### 2 — Wan Models Training

Everything Wan: 2.1 and 2.2, Text-to-Video, Image-to-Video, Text-to-Image, FramePack and Fun-Control — including proper dual-model Wan 2.2 training with smart CPU/GPU offloading. My Wan 2.2 configs come from **64+ research trainings analyzed on an 8x B200 machine**, published for literally every GPU tier.

![Wan Models Training tab](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/ZV5nMQiO6P6R8UlgFtsks.png)

The Wan tab is the deepest video-training UI in the app — here it is, section by section:

**Configs, Accelerate & checkpointing** — preset save / load, instant settings search, Accelerate multi-GPU launch, output naming, exact training resume from saved states, LoRA save precision and checkpoint retention (save every N epochs / steps, keep last N, save state on train end):

![Wan tab 1/7 — configs, Accelerate and checkpointing](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/PQUiH21Zf-TB7hncYkYpI.png)

**Video dataset preparation** — point it at Kohya-style folders of videos and/or images and the full dataset TOML is generated: frame extraction method (head / chunk / slide / uniform), Frame Stride, Frame Sample Count, Target Frames with **Auto Normalize** (scans your videos and clamps Target Frames to the shortest one so no clip is silently skipped), Maximum Frames, Source FPS override and automatic caption file creation:

![Wan tab 2/7 — video dataset preparation](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/C-pxSq-GiEaTZlLLJS_os.png)

**The built-in Complete Wan Dataset Guide** — a full reference rendered right inside the tab: the N*4+1 valid frame-count rule, optimal resolutions per model, video / image / mixed dataset folder structures, multiple datasets with different repeat counts, model-specific dataset recommendations (T2V, I2V, T2I, FLF2V, Fun-Control), automatic frame processing behavior, supported file formats and the experimental One Frame Training mode — you never need to leave the app to look any of this up:

![Wan tab 3/7 — built-in complete Wan dataset guide](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/xiZCOYjAoj5eSw3HHk3Rx.png)

**Wan model settings** — every Wan variant in one dropdown (T2V 1.3B / 14B, I2V, T2I, FLF2V, Fun-Control) with per-model max block swap limits, the **Wan 2.2 dual-model system** (high-noise + low-noise DiT paths with automatic timestep boundary switching and inactive-model CPU offload), DiT / VAE / T5 / CLIP paths with browse buttons and auto-fill, FP8 base + Scaled FP8 + FP8 T5, block swap with pinned-memory and H2D-only modes, VAE tiling / chunking, and Torch Compile with resident-blocks-only compilation for block-swapped training:

![Wan tab 4/7 — Wan model settings](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/dlrCg1q3-KhSPb5jayvGQ.png)

**Advanced timestep sampling, loss weighting & CUDA** — timestep sampling methods with Discrete Flow Shift guidance per model (Wan 2.2 T2V-A14B needs 12.0, I2V-A14B needs 5.0, Wan 2.1 needs 1.0 — the tooltip tells you), min / max timestep focus training, stratified timestep buckets for small datasets, loss weighting schemes (logit normal, mode), TF32 and cuDNN benchmark switches, and memory-efficient model-loading options:

![Wan tab 5/7 — timestep sampling, loss weighting and CUDA](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/fO1lF-ba7OQQLiveIgKuT.png)

**Training, network, optimizer & caching** — epochs / steps, attention backends (SDPA, FlashAttention, SageAttention, xFormers), gradient checkpointing, LoRA network dimension / alpha / dropout, the full optimizer + scheduler catalog, and separate latent-caching and text-encoder-caching controls with skip-existing and keep-cache behavior:

![Wan tab 6/7 — training, network, optimizer and caching](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/42NnYtB8c1TYqkX0Dpe0X.png)

**Samples, advanced, metadata & Hugging Face** — live sample video generation during training from a prompt file, advanced flags, SwarmUI auto-detection metadata, direct Hugging Face upload, then one click on Start Training runs latent caching → text-encoder caching → training back-to-back:

![Wan tab 7/7 — samples, advanced, metadata and Hugging Face](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/g4ObWKb5V6Kw_gbw_g5C-.png)

### 3 — FLUX Training

FLUX 2 Dev and FLUX Klein 9B / 4B — LoRA **and** Full Fine-Tuning. LoRA from as low as **5.6 GB VRAM** (Klein 4B, rank 128 at 1024px), 9.6 GB for Klein 9B, 18 GB for FLUX 2 Dev. Klein 9B trains 46% faster here than on the stock trainer.

![FLUX Training tab](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/9vrNlayFDTLn5OfadXGHI.png)

**Model family, dataset & model settings** — one Model Family dropdown switches the whole tab between FLUX.2 Dev and Klein 9B / 4B, LoRA or Full Fine-Tuning modes, dataset TOML generation with optional control images (for FLUX.1 Kontext-style training), DiT / VAE / text-encoder paths with auto-fill, FP8 base / scaled / text-encoder options and per-model block swap limits (FLUX 2 Dev up to 29 blocks, Klein 9B 16, Klein 4B 13):

![FLUX tab 1/3 — model family, dataset and model settings](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/DDM7AW3LThAarlq0oKJqx.png)

**Torch Compile, training & sampling** — Torch Compile backend / mode / dynamic shapes, flow matching and timestep settings, epochs / batch size / attention backends, live sample generation during training and latent + text-encoder caching controls:

![FLUX tab 2/3 — Torch Compile, training and sampling](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/a8ykn8n1ANIjYdJy2Pt9g.png)

**Optimizer, LoRA & advanced** — the full optimizer catalog with schedulers, LoRA rank / alpha / dropout, advanced flags, metadata and direct Hugging Face upload — then the one-click start:

![FLUX tab 3/3 — optimizer, LoRA and advanced](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/RoZHPNpkH9QLyqqWFCw-Q.png)

### 4 — Z Image Training

Z-Image Base and Turbo, LoRA and Full Fine-Tuning from **6 GB VRAM** — and the single biggest speed win of the whole app: 53% faster than stock. I published the first FP8 Scaled Z-Image Turbo model in the community using this very app's converter.

![Z Image Training tab](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/UeNpqAwgSy35VBVosIJc8.png)

**Configs, dataset & model settings** — Base or Turbo checkpoints, LoRA or Full Fine-Tuning / DreamBooth, dataset TOML generation with automatic captions, FP8 options, block swap and complete checkpoint management:

![Z Image tab 1/3 — configs, dataset and model settings](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/_15Kf931W7lEsd13KoaR5.png)

**Torch Compile, training & sampling** — the 53%-faster Torch Compile pipeline, flow matching and timestep settings, epochs / batch / attention backends, live sample generation and full latent + text-encoder caching:

![Z Image tab 2/3 — Torch Compile, training and sampling](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/KBGLqLsEAy98UwT7TknEC.png)

**Optimizer, LoRA & advanced** — optimizer + scheduler catalog, LoRA architecture settings, advanced flags, checkpoint retention, metadata and Hugging Face upload with the one-click start:

![Z Image tab 3/3 — optimizer, LoRA and advanced](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/I2gCy0yLQkKVF3VPfhRmB.png)

### 5 — Ideogram 4 Training

Ideogram 4 LoRA and Full Fine-Tuning with ready demo presets. Ideogram 4 loves structured JSON captions — and this tab is built for them: it pairs perfectly with my JSON-prompt captioning workflow ([tutorial](https://youtu.be/TW3MRdd0MV4), [Ultimate Image Captioner Pro](https://www.patreon.com/SECourses/posts/ultimate-image-captioner-pro-162527725)).

![Ideogram 4 Training tab](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/sT1S15L3C5n80z9KczL4q.png)

**Configs, dataset & model settings** — LoRA or Full Fine-Tuning, dataset TOML generation with automatic captions and **structured JSON-caption checking** (your caption files are validated with clear warnings before training, not after a wasted run), model paths with auto-fill, FP8 options and Torch Compile:

![Ideogram 4 tab 1/3 — configs, dataset and model settings](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/AVYoFiA3P_RAMRDXOi1rR.png)

**Flow matching, training & sampling** — flow matching and timestep settings, epochs / batch / attention backends, **Ideogram sampler presets for live sample generation**, loss diagnostics, caching and the optimizer catalog:

![Ideogram 4 tab 2/3 — flow matching, training and sampling](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/_6HUAyvtbyeZeY9lP19ZD.png)

**LoRA, advanced & Hugging Face** — LoRA rank / alpha / dropout, advanced flags, SwarmUI metadata and direct Hugging Face upload with the one-click start:

![Ideogram 4 tab 3/3 — LoRA, advanced and Hugging Face](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/PeD1-vSZy0HJ0_wrNZq9p.png)

### 6 — Krea 2 Training (with Exclusive INT8 ConvRot)

Krea 2 Raw and Turbo — LoRA and Full Fine-Tuning, 36% faster than stock. And Krea 2 is where I shipped something nobody else has: **INT8 ConvRot quantized training**.

![Krea 2 Training tab](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/24yc-cxY8iOyDYfLwduia.png)

ConvRot rotates the frozen base weights with a Hadamard transform before quantizing them to INT8, then runs the forward pass as a real fused Triton INT8 matmul. Scaled FP8 cannot do that — Krea 2's FP8 path has no FP8 matmul at all: it dequantizes the weights back to BF16 on **every forward pass** and then runs an ordinary BF16 matmul, which is why FP8 only saves VRAM and actually measures slower than BF16 once you are not bandwidth-starved. ConvRot's INT8 tensor-core path is a genuinely cheaper matmul — and INT8 tensor cores exist on every NVIDIA GPU from Turing onward, which finally makes Krea 2 fast on RTX 30 series cards that have no FP8 hardware at all.

**Measured end-to-end training speed** — real rank-128 LoRA runs at 1024×1024 through the GUI's own pipeline on an RTX 5090 (the shipped presets' exact settings; every number is the steady-state median over long runs):

![Krea 2 INT8 ConvRot measured end-to-end training speed table](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/3hkEAQ5SutoON0LrO8XBo.png)

- **Same VRAM class as FP8** (1 byte per base weight, and measured ~0.9 GB *less* peak VRAM than Scaled FP8), roughly **2.5x faster Linear forward than BF16 and 2.7x faster than Scaled FP8** in isolated kernel benchmarks — on both RTX 5090 and RTX 3090.
- **More accurate than Scaled FP8**: mean per-step loss deviation from the BF16 reference is 0.273% for ConvRot (bf16 backward) vs 0.289% for FP8, and generated images deviate from BF16 by the same amount FP8's do (1.25 vs 1.23 grey levels out of 255).
- The **demo preset now ships with ConvRot INT8 as the default**, and a new Krea_2_LoRA_Demo_High_VRAM.toml preset (blocks_to_swap = 16) collects 1.72x the throughput of the old FP8 default on 24 GB+ cards.
- Works with block swap, gradient checkpointing and torch.compile — triton ships with the installer on both Windows and Linux.

Here is where it lives in the interface — the Model Settings row with the ConvRot INT8 checkbox and the bf16 / int8 backward selector:

![Krea 2 Model Settings — INT8 ConvRot quantized training controls](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/cOqOGwv2VsApsWdwjBKOH.png)

**Configs, dataset & INT8 ConvRot model settings** — LoRA or Full Fine-Tuning, dataset TOML generation, and the model settings row where ConvRot INT8 lives: one checkbox plus a bf16 / int8 backward selector, next to FP8 Base + Scaled FP8, block swap up to 26 with H2D-only mode, and optional Turbo DiT for training-time samples. The GUI validates every illegal combination (ConvRot + FP8, ConvRot + Turbo sampling, ConvRot + Full Fine-Tuning) before anything launches:

![Krea 2 tab 1/3 — configs, dataset and INT8 ConvRot](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/kwUholbD22YxJDtxBsJGM.png)

**Flow matching, training & sampling** — flow matching and timestep settings with the Krea-2-specific shift schedule, epochs / batch / attention backends, live sample generation, caching and the optimizer catalog:

![Krea 2 tab 2/3 — flow matching, training and sampling](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/oaqJ8lWELGkHLXZs4As9z.png)

**LoRA, advanced & Hugging Face** — LoRA architecture settings, advanced flags, metadata and Hugging Face upload — a single click runs caching and training back-to-back:

![Krea 2 tab 3/3 — LoRA, advanced and Hugging Face](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/w6HX3g-869nNaw4yimRjH.png)

### 7 — LTX 2.3 Video Training (Added in V32)

The LTX tab provides full LoRA training for LTX-2 19B and LTX-2.3 22B text-to-video models — including **audio-video joint training**. The complete pipeline is chained behind one Start Training button: dataset TOML generation from your folders (the 8k+1 frame rule and 25 FPS resampling are handled for you), VAE latent caching, Gemma 3 12B text-encoder caching, then training. The demo preset trains a rank-128 LoRA on 24–32 GB GPUs, and the model downloader grabs the full ~59 GB model bundle with no gated-repo login needed.

**Model, quantization & video dataset** — LTX-2 19B / LTX-2.3 22B checkpoints with automatic version detection, Gemma 3 12B text encoder (single-file FP8 or HF folder with 8-bit / 4-bit loading), video / audio-video / audio-only training modality, first-frame conditioning probability, attention backends with split-attention VRAM control — and the richest quantization panel in the app: FP8 Base / Scaled with keep-blocks and W8A8 activations, **INT8 ConvRot** (quantize at load or load pre-quantized checkpoints, group size, MSE clip, per-layer quality report), NF4, block swap up to 47 blocks with pinned-memory and H2D-only modes, blockwise gradient checkpointing with CPU offload, and low main-RAM loading. Below it, the video dataset generator applies the 8k+1 frame rule, target FPS resampling and automatic captions:

![LTX 2.3 tab 1/3 — model, quantization and video dataset](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/c4qYV1lGvdgWoYGSNSm5G.png)

**Audio, caching, network & training** — audio source selection (video container or external .wav) with separate audio buckets, VAE latent caching with spatial / temporal tiling and chunking, Gemma text-encoder caching with pre-connector features for connector LoRA training, LoRA network settings with target presets (t2v / audio / full) and automatic ComfyUI-format export alongside the musubi format, AdaFactor-tuned optimizer defaults, timestep sampling with shifted_logit_normal, and video sample generation during training with tiled VAE decode, sampling presets and optional audio merge into the preview mp4:

![LTX 2.3 tab 2/3 — audio, caching, network and training](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/crn8JQz7AYlWN2tUzzcyT.png)

**Logging, metadata & one-click start** — TensorBoard / Weights & Biases logging, metadata, Hugging Face upload and the single Start Training button that chains VAE caching → text-encoder caching → training:

![LTX 2.3 tab 3/3 — logging, metadata and one-click start](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/TzcYo7CF-VSp7EnpMDmGx.png)

### 8 — MiniMax H3 Video + Audio Training (NEW in V34)

The newest training tab brings complete **MiniMax H3 LoRA training** to consumer GPUs: text-to-video-with-audio (T2VA), first/last-frame FL2VA and reference-to-video Ref2VA. One Start Training button chains dataset TOML generation, video + audio latent caching, Qwen3-VL 32B text-embedding caching and training.

- **Dataset and audio safety** — batch size is fixed to 1, frame counts are normalized to the `17n+5` rule (released range 124–345), videos are normalized to 24 FPS, and resolutions use 32-pixel steps. Embedded audio or same-stem `.wav` sidecars provide joint audio supervision; silent clips are never trained toward silence.
- **Distillation protection and teacher matching** — the default guidance loss preserves H3's CFG-distilled behavior and automatically creates its unconditional probe cache. Advanced users can select the `first,last` endpoint teacher or the stronger `ref` self-reference teacher and tune sigma protection, timestep focus, DC/magnitude shaping and audio-loss weighting.
- **Consumer-GPU memory stack** — auto-detected approximately 21 GB pruned ConvRot INT8 transformer, approximately 15.7 GB NVFP4+AWQ text encoder with layer-by-layer CPU streaming, and H2D-only swapping for up to 48 of 50 transformer blocks. Full BF16 and compatible ComfyUI pre-quantized ConvRot checkpoints are supported; unsupported FP8 files are rejected before launch.
- **Tested presets** — the 24 GB preset runs 384px / 124-frame training at a measured 22.5 GB peak on an RTX 3090; the lowest-VRAM preset uses 256px / 124 frames and heavy block swapping.

![MiniMax H3 Video Training tab — T2VA, FL2VA and Ref2VA video + audio LoRA training](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/Y3e6_1J65e3Te14ypNeDg.png)

### 9 — Image Captioning

Automatic dataset captioning with Qwen2.5-VL, built in — and far deeper than a simple "caption my folder" button:

- **Single-image and batch captioning** with recursive subfolder scanning, live progress, ETA and a safe stop button.
- **TXT and JSONL output formats** — JSONL mode can optionally copy the images into a self-contained dataset, ready for structured-caption trainings like Ideogram 4.
- **FP8 model loading** to fit the captioner on smaller GPUs, and **automatic model unloading** to release VRAM the moment captioning finishes — so you can go straight into training.
- **Full generation control**: custom system prompt, temperature, top-k, top-p, repetition penalty and maximum tokens.
- **Caption post-processing rules**: prefix, suffix and word replacements — applied in the right order so your trigger words always end up exactly where you want them.
- Skip-existing or append-to-existing modes — recaption safely without destroying manual edits.

![Image Captioning tab](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/VHk22EXu-mdHwDySkCCe7.png)

### 10 — Model Quantizer

Turn your trained checkpoints (or any base model) into deploy-ready quants: FP8 Scaled, INT8 ConvRot, NVFP4, MXFP8. The INT8 Row ConvRot HQ preset produces quants that are **96.2% similar to BF16 while GGUF Q8 manages only 90.0%** — and they run about 2x faster than GGUF Q8 on RTX 3000 / 4000 / 5000. ComfyUI and SwarmUI load them natively. This is a full production quantization suite:

- **13 quality / workflow presets** and **28 model-family profiles** — FLUX, Klein, Krea, Ideogram, Wan, LTX, Qwen Image, Qwen VLM/Qwen3.5+, Chroma/Distilled, Mistral, NeRF, Radiance, visual encoders, Z-Image, Hunyuan, ERNIE, Gemma, T5 and more — each with recommended exclusions and defaults baked in.
- **Learned ConvRot** using SVD and Prodigy optimization, activation calibration and bias correction — the "Best Quality / Slow" preset that beats GGUF Q8.
- Optional **LoRA-informed calibration** — quantize a base model with your LoRA's behavior taken into account.
- **Per-layer control**: layer exclusions, sensitive-layer protection, mixed precision, dry-run analysis and automatic layer-template generation.
- **Single-file conversion queue and recursive folder batch conversion** — both with cancellation — plus low-memory conversion controls for modest machines.

- The **LTX 2.5** model preset creates the video-tuned approximately 22 GB ComfyUI checkpoint directly from the original BF16 dev or distilled model in one streaming pass. It automatically detects the variant and applies the benchmark-validated mixed BF16/INT8 layer plan, architecture checks, per-layer ConvRot group-size search and JSON quality report; older V32.4 workflow configurations migrate automatically.
- **Architecture-aware protection** dynamically preserves the final Qwen language layer, embeddings, MTP weights and visual stack. Pre-quantized ConvRot validation and mixed group-size handling were also strengthened.
- **Cleaner workflow controls** show quantizer panels only when relevant and keep preset/workflow switching consistent. BF16/FP16 output policy and a regex for layers that must retain their source dtype are exposed directly in the GUI.

![Model Quantizer tab](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/Nu2G3yw12oAN381kmomDw.png)

### 11 — LoRA Extractor

Extract a small, distributable LoRA out of any full fine-tuned / DreamBooth checkpoint — keep the full-quality fine-tune for yourself, ship the LoRA. Single-file and recursive batch extraction, configurable SVD rank and clamp quantile, device and precision selection, metadata preservation, and progress with cancellation:

![LoRA Extractor tab](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/ZGxWs5r8jp7GePJ8IhvTW.png)

### 12 — LoRA Merger

Merge up to **three LoRAs simultaneously**, each with its own independent multiplier — LoRA into LoRA, or bake LoRAs directly into a DiT base checkpoint. Single and recursive batch modes, model-specific profiles for Qwen, Wan, SkyReels and custom architectures, plus device and output-dtype controls:

![LoRA Merger tab](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/GXS5xNRKYsH104YW5yfzq.png)

### 13 — LoRA Converter

Move your LoRAs between ecosystems without touching a script: **six bidirectional conversion profiles** covering Diffusers, Musubi, ComfyUI and SwarmUI formats, dedicated HunyuanVideo 1.5 and Z-Image conversions, single-file and recursive folder batch processing:

![LoRA Converter tab](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/G71vAxjHtm8xuqgj4FU4t.png)

### 14 — Image Preprocessing

A unique safety net: this tool runs your dataset through the **actual Musubi Tuner training code** and shows you exactly what the trainer will see — final resolutions, aspect ratios, bucketing decisions. It also catches the classic silent killer: EXIF-rotated images that would train sideways. Preview first, never waste a training run again.

![Image Preprocessing tab](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/STa9YIEOYrDA1PBZe6YmU.png)

### 15 — Version History

The full changelog of all 44 releases through V34 now lives **inside the app** — every version in its own collapsible section with an Open / Close All button, so you always know exactly what changed and when. This app is in constant, active development: what you subscribe to today keeps getting better every single week.

![Version History tab — every release in collapsible sections](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/5rKrK8K-wqUcOLmMx5C6z.png)

---

## Preflight Guardrails — It Refuses to Waste Your GPU Time

The most expensive failure in training is the one you discover 20 minutes in — after caching, after model loading, after VRAM allocation. This app validates your configuration **before** any of that happens, with plain-language messages that tell you exactly what to change:

- Invalid full-fine-tuning and quantization combinations are rejected before model loading.
- FP8 conflicts are detected (e.g. fp8_base without fp8_scaled, FP8 + ConvRot).
- MiniMax H3's unsupported FP8 modes are rejected before caching or model loading; compatible ConvRot INT8 and NVFP4+AWQ formats are auto-detected.
- Unsafe block-swap combinations and per-model block-swap limits are enforced.
- Multi-GPU incompatibilities are caught before the run starts.
- Optimizer and mixed-precision conflicts are explained before VRAM is allocated — including the strict Automagic2 requirements, with a recommendation to switch to Automagic3 when appropriate.
- Krea 2 ConvRot conflicts with FP8, Turbo sampling and full fine-tuning are prevented outright.

## 1-Click Installation — Windows, RunPod, Massed Compute, SimplePod

- **Windows** : run **Windows_Install_and_Update.bat** — that's it. The installer creates an isolated Python 3.12 venv (your system and other apps are never touched), and uses ultra-fast **uv** installation with my pre-compiled wheels. The same bat file also updates existing installs.
- **RunPod / SimplePod** : follow **Runpod_SimplePod_Musubi_Trainer_Instructions.txt** — Python 3.12 is installed automatically. Installation is literally ~100x faster than pip thanks to uv.
- **Massed Compute & local Linux** : follow **Massed_Compute_Instructions_READ.txt**.
- Everything runs on **Torch 2.13 + CUDA 13.1** with pre-compiled FlashAttention, SageAttention, xFormers, TorchAO and mslk — supporting effectively every NVIDIA GPU: RTX 3000 / 4000 / 5000, A40, L40S, A100, H100, B200 and more.

### Windows Requirements

- Python 3.12.10 (3.10 / 3.11 / 3.13 also work), Git, FFmpeg, cuDNN 9.17+, CUDA 13.0, Visual Studio Community Edition with all C++ options, NVIDIA driver 590+.
- Follow this requirements tutorial video exactly : https://youtu.be/DrhUHnYfwC0
- Follow its updated post with links and screenshots exactly : https://www.patreon.com/posts/windows-AI-requirements-tutorial-111553210

### Cloud Registration Links

- **Massed Compute (recommended)** : [register here](https://vm.massedcompute.com/signup?linkId=lp_034338&sourceId=secourses&tenantId=massed-compute) — special SECourses coupon : **SECourses** — how-to starts at 12:58 : https://youtu.be/KW-MHmoNcqo?si=G1WbG-Qw4ujWvOtG&t=778
- **RunPod** : [register here](https://get.runpod.io/955rkuppqv4h) — how-to starts at 22:03 : https://youtu.be/KW-MHmoNcqo?si=QN8X8Sjn13ZYu-EU&t=1323
- **SimplePod (cheaper than RunPod — RTX 5090 at $0.45/hr vs $0.89/hr)** : [register here](https://simplepod.ai/ref?user=secourses) — [use our ready template](https://dash.simplepod.ai/account/explore/100/ref-secourses/) — tutorial from 21:51 : https://youtu.be/yOj9PYq3XYM?si=Z86wZZLBeYzWo1Qo&t=1311

## Built-In Model Downloader — Never Hunt for Model Files Again

- Run **Windows_Download_Training_Model_Files.bat** (or the cloud equivalents) and choose the bundle you need: all Qwen variants, Wan 2.1 / 2.2 T2V and I2V, FLUX 2 Dev, Klein 9B / 4B, Z-Image Base / Turbo, Ideogram 4, Krea 2, LTX 2.3 and **MiniMax H3 Training Models — Low VRAM**. Select several at once with comma lists or ranges — e.g. **1-3,5**.
- The public MiniMax H3 low-VRAM bundle is approximately 42.5 GB and needs no gated-repository login. Optional entries provide the higher-quality ConvRot INT8 text encoder and Ref2VA transformer.
- My own downloader engine (not the Hugging Face client): up to **16 parallel connections** reaching **~1 GB/s on cloud machines**, with an automatic **aria2 → Python range-downloader → Hugging Face hub fallback chain**, retry with exponential backoff, and resume that preserves partial downloads.
- **Integrity you can trust**: SHA-256 (or Git-SHA) verification of every file, immutable Hugging Face revision pinning, and **atomic installation** — an incomplete file can never replace a good one. Already-verified local files are reused across model folders instead of re-downloading.
- **It protects your disk and your time**: total download size is calculated and free disk space checked *before* anything starts; offline and force-redownload modes are available.
- The standard model bundles use BF16 weights where on-load quantization is supported, so Musubi can convert to FP8, FP8-scaled or ConvRot INT8 while loading. MiniMax H3 additionally offers compatible pre-quantized ConvRot INT8 and NVFP4+AWQ files for its consumer-GPU memory path.
- When you load any config, the app validates its model paths and **auto-fills them from the downloader's default locations** — on Windows, Linux and every cloud.

![V34 model downloader interface with MiniMax H3 training models](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/pBTxzQd0e4k1Zt0NtKHYw.png)

## Researched Presets — Not Guesses

- **Qwen Image** : LoRA + Fine-Tuning configs from my 50+ full training research runs, tiered from **6 GB to 96 GB GPUs**, in 75 / 150 / 200-epoch quality variants (folder: `Qwen_Training_Configs`). 30+ result examples: [see this article](https://medium.com/@furkangozukara/qwen-image-lora-trainings-stage-1-results-and-pre-made-configs-published-as-low-as-training-with-ba0d41d76a05).
- **Wan 2.2** : official configs distilled from 64+ unique research trainings — Text-to-Video, Image-to-Video and Text-to-Image, for every GPU tier (folder: `Wan22_Training_Configs`).
- **Krea 2** : the demo preset ships with **INT8 ConvRot as the default** (faster and slightly more accurate than the old Scaled FP8 default at the same VRAM floor), plus the new Krea_2_LoRA_Demo_High_VRAM.toml for 24 GB+ cards at 1.72x the old throughput.
- **MiniMax H3** : two tested rank-128 LoRA presets — `MiniMax_H3_LoRA_Demo_24GB.toml` for 384px / 124 frames and `MiniMax_H3_LoRA_Demo_Lowest_VRAM.toml` for 256px / 124 frames with aggressive H2D-only block swapping.
- **Everything else** : ready demo presets for FLUX 2, Klein 4B/9B, Z-Image Base/Turbo, Ideogram 4 and LTX 2.3 (folder: `Demo_Training_Configs_FLUX-2_Z-Image_FLUX-Klein_WAN-21_Krea2_Ideogram4`).
- Sample generation during training, TensorBoard / Weights & Biases logging with gradient statistics, exact mid-epoch training resume, checkpoint retention policies and direct Hugging Face upload are all built in.

## Quality of Life You Will Not Find Anywhere Else

- **Search bar over 500+ settings** — type "block swap" and jump straight to the option, no panel hunting.
- **Tooltips everywhere** — every single parameter documented in plain language, with recommended values.
- **Config save / load** with automatic validation, model-path auto-fix and cross-platform path normalization (spaces in paths just work). Runtime TOMLs preserve LoRA versus Full Fine-Tuning mode, older configs infer it safely, and reloading a saved or failed full-finetune run no longer falls back to LoRA.
- **Dataset folder safety** — forget the N_ repeat prefix on a training folder and the generator counts it as 1 repeat and tells you exactly how to fix the name; a 0_ prefix (which would silently skip the dataset) is caught and corrected too, and every generated dataset reports its final num_repeats so you can verify at a glance.
- **Built-in dataset guides and normalization** — the Wan tab renders a complete dataset reference, while MiniMax H3 automatically enforces batch size 1, `17n+5` frame counts, 24 FPS and 32-pixel resolution steps.
- **Errors and warnings surface as Gradio notice bubbles** — no more scanning terminal walls.
- **Accurate step-speed display** almost immediately after training starts (fixed in my fork).
- **Automagic 1 / 2 / 3 optimizers** (from Ostris AI-Toolkit) with adaptive learning rates — plus the full standard optimizer catalog, each with guidance and unsafe-combination validation before training starts.
- **Guarded SDPA acceleration on Windows** — external FlashAttention is used only after passing real CUDA forward/backward probes, with automatic fallback, so you get maximum speed without mystery crashes.
- **The active quantization is always visible** — training stdout states exactly which base quantization is live (e.g. "base weights quantized to ConvRot INT8 (224 Linears, fused Triton INT8 GEMM)"), so a silently slow fallback can never hide.

---

## Who Am I and Why Trust This

I am Dr. Furkan Gözükara (PhD in Computer Engineering) — SECourses is my full-time work: tutorials, research and production-grade AI apps. This trainer alone has shipped **44 releases in 11 months**, each one announced with full changelogs (now built into the app itself). My configs are not copied defaults — they come from hundreds of paid-GPU research trainings that I publish and document. When something breaks upstream, my fork usually has the fix before the issue is even triaged.

## What Your Subscription Gets You

- The full **SECourses Musubi Trainer** app with 1-click installers (this post's zip file, updated constantly).
- All researched training configs — Qwen tiers, Wan 2.2 official configs, and demo presets for every supported model.
- Every other SECourses app and script in the [Patreon posts index](https://github.com/FurkanGozukara/Stable-Diffusion/blob/main/Patreon-Posts-Index.md) — SwarmUI installers, ComfyUI installers, captioners, upscalers and much more.
- Private Discord support — tell me your username and get your special rank : [SECourses Discord](https://discord.com/servers/software-engineering-courses-secourses-772774097734074388).
- Constant updates: when a new model drops, this app gets it fast — Krea 2, Ideogram 4, LTX 2.3 and MiniMax H3 were all added rapidly, with LTX 2.5 quantization following in V34.

**[→ Join SECourses on Patreon and download SECourses_Musubi_Trainer_v34.zip now ←](https://www.patreon.com/cw/SECourses/membership)**

---

## How To Install and Use

### Windows

- Install the requirements (see the requirements section above), then just run **Windows_Install_and_Update.bat** for install and update.
- Run **Windows_Download_Training_Model_Files.bat** to download the exact right model files for your training.
- Start the app with **Windows_Start_App.bat**, load a preset config matching your GPU, generate your dataset TOML with the Generate Dataset Configuration button (Kohya folder logic, e.g. parent folder > `1_ohwx man`), and press Start Training.
- Follow the main tutorial for the complete workflow : https://youtu.be/DPX3eBTuO_Y

### Massed Compute (Recommended Cloud)

- Register via [this link](https://vm.massedcompute.com/signup?linkId=lp_034338&sourceId=secourses&tenantId=massed-compute) — coupon **SECourses** for all GPUs. GPU guide : https://www.patreon.com/posts/126671823
- Select RTX A6000 or better (L40S, A6000 ADA, A100, H100, RTX 6000 PRO), pick the **SECourses** image from the Creator dropdown, then follow **Massed_Compute_Instructions_READ.txt**.

### RunPod / SimplePod (Cloud)

- Register via [RunPod](https://get.runpod.io/955rkuppqv4h) or [SimplePod](https://simplepod.ai/ref?user=secourses) and follow **Runpod_SimplePod_Musubi_Trainer_Instructions.txt** — use the template written in the instructions file.

### Auxiliary Tools & Resources

- Ultimate Batch Image Preprocessing app : https://www.patreon.com/posts/120352012
- Batch Image Caption Editor app : https://www.patreon.com/posts/108992085
- How to install and use SwarmUI with ComfyUI backend : https://youtu.be/c3gEoAyL2IE
- Extremely fast Hugging Face upload / download notebook : https://youtu.be/X5WVZ0NMaTg
- Training style images dataset : [GTA5_Style_Dataset.zip](https://www.patreon.com/file?h=137551634&m=556630570) — model : [on CivitAI](https://civitai.com/models/2084406?modelVersionId=2358426)
- Example training images dataset : https://www.patreon.com/posts/114972274

## Version History

The complete changelog of all 44 releases from V1 (28 August 2025) through V34.0 (14 August 2026) is built into the app — open the **Version History** tab and expand any release. Recent highlights:

- **V34.0** — MiniMax H3 T2VA / FL2VA / Ref2VA video + audio LoRA training; consumer-GPU ConvRot/NVFP4 memory stack and demo presets; LTX 2.5 quantization; quantizer upgrades; full-finetune config safety and H3 reliability fixes.
- **V32.3** — FLUX 2 Klein training fix and related demo-preset updates.
- **V32.1** — Interface fully modernized; Automagic optimizers fully working; dataset repeat-count safety (forgotten N_ folder prefixes counted as 1 repeat with a clear notification); full version history moved into the app.
- **V32** — LTX 2.3 (22B) / LTX-2 (19B) video training tab with INT8 ConvRot and audio-video modes.
- **V31** — Krea 2 INT8 ConvRot quantized training (faster than BF16, more accurate than Scaled FP8).
- **V30** — Massive performance release: up to 53% faster; Full Fine-Tuning for FLUX 2, Klein, Ideogram 4 and Krea 2; guarded Windows SDPA.
- **V29** — Torch 2.13 + CUDA 13.1 stack with self-compiled abi3 wheels; Krea 2 and Ideogram 4 training added; auto model-path detection.
- **V25 → V27** — uv installers (~100x faster), Qwen Edit 2511, Wan 2.2 official configs, FLUX / Z-Image tabs, Model Quantizer.

**Ready to train? [Join SECourses](https://www.patreon.com/cw/SECourses/membership), grab [SECourses_Musubi_Trainer_v34.zip](https://www.patreon.com/file?h=137551634&m=717993383), watch [the main tutorial](https://youtu.be/DPX3eBTuO_Y) — and your first LoRA can be training within the hour.**
