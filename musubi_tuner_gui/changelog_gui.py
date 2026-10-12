import gradio as gr

# Each entry: (accordion title, markdown body).
# Newest first. The newest entry is rendered open by default.
CHANGELOG_ENTRIES = [
    (
        "V37.0 — 12 October 2026 — DoRA for Every Model, Best Checkpoint Finder, Automagic EXPERIMENT, Smarter Dataset Check",
        """
**🧬 DoRA training for every model with a one-click LoRA / DoRA switch, DoRA files that load directly in ComfyUI and SwarmUI, a 🏆 Best Checkpoint Finder that picks the checkpoint for you, the AutomagicEXPERIMENT optimizer on every pipeline, and a dataset check that no longer skips folders.**

- **🏆 Best Checkpoint Finder** (image and video models, off by default): turn it on, keep or edit the test prompts (preset from the `Grid_Find_Best_Checkpoint` tutorial prompts) and set your subject (read from your `10_ohwx man` folder with one click). The run renders the prompts with the base model and at every checkpoint save with the same seeds; when training ends, local models score every checkpoint for likeness to your training photos (face recognition, or image similarity without faces) and how far the faces moved away from the base model's own person, prompt following and over-training signs (the subject drawn as close as in the training photos whatever the prompt asks, copies of training photos, colour or sharpness drift), each compared with the base model's sample of the same prompt and seed, and the finder names the checkpoint with the most likeness before over-training starts (calibrated on Krea 2 runs judged by eye) with its reasons, a trend chart with the over-training onset marked, a score table, a base / best / last gallery and a full HTML report; a run whose likeness never rose above the base model is called undertrained. 🔍 Analyze scores any finished run that saved samples. The scoring models (about 4.7 GB) download on first use to `Training_Models_Checkpoint_Finder`; edit models are skipped (their samples need control images).
- **DoRA (Weight-Decomposed LoRA) for every model**: Qwen-Image, Qwen-Image 2.1, FLUX.2 Dev / Klein, Z-Image, Krea 2, Ideogram 4, ERNIE-Image, Mage-Flow, HiDream-O1, Wan 2.1 / 2.2, LTX 2.3, MiniMax H3, Kandinsky 6 and FLUX.1 / SDXL (kohya). **Adapter Type** (LoRA / DoRA) is the first setting of the 🎯 LoRA Network section. DoRA learns a magnitude for every input channel next to the LoRA matrices, which usually follows the trained subject more faithfully at the same rank. Every shipped LoRA preset now trains DoRA, except the Wan 2.2 presets that train the high- and low-noise experts in one run (DoRA learns its magnitudes for one base model; the UI explains this if you select DoRA there). Switch back to LoRA at any time: the choice is stored in Network Arguments as `use_dora=True`, so presets stay readable by the classic tabs and the command line. Measured on every model's 32 GB preset (RTX 5090): DoRA takes about 10% more time per step than LoRA (median of 18 models; 3-19% on most, 29-43% on the small fast Mage-Flow, HiDream-O1 and LTX 2.3 presets) and the same VRAM within about half a GB.
- **Works directly in ComfyUI and SwarmUI**: the file stores `<layer>.dora_scale` in the input-axis layout ComfyUI (and Forge / A1111) apply exactly; every model ComfyUI supports was checked with ComfyUI's own LoRA loader (Kandinsky 6 has no ComfyUI support yet); FLUX.2, Qwen-Image, Krea 2, Ideogram 4, ERNIE-Image, Mage-Flow, Wan, LTX 2.3, MiniMax H3 and FLUX.1 load in stock ComfyUI. LTX 2.3 still writes its `.comfy.safetensors`, now with DoRA. Z-Image, Qwen-Image 2.1 and HiDream-O1 need the updated **SECoursesAudioTools** nodes, which make ComfyUI load layer names it used to skip, for LoRA files too: Qwen-Image 2.1's MLP layers, Z-Image's q / k / v (no conversion needed anymore) and every HiDream-O1 layer.
- **Lean DoRA implementation**: no merged weight is ever built; BF16, FP8, FP8-scaled and ConvRot INT8 base models, block swap, gradient checkpointing and torch.compile all work, and layers that share one input keep it once, as with LoRA.
- **AutomagicEXPERIMENT optimizer** (from ai-toolkit) next to Automagic v1-v3 on every training pipeline, including FLUX.1 / SDXL: Automagic v3's learning-rate controller with a full per-element second moment stored as ConvRot INT8 (about 2 bytes of optimizer state per trained value). Fused or non-fused mode is chosen automatically like Automagic v3.
- **🔍 Check dataset**: a training folder that contains other folders (old caches, control images, extracted audio) is no longer reported as skipped: only the files directly inside it are trained and the inner folders are listed. A **🔁 Repeats & steps** panel explains each folder's repeat count and computes samples per epoch, steps per epoch and total steps. **🧹 Clean old caches** deletes cache folders left by other resolutions or models after a confirmation, never images or captions.
- **Arguments typed any way**: Optimizer, LR Scheduler and Network Arguments accept spaces, commas, new lines, quotes and `key = value`; `betas=0.9 0.999`, `true` / `false` / `none` and decimal commas reach the trainer as intended.
- **Presets and model files**: a fresh install lists the shipped presets; loading a preset fills its model paths from the `Training_Models_*` folders next to the app on Windows and Linux (relative paths included); the last preset loads at start; older presets are recognized correctly.
- **Hardware & Launch**: GPU IDs accept `0 1`, `0;1` or `[0, 1]`; extra accelerate arguments keep their quotes.
- **2 px larger fonts** everywhere (14 px base) for readability.
- **Libraries**: transformers 5.19, diffusers 0.41, newer accelerate / safetensors / huggingface_hub / PyAV; unused packages removed.
- **Training samples**: Krea 2 and Qwen-Image 2.1 samples generated during training now use CFG like the official samplers (they were generated without it, so RAW Krea 2 samples looked much worse than the trained LoRA), samples show the trained weights without LoRA dropout (network / rank / module dropout made them random), a step that ends an epoch is sampled once, and the Krea 2 Turbo sampling and Ideogram 4 unconditional-DiT sampling options work in LoRA training (they were ignored). The musubi generation scripts (Qwen-Image, Qwen-Image 2.1, FLUX.2, Z-Image, ERNIE, Krea 2, Wan) load DoRA files with their magnitudes.
- **Linux (Massed Compute, RunPod)**: paths typed or pasted from Windows (backslashes, a trailing separator, `..\\Training_Outputs`, `file:///` URLs) are normalized, and cached latents are read in the same order as on Windows, so the same seed trains the same batches. The full test suites pass on Ubuntu.
- **Fixes**: the FLUX.1 FP8 Scaled presets for 6, 8, 10, 12 and 16 GB GPUs put the whole FP8 model on the GPU while loading, before block swap moved it out: the 6 GB preset peaked at 10.7 GB (Windows spilled into system RAM, Linux ran out of memory); it now loads in its 4.7 GB (sd-scripts `load_flow_model`); every FLUX.1 preset was re-measured with DoRA and records its peak, and the 24 GB FP8 Quality 2 preset is now named 21.03 GB; Wan image-to-video training on a dataset cached for text-to-video stopped at the first step (`KeyError: 'latents_image'`): the cache is now re-encoded with the first-frame latents; LTX 2.3 training with sample generation stopped at the first sample ("Attempted to access the data pointer on an invalid python storage") because the sampler opened the model file a second time; FLUX.1 LoRA with `split_qkv=True` stopped at network creation in sd-scripts; the LoRA Merger refuses to add DoRA files key by key (it would multiply the base weights) and points to merging into the model; settings that only apply to some modes or values (sigmoid scale, mode scale, base weights, data-loader workers) appear only where they take effect.
""",
    ),
    (
        "V36.0 — 12 October 2026 — New Unified Interface, SDXL / FLUX.1 Training, Dataset Check, VRAM Panel",
        """
**🧭 One workspace per category instead of one tab per trainer: pick the base model, choose LoRA or Fine-Tuning, and the measured preset for your GPU loads. The classic per-model tabs are still available with `python gui.py --classic`.**

- **Five tabs**: 🖼️ Image Models, 🎬 Video Models, 🔊 Audio Models, 🧰 Tools, 📜 Version History. Each model tab shows the base models as cards, the training types the model supports, the GPU preset picker (auto-detected tier), model tips and every setting in collapsible sections (advanced settings hidden until asked for, a settings search, expand / collapse all).
- **Presets moved into `Training_Presets`**: every `<Model>_Training_Configs` folder now lives in `Training_Presets` next to the app folder (older installs that keep them in the install root are still found). Your own presets are saved by name into `user_presets/<model>` or to any full path, the last preset per model and training type reloads at start, and a preset made for another model, training type or category is detected before it is applied.
- **SDXL and FLUX.1 Dev training** through kohya sd-scripts (the premium fork with transformers 5 support, installed into `sd-scripts`): the proven DreamBooth / LoRA presets for every GPU tier, converted from the kohya_ss configs, with the same effective sd-scripts arguments. kohya_ss JSON configs and runtime TOMLs load directly.
- **Live Training Monitor** under Start: stages (caching, training, merging), step progress, epoch, loss chart, learning rate, speed, ETA, VRAM of the training GPU and the last 100 log lines; it recovers after a page reload and a stopped run never leaves its question behind. Model downloads show their log in the Model Files section.
- **🔍 Check dataset**: statistics, every problem with example files (missing or unreadable captions, unpaired or misnamed reference images, unreadable media, wrong folder names) and sample previews of what the selected model trains on: edit pairs side by side, a video's first frame, an audio waveform. Works for dataset folders, musubi dataset TOMLs and kohya folder / TOML datasets.
- **⚡ VRAM panel**: changing resolution, frames, frame rate, batch size, control images, quantization, block swap, checkpointing, optimizer or low-VRAM text caching shows what it does to VRAM, compared with the preset's measured peak and your GPU, before Start.
- **Settings reach the trainer exactly**: every value is written in the type the trainer reads (the runtime TOML skips argparse conversion), paths pasted with quotes or either slash are cleaned, `1e-4`, `0,0001` and `10,000` are understood, and Start / Print Command list every unusable setting by name before anything runs.
- **Settings that change together**, as in the classic tabs: HiDream-O1 dev selects its own checkpoint and noise schedule, Mage-Flow Edit its edit checkpoint, Wan FP8 Scaled follows FP8 Base, Klein sample defaults follow the model version, frame counts round to the model's grid.
- **Every dataset a model supports**: video models train images, videos or both (also from one folder); LTX 2.3 Audio trains on audio clips and the sound of your videos (extracted automatically).
- **Fixes**: the Model Quantizer's timer runs only during a conversion; the Wan tab refuses more swapped blocks than the model allows before caching; tips quote the measured presets (block-swap limits, VRAM, resolutions).
""",
    ),
    (
        "V35.0 — 11 October 2026 — 5 New Models, Measured GPU-Tier Presets for Every Model, LTX 2.3 Full Fine-Tuning",
        """
**🚀 Five new model families, a measured 6-32 GB preset library for every model, LTX 2.3 full fine-tuning, and a faster, reorganized interface.**

- **New models, each with its own tab** (Image Models / Video & Audio Models groups):
  - **Qwen-Image 2.1** (Sept 2026) — LoRA and Full Fine-Tuning, editing LoRAs with ordered control images, RGBA VAE, Qwen3-VL 8B text encoder.
  - **ERNIE-Image** (Baidu, 8B DiT, Ministral 3 text encoder, FLUX.2 VAE) — LoRA.
  - **Mage-Flow** and **Mage-Flow-Edit** (Microsoft, 4B, MIT license) — LoRA, native-resolution text-to-image and instruction editing with 1-3 reference images.
  - **HiDream-O1** (pixel-space Qwen3-VL model, one checkpoint, no VAE/text encoder) — LoRA and Full Fine-Tuning, t2i and i2i.
  - **Kandinsky 6 Lite** — video + audio LoRA (text-to-audio-video and first-frame conditioned).
- **Measured GPU-tier presets for every model** — 6, 8, 10, 12, 16, 24 and 32 GB tiers for the new models and for FLUX.2 Dev, FLUX.2 Klein 9B / 4B, Z-Image Base / Turbo, Krea 2, Ideogram 4, Wan 2.1, LTX 2.3 and MiniMax H3, LoRA **and** Full Fine-Tuning wherever the backend supports it (a tier ships only when a configuration really fits it; each folder's README lists the tiers that cannot). Every preset was trained on a real GPU with VRAM sampled every 100 ms; **training and caching both keep at least 1.5 GB of the tier free** (a 32 GB preset peaks at 30.5 GB or less). File names carry the tier and the measured peak (`24_GB_GPUs_21.3_GB_Quality_1.toml`), each preset starts with a description line, and every model folder has a README table with peaks and speeds. Presets now live in one folder per model (`Qwen_Image_2.1_Training_Configs/LoRA_Training`, `FLUX_2_Training_Configs/Fine_Tuning_Training`, ...), like `Wan22_Training_Configs`.
- **⚡ GPU preset picker on every trainer tab**: filter by model family, training type and GPU VRAM (auto-detected), the best preset for your GPU is pre-selected, and Apply loads it exactly like the Load button. Shipped presets are read-only, so Save can never overwrite them. One compact row (GPU VRAM · preset · Apply); the model and training-type filters only appear when there is a choice, and the hand-made Qwen Image presets (`Tier1_15150_MB...`) are pre-selected by their measured VRAM. Switching the tab's Training Mode (LoRA / Full Fine-Tuning) switches the picker to the matching presets.
- **LTX 2.3 Full Fine-Tuning**: new Training Mode in the LTX tab (dense BF16, fused-backward Adafactor, trainable block-swap ring) and an automatic post-training merge into an inference-ready single-file LTX bundle.
- **Live training monitor** under Start Training on every tab: current phase (caching latents, caching text, training, merging), elapsed time and live VRAM per GPU with the peak seen.
- **One-click model downloads inside the new tabs** (⬇️ Download Model Files streams the downloader log into the tab), plus new downloader entries 20-26 (Qwen-Image 2.1, ERNIE-Image, Mage-Flow, Mage-Flow-Edit, HiDream-O1 full/dev, Kandinsky 6 Lite).
- **Faster, smoother UI**: grouped navigation, lazily rendered tabs, and Gradio run history disabled (it was serializing 200+ settings into browser storage on every click).
- **Backend**: musubi-tuner upgraded to upstream 0.3.6 + dev, transformers 5.17 / diffusers 0.40 / accelerate 1.15, plus Qwen-Image 2.1 (PR #1145), ERNIE-Image (PR #934), Mage-Flow (PR #1010), Kandinsky 6 (AkaneTendo25), INT8 ConvRot cuBLAS backend (#1155), offloader record-stream fix (#1150) and corrupted-cache handling (#1088). New in the fork: **Mage-Flow activation CPU offload** (it raised "not implemented" before), which makes a 6 GB Mage-Flow preset possible. **Changing the training resolution no longer reuses old latents**: image caches are named by the source image size, so with *skip existing* a 768 px run silently trained on the 1024 px latents cached before — caches now record their bucket and are re-encoded when it changes, and video training ignores clips cached for another resolution or frame count instead of mixing them in.
- **Fixes found by the preset measurements**: transformers 5 broke HiDream-O1 (RoPE + causal mask) and fresh LTX-2 Gemma text caching; HiDream-O1 text caching lost reference-image support in the merge; Kandinsky 6 training skipped its audio inputs; Mage-Flow used the wrong network module name; Qwen-Image 2.1 and HiDream-O1 allowed one block swap too many for full fine-tuning; new tabs could inherit a stray legacy `config.toml`. **LTX 2.3 with block swap put the whole 22B model on the GPU while loading** (25.8 GB at every block-swap setting); its quantized weights now stay on the CPU, and the loader skips the ~18.5 GB of audio / VAE / vocoder tensors in the single-file checkpoint that a video-only run never uses, which is what makes the LTX 2.3 presets below 32 GB possible. **FLUX.2 block swap now reaches every swappable block** (Klein 9B stopped at 16 instead of 18, FLUX.2 Dev at 29 instead of 30), and too-high values name the maximum. Ideogram 4 LoRA with activation CPU offload stopped with "Expected all tensors to be on the same device"; swapping all 33 Ideogram 4 blocks works only with H2D-only block swap and is now refused otherwise. MiniMax H3 with the standard block swap and the guidance loss stopped at the first step ("block 0 ... is on cpu"). Kandinsky 6 latent caching planned its VAE tiles for all free GPU memory (6.7 GB at 512x288 while training needs under 4 GB); the new *VAE Tiling Memory (GB)* option caps it. LTX 2.3 LoRA with exactly one swapped block silently corrupted training (loss ~1.8 instead of ~0.3) and is now refused with a clear message. **LTX 2.3 text caching needed 13.3 GB of VRAM** even where training needs 8 GB: the FP8 Gemma weights were loaded onto the GPU before being offloaded, every prompt computed full-vocabulary logits nobody uses, and the text projection copied all 49 Gemma layers in float32. It now needs 10.8 GB. The new *Low-VRAM Text Caching* option (TE caching panel) needs 5.3 GB by moving the embedding table, the projection and the connector to the GPU one at a time. That gives LTX 2.3 LoRA and full fine-tuning their first 10 and 12 GB presets. Z-Image latent caching of a 1536 px image reserved about 20 GB because the VAE's attention built a full score matrix; it now needs about 9 GB with the same latents. The cached embeddings are bit-identical. INT8 ConvRot training could stop with "CUBLAS_STATUS_NOT_SUPPORTED" on RTX 30xx / A-series (Ampere) GPUs when the token count was not a multiple of 32 (LTX 2.3 at 512x384). Applying a FLUX.2 Klein preset in the FLUX tab could be rejected by the model-version dropdown.
- **Settings and dataset fixes**: **a path with a folder or file name starting with "x"** (`C:\\Users\\xavier\\...`, `D:\\xl_models\\...`) was saved as invalid TOML, so the preset no longer loaded and training could not start; such paths are written correctly now (in presets, runtime and dataset TOMLs), and files saved that way by older versions load again. In the LTX 2.3 and MiniMax H3 tabs a fractional *Network Alpha* (0.5) became 0, which the trainer reads as alpha = rank, and a warmup / decay given as a fraction of all steps (0.1) became 0; both reach the trainer now. The GPU presets save a checkpoint every 25 epochs (8 for a 200-epoch run) and keep only the newest training state. The V34 demo presets they started from saved every epoch, and full fine-tuning also saved its optimizer state, so up to 200 full checkpoints per run. Loading a run TOML with several base weights showed them as a Python list in the FLUX tabs. The Model Quantizer's status poll no longer keeps running in the background on every tab. Switching LTX 2.3 from Full Fine-Tuning back to LoRA turned on gradient clipping (1.0), which a fresh LoRA tab doesn't use; it now restores the tab's own LoRA defaults. The Wan tab accepted up to 59 swapped blocks and only failed in the trainer after caching; it now refuses more than the model allows (38 for 14B / A14B, 28 for 1.3B) at Start. A single *Base Weights* path reached the backend as text, which then read it character by character; base weights and their multipliers are now written as lists (several paths separated by `;` or one per line, multipliers by spaces, decimal commas like `0,5` accepted), and the Wan tab keeps every one of them when a config is loaded. Paths pasted with quotes (Explorer's *Copy as path*) work in the model, dataset and output path fields. Start regenerates the dataset TOML when the Parent Dataset Folder or the resolution changed since *Generate* (it used to train the old folder), and Z-Image does so in folder mode too. Folders that mix images and videos are no longer skipped after their first caching (the cache's `_images` folder counted as a foreign sub-folder), and Wan trains their images as well. The Wan tab no longer loads a saved 0 (block swap, VAE chunk size, tile size, timestep buckets) as empty. Shipped presets stay read-only however the path is capitalized. LTX 2.3 Full Fine-Tuning turns the LoRA-only base-model quantization off.
- **More fixes from testing every tab like a user**: editing / i2i datasets without the reference sub-folder are now rejected before caching (Mage-Flow-Edit used to start without references); ⬇️ Download Model Files keeps the model file an applied preset names (the Mage-Flow-Edit DiT) instead of the tab default; the app no longer aborts at launch with "localhost is not accessible" when the machine is busy (Gradio allowed the first page request only 3 s); example dataset TOMLs are no longer listed as presets; the LTX 2.3 and MiniMax H3 tabs open with the Accelerate panel collapsed like every other tab; the Qwen Image tab's stale "Custom configuration loaded" banner is gone; switching a new-model tab to Full Fine-Tuning now applies that model's fused-Adafactor recipe (it only swapped the panels before); MiniMax H3 latent caching from a video folder failed with NotImplementedError after the upstream merge (the fork's video datasources lacked upstream's item extras); H2D-only block swap combined with activation CPU offload or Sample With Offloading is now refused in the tab with a clear message instead of failing after caching.
- **HiDream-O1 full fine-tuning on 16 GB-class GPUs**: new *Freeze Text Token Embeddings* option (fork) — the resident 0.6B-parameter token table's update alone peaked at 15.5 GB at any resolution; every transformer layer still trains. Text-to-image block-swap runs also keep the unused 0.57B-parameter vision encoder on the CPU (about 1.15 GB less VRAM), which brings HiDream-O1 LoRA down to 6 GB GPUs.
""",
    ),
    (
        "V34.0 — 14 August 2026 — MiniMax H3 Video+Audio Training",
        """
**🎬🔊 Full MiniMax H3 text-to-video-with-audio (T2VA), first/last-frame (FL2VA), and reference-to-video (Ref2VA) LoRA training.**

- **New "MiniMax H3 Video Training" tab**: complete pipeline from dataset TOML generation (17n+5 frame rule, 24 fps timestamp normalization, and the hard batch-size-1 rule handled automatically) through video+audio VAE latent caching and Qwen3-VL 32B text-embedding caching to LoRA training with `networks.lora_minimax_h3` — all chained from one Start Training button.
- **Joint audio supervision**: clips with a real audio track (or a same-stem `.wav` sidecar) train the audio stream too; silent clips are never supervised toward silence. The cache reports the supervised-audio fraction, and `video_only` / `audio_loss_weight` controls are exposed.
- **Guidance-distillation countermeasure enabled by default** (scale 4.0, sigma-min gate 0.15): H3 checkpoints are CFG-distilled and plain flow training slowly washes them out; the guidance loss re-anchors the target in the distilled space. The required uncond probe cache is generated automatically during text-encoder caching — nothing to configure.
- **Teacher matching mode** (t2va) with **two teachers**: the FL2VA endpoint teacher (`first,last` — conditioned on the real first/last frames; the tab automatically builds the FL2VA latent caches and dual-presentation text caches it needs) and the new **reference teacher** (`ref` — the frozen base copies the training clip itself via the Ref2VA layout: complete information at every sigma, a 3-5x lower teaching-band floor, and **audio becomes a real teaching target**). Sigma-max band protection, DC/magnitude loss shaping, and timestep-focus controls are exposed for both.
- **Consumer-GPU memory stack**: pruned ConvRot INT8 transformer (≈21 GB, auto-detected — no flags), NVFP4+AWQ text encoder (≈15.7 GB) streamed layer-by-layer from CPU (`text_encoder_blocks_to_swap`, 50 = minimum VRAM), H2D-only block swap of up to 48/50 DiT blocks, and on-the-fly `convrot_int8` / `prune_adaln` for full BF16 checkpoints. FP8 is not supported by H3 and is blocked with a clear error.
- **Model downloader**: new "MiniMax H3 Training Models (Low VRAM)" bundle (≈42.5 GB total, public Comfy-Org repo, no login), plus optional higher-quality ConvRot INT8 text encoder and Ref2VA transformer entries.
- **Demo presets** (both rank-128 LoRA @ LR 1e-4, AdamW8bit, guidance loss 4.0): `MiniMax_H3_LoRA_Demo_24GB.toml` (384px @ 124 frames, H2D swap 40 — measured 22.5 GB peak and ≈24 s/it on an RTX 3090) and `MiniMax_H3_LoRA_Demo_Lowest_VRAM.toml` (256px @ 124 frames, H2D swap 48 ring 1 — ≈8.5 s/it on an RTX 5090).
- **Backend**: merged the complete upstream `feat-h3-teacher-matching` branch (40+ commits: H3 transformer/VAEs/text encoder, packing, dual flow schedules, ConvRot INT8 + NVFP4 support, TE layer streaming, guidance loss, teacher matching) into our musubi-tuner fork while keeping every existing model family unchanged — 758 backend and 201 GUI tests green.
""",
    ),
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
