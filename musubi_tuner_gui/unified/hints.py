"""Model tips shown in the unified UI's "Model tips" card (under the base-model picker).

Pure data, keyed by ``registry.ENTRIES`` key (FLUX.1 Dev and SDXL tips live in another module).
Each entry has ``about`` (one sentence), ``all`` (tips for every training type), ``lora`` / ``full``
(only for the training types the entry supports), ``vram`` and ``dataset`` (one line each; ``vram``
may instead map each training type to its own line). The card shows every tip of a list.

Every number comes from the shipped presets (``Training_Presets/<Model>_Training_Configs``: file
names, their description headers and README tables), from the defaults that load when a model and
training type has no preset (``Backend.defaults`` plus the entry's ``default_overrides``), or from
the trainer / classic-tab code (block-swap limits, validation rules). Tips that no source supports
are left out on purpose; a training type without a shipped preset says so instead of quoting
measured numbers.
"""

MODEL_HINTS: dict[str, dict] = {
    # ------------------------------------------------------------------ image models
    "krea2": {
        "about": "Photoreal and artistic text-to-image model with a full-step RAW checkpoint for training and a distilled Turbo checkpoint for fast inference.",
        "all": [
            "Keep timestep sampling on krea2_shift: it applies Krea's resolution-aware shift per image, so mixed resolutions need no fixed flow shift.",
            "RAW samples need CFG (tab default CFG 5.5, 28 steps; without a negative prompt an empty one is used, as in Krea's sampler); CFG-off RAW output is blurry by design.",
            "Up to 26 blocks can be swapped; lowering block swap is where most of the speed comes from when VRAM allows.",
        ],
        "lora": [
            "Recommended workflow: train the LoRA on RAW and use it on Turbo; set a Turbo DiT only for Turbo previews (CFG 1, about 8 steps).",
            "Every LoRA preset uses ConvRot INT8: FP8-class VRAM but faster and closer to BF16; it cannot be combined with FP8 or a Turbo sample DiT.",
            "Turbo sample generation cannot run with block swap, so leave the Turbo DiT empty when blocks are swapped.",
            "FP8 Base and Scaled FP8 must be enabled together on Krea 2; plain FP8 is rejected.",
            "LoRA presets: rank 128, AdaFactor LR 6e-5, 1024 px, 200 epochs, a checkpoint every 25 epochs; 12 GB and below lower the rank (and the resolution on 10 and 8 GB).",
        ],
        "full": [
            "Fine-tuning presets: full BF16 weights, fused-backward AdaFactor at LR 1e-5 with warmup, 1024 px (512 px on 12 GB), 200 epochs, a checkpoint every 25.",
            "FP8, ConvRot INT8 and H2D-only block swap are LoRA-only; fine-tuning trains unquantized weights.",
            "Set Krea 2 DiT Variant to match the checkpoint: raw (default), or turbo when fine-tuning the Turbo model.",
            "Samples come from the model being trained; a Turbo fine-tune samples with a fixed shift, so use CFG off and few steps.",
        ],
        "vram": {
            "lora": "ConvRot INT8 LoRA at 1024 px, rank 128: 23 GB peak without block swap (32 GB), 22.3 GB with block swap 3 and 14.2 GB with block swap 23, both H2D-only (24 / 16 GB); 12-8 GB lower rank or resolution, down to 6.1 GB.",
            "full": "Fine-tuning at 1024 px: 30.1 GB with block swap 5 (32 GB), 22.1 GB with block swap 15 (24 GB), 14 GB with block swap 25 (16 GB); 12 GB trains 512 px (10.4 GB); no preset fits 10 GB or less.",
        },
        "dataset": "1024x1024 images in Kohya folders such as 1_ohwx, each with a same-name .txt caption.",
    },
    "flux2_dev": {
        "about": "Black Forest Labs FLUX.2 dev text-to-image model with the Mistral 3 text encoder and strong prompt adherence.",
        "all": [
            "Keep timestep sampling on flux2_shift, the schedule the tab recommends for FLUX.2.",
            "FP8 text-encoder options are not available for dev (Mistral 3); they only work on FLUX.2 Klein.",
            "An optional control_images sub-folder in a dataset folder adds reference images for Kontext-style edit training.",
            "Up to 30 blocks can be swapped (the dev maximum); fewer swapped blocks train faster when VRAM allows.",
            "The presets cache the Mistral 3 text encoder on the CPU, which needs about 50 GB of free system RAM.",
        ],
        "lora": [
            "LoRA presets: rank 128, AdaFactor LR 6e-5, 1024 px, FP8 Base with Scaled FP8, 200 epochs, a checkpoint every 25 epochs; 16 GB trains 768 px at rank 32 with activation CPU offload.",
            "Enable Scaled FP8 together with FP8 Base, as the FLUX.2 docs recommend.",
        ],
        "full": [
            "Fine-tuning presets (32 and 24 GB): full BF16 weights, fused-backward AdaFactor at LR 1e-5 with warmup, 1024 px, block swap 27 / 30, 200 epochs, a checkpoint every 25 epochs.",
            "Fine-tuning the 32B DiT in BF16 needs about 90 GB of system RAM for block swap.",
            "FP8 base and H2D-only block swap are LoRA-only; Fused Backward needs gradient accumulation 1 and cannot be combined with Patch Optimizer for Block Swap.",
            "Checkpoints hold the full DiT, not a LoRA; reload or run them with model version dev.",
        ],
        "vram": {
            "lora": "LoRA presets at 1024 px, rank 128: 19.8 GB peak with block swap 27 (24 GB), 29.9 GB with block swap 20 and H2D-only streaming (32 GB); 13.5 GB at 768 px, rank 32, with block swap 29 and activation CPU offload (16 GB); no preset fits 12 GB or less; max block swap 30.",
            "full": "Fine-tuning at 1024 px: 23.3 GB peak with block swap 27 (32 GB, about 139 s/step on an RTX A6000), 21.5 GB with block swap 30, the dev maximum (24 GB, about 173 s/step); no preset fits 16 GB or less.",
        },
        "dataset": "1024x1024 images (sides in multiples of 16) with same-name .txt captions; control_images only for reference-image training.",
    },
    "flux2_klein_9b": {
        "about": "Compact FLUX.2 Klein 9B with the Qwen3 8B text encoder, as a base model or a distilled 4-step model.",
        "all": [
            "Train on klein-base-9b; the distilled klein-9b is mainly meant for inference.",
            "Model Version sets the sample defaults: 50 steps at guidance 4.0 for the base model, 4 steps at guidance 1.0 for the distilled one.",
            "FP8 for the Qwen3 text encoder can be enabled for training and caching to save VRAM; the presets leave it off.",
            "An optional control_images sub-folder in a dataset folder adds reference images for edit-style training.",
            "Up to 18 blocks can be swapped (the Klein 9B maximum); fewer swapped blocks train faster when VRAM allows.",
        ],
        "lora": [
            "LoRA presets: rank 128, AdaFactor LR 6e-5, 1024 px, unquantized BF16 base, 200 epochs, a checkpoint every 25 epochs; 10 GB drops to rank 64, 8 GB to rank 32 at 768 px.",
            "No preset uses FP8 Base; if you enable it to save VRAM, enable Scaled FP8 with it, as the FLUX.2 docs recommend.",
        ],
        "full": [
            "Fine-tuning presets: full BF16 weights, fused-backward AdaFactor at LR 1e-5 with warmup, 1024 px, 200 epochs, a checkpoint every 25.",
            "FP8 base options and H2D-only block swap are not supported for full fine-tuning.",
            "Reload or run the fine-tune with the same model version (klein-base-9b or klein-9b); it is not detected automatically.",
        ],
        "vram": {
            "lora": "LoRA at 1024 px, rank 128: 22.4 GB without block swap (32 GB) down to 9.7 GB with block swap 17 (12 GB); 10 and 8 GB swap 18 at rank 64 / 32 (7.6 / 6.1 GB).",
            "full": "Fine-tuning at 1024 px: 24.4 GB without block swap (32 GB), 20.4 GB with block swap 4, 14.3 GB with block swap 12, 10.3 GB with block swap 17 (12 GB); no preset fits 10 GB or less.",
        },
        "dataset": "1024x1024 images (sides in multiples of 16) with same-name .txt captions; control_images only for reference-image training.",
    },
    "flux2_klein_4b": {
        "about": "Smallest FLUX.2 Klein (4B) with the Qwen3 4B text encoder, base or distilled 4-step, for fast training on 8-12 GB GPUs.",
        "all": [
            "Train on klein-base-4b; the distilled klein-4b is mainly meant for inference.",
            "Model Version sets the sample defaults: 50 steps at guidance 4.0 for the base model, 4 steps at guidance 1.0 for the distilled one.",
            "FP8 for the Qwen3 text encoder can be enabled for training and caching to save VRAM; the presets leave it off.",
            "An optional control_images sub-folder in a dataset folder adds reference images for edit-style training.",
            "Up to 13 blocks can be swapped (the Klein 4B maximum); fewer swapped blocks train faster when VRAM allows.",
        ],
        "lora": [
            "LoRA presets: rank 128, AdaFactor LR 6e-5, 1024 px, unquantized BF16 base, 200 epochs, a checkpoint every 25 epochs; the 6 GB preset trains rank 32 at 768 px.",
            "The 24 and 32 GB presets turn off gradient checkpointing for speed.",
            "No preset uses FP8 Base; if you enable it to save VRAM, enable Scaled FP8 with it, as the FLUX.2 docs recommend.",
        ],
        "full": [
            "Fine-tuning presets: full BF16 weights, fused-backward AdaFactor at LR 1e-5 with warmup, 1024 px, 200 epochs, a checkpoint every 25.",
            "FP8 base options and H2D-only block swap are not supported for full fine-tuning.",
            "Reload or run the fine-tune with the same model version (klein-base-4b or klein-4b); it is not detected automatically.",
        ],
        "vram": {
            "lora": "LoRA at 1024 px, rank 128: 11.1 GB without block swap (16 GB) down to 5.8 GB with block swap 11 (8 GB); 24 and 32 GB use 22.3 GB without gradient checkpointing; 6 GB: 4.2 GB at 768 px.",
            "full": "Fine-tuning at 1024 px: 12.1 GB without block swap (16-32 GB), 9.4 GB with block swap 6, 8.0 GB with block swap 8, 5.8 GB with block swap 13 and activation CPU offload (8 GB).",
        },
        "dataset": "1024x1024 images (sides in multiples of 16) with same-name .txt captions; control_images only for reference-image training.",
    },
    "qwen_image": {
        "about": "Qwen-Image text-to-image (original and 2512) plus Qwen-Image-Edit Plus 2509 and Edit 2511 for instruction-based editing.",
        "all": [
            "Model Version original covers Qwen-Image and 2512 (trained exactly the same way); edit-2509 and edit-2511 train the edit models.",
            "Use the BF16 checkpoints from the downloader; pre-quantized FP8 or GGUF files do not work, and FP8 Base converts BF16 while loading.",
            "Presets come in 200, 100 and 50 epoch sets; 200 epochs uses the lowest learning rate and gives the best quality.",
            "The researched recipe trains at 1328x1328 with sigmoid timesteps, mode weighting 1.0 and flow shift 2.2.",
            "Edit models also train without control images; control images (up to 3, or auto-generated black ones) teach new edits but need far more VRAM.",
            "🔍 Check dataset in 📁 Dataset shows every target beside its references and lists images without one or misnamed references.",
        ],
        "lora": [
            "LoRA presets: AdaFactor, rank 128, LR 6e-5 for 200 epochs, 1.5e-4 for 100 or 3e-4 for 50.",
            "Tier1 presets keep BF16 weights, Tier2 and up use FP8 Scaled, and Tier3-6 also lower rank or resolution.",
        ],
        "full": [
            "Fine-tuning presets: full BF16, fused-backward AdaFactor at LR 9e-6 for 200 epochs, 2.25e-5 for 100 or 4.5e-5 for 50.",
            "BF16 fine-tunes are about 40 GB; the Model Quantizer tab converts them to FP8 Scaled at half the size.",
            "The LoRA Extractor tab can pull a small LoRA out of the fine-tuned checkpoint.",
        ],
        "vram": {
            "lora": "LoRA presets from about 5 GB (Tier6, 768 px) to 51 GB; the MB in each preset name is its measured VRAM.",
            "full": "Fine-tuning presets from 5 GB (Tier2, 1024 px) to 84 GB; the MB in each preset name is its measured VRAM.",
        },
        "dataset": "1328x1328 images in folders like 1_ohwx with a trigger-word caption; Edit controls go in edit_images (image1.jpg -> image1_0.png, image1_1.png).",
    },
    "qwen_image_21": {
        "about": "Qwen-Image 2.1 with its own DiT, RGBA VAE and Qwen3-VL 8B text encoder, for text-to-image and editing LoRAs from ordered reference images.",
        "all": [
            "Qwen-Image 1.x models and caches cannot be reused; Qwen-Image 2.1 needs its own downloaded bundle.",
            "Image sides snap to multiples of 32; presets train at 1328x1328 with sigmoid timesteps, mode weighting 1.0 and flow shift 2.2.",
            "For editing, enable control images and add ordered references in control_images (image1.png -> image1_0.png, image1_1.png).",
            "🔍 Check dataset in 📁 Dataset shows every target beside its references and lists images without one or misnamed references.",
            "After changing control images, rebuild both latent and text caches with skip-existing turned off.",
            "Text-encoder caching on the CPU needs about 20 GB of free system RAM.",
        ],
        "lora": [
            "LoRA presets: rank 128, AdaFactor LR 6e-5, 1328 px, 200 epochs, a checkpoint every 25 epochs.",
            "8 GB and 6 GB tiers drop to rank 64 / 32 (6 GB at 768 px) with activation CPU offload.",
            "Tiers of 16 GB and below cache the Qwen3-VL text encoder on the CPU in full precision (slower, lossless).",
        ],
        "full": [
            "Fine-tuning presets: full BF16, fused-backward AdaFactor at LR 9e-6, 1328 px (1024 px on 8 GB, 512 px on 6 GB).",
            "Block swap during fine-tuning needs about 40 GB of system RAM.",
            "FP8 text-encoder caching during fine-tuning needs the text cache device set to cuda.",
        ],
        "vram": {
            "lora": "LoRA: 20 GB peak without block swap (24 GB tier) down to 4.2 GB at 768 px rank 32 (6 GB tier); max block swap 30 of 32.",
            "full": "Fine-tuning: 20.3 GB without block swap (24 GB tier), 8.1 GB at 1328 px with block swap 30 (10 GB), 4.2 GB at 512 px (6 GB); max block swap 30 of 32.",
        },
        "dataset": "Images at 1328x1328 (sides in multiples of 32) with same-name .txt captions; editing adds a control_images sub-folder.",
    },
    "zimage_base": {
        "about": "Z-Image Base, the quality-first 6B Z-Image model and the recommended base for LoRAs that should also work on Z-Image Turbo.",
        "all": [
            "The presets train 1024 px for 200 epochs with timestep sampling shift and flow shift 2.0.",
            "Up to 28 blocks can be swapped; fewer swapped blocks train faster when VRAM allows.",
            "FP8 for the Qwen3 text encoder can be enabled for training and caching to save VRAM; the presets leave it off.",
            "Training samples default to about 25 steps at CFG 4.0.",
        ],
        "lora": [
            "LoRA presets: rank 128, AdaFactor LR 6e-5, unquantized BF16 base, a checkpoint every 25 epochs; the 6 GB preset trains rank 32 at 768 px.",
            "No preset uses FP8 Base; as an optional VRAM saver enable it together with Scaled FP8 (the Z-Image docs note quality may drop slightly).",
        ],
        "full": [
            "Fine-tuning presets: full BF16 weights, fused-backward AdaFactor at LR 9e-6 with warmup, a checkpoint every 25 epochs; the 6 GB preset trains 512 px.",
            "Fused Backward only helps DreamBooth fine-tuning with AdaFactor; it has no effect on LoRA.",
            "FP8 base options are LoRA-only; fine-tuning trains unquantized weights.",
        ],
        "vram": {
            "lora": "LoRA at 1024 px, rank 128: 16.4 GB without block swap (24 and 32 GB) down to 6.4 GB with block swap 28 (8 GB); 6 GB: 4.1 GB at 768 px, rank 32.",
            "full": "Fine-tuning at 1024 px: 15.3 GB without block swap (24 and 32 GB) down to 6.4 GB with block swap 28 (8 GB); 6 GB: 4.1 GB at 512 px.",
        },
        "dataset": "1024x1024 images in folders like 1_name with same-name .txt captions.",
    },
    "zimage_turbo": {
        "about": "Z-Image Turbo, the distilled 8-step Z-Image model; LoRAs train on it through a training adapter.",
        "all": [
            "For LoRAs that should work on both Base and Turbo, train on Z-Image Base instead.",
            "Up to 28 blocks can be swapped; fewer swapped blocks train faster when VRAM allows.",
            "Training samples default to about 20 steps at CFG 1.0.",
        ],
        "lora": [
            "Keep the training adapter (zimage_turbo_training_adapter_v2 from the Turbo bundle) at Training Adapter Strength 1.0; it is applied as base weights.",
            "LoRA presets: rank 128, AdaFactor LR 1e-5 (Base uses 6e-5), unquantized BF16 base, 200 epochs, a checkpoint every 25; 1536 px on 32 to 12 GB, 1024 px with activation CPU offload on 10 and 8 GB, 512 px at rank 32 on 6 GB.",
        ],
        "full": [
            "No measured Z-Image Turbo fine-tuning preset yet: the defaults load AdamW8bit at LR 1e-4 with Fused Backward Pass and Full BF16 Training off, and empty model file fields.",
            "Set yourself, as the Z-Image Base fine-tuning presets do: Full BF16 Training and Fused Backward Pass on, Optimizer Type AdaFactor with arguments scale_parameter=False relative_step=False warmup_init=False weight_decay=0.01.",
            "Then Learning Rate 9e-6, scheduler constant_with_warmup with 10 warmup steps, Max Gradient Norm 0, and Blocks To Swap as VRAM requires.",
            "The training adapter is LoRA-only: fine-tuning leaves base weights out and trains the Turbo DiT directly.",
            "FP8 base options are LoRA-only; fine-tuning trains unquantized weights.",
        ],
        "vram": {
            "lora": "LoRA at 1536 px: 19.4 GB without block swap (24 and 32 GB), 14.3 GB with block swap 15 and 10.2 GB with block swap 27 (16 / 12 GB, H2D-only streaming); 1024 px: 8.1 GB (10 GB) and 6.4 GB with block swap 28 (8 GB).",
            "full": "No measured fine-tuning preset yet; up to 28 blocks can be swapped.",
        },
        "dataset": "Images in folders like 1_name with same-name .txt captions; most LoRA presets train 1536x1536, the smallest ones 1024 or 512 px.",
    },
    "ideogram4": {
        "about": "Ideogram 4 for typography, posters and design work, trained on structured JSON captions.",
        "all": [
            "Ideogram 4 was trained on structured JSON captions (keys such as high_level_description and style_description); plain text also works.",
            "Validate Structured Captions checks JSON captions during caching; Warn Instead of Failing turns errors into warnings.",
            "Inference prompts are always checked as structured JSON, so test the result with JSON prompts.",
            "Samples use a sampler preset (V4_DEFAULT_20, V4_QUALITY_48 or V4_TURBO_12) instead of manual steps and CFG.",
        ],
        "lora": [
            "LoRA trains on the official FP8 conditional DiT as a frozen base, so no FP8 flags are needed.",
            "LoRA presets: rank 128, AdaFactor LR 6e-5, 1024 px, 200 epochs, a checkpoint every 25 epochs; 12 GB and below lower the rank (and the resolution on 10 and 8 GB).",
            "Training samples use single-DiT CFG unless you set the unconditional DiT and enable Use Official Asymmetric CFG for LoRA Samples.",
            "A LoRA cannot be merged back into the FP8 base weights.",
        ],
        "full": [
            "Fine-tuning cannot use the official FP8 or NVFP4 release; select a plain FP32, FP16 or BF16 conditional DiT (not downloaded for you).",
            "The fine-tuning preset (not VRAM-measured): full BF16 weights, fused-backward AdaFactor at LR 1e-5 with warmup, block swap 32 (the fine-tuning maximum), a checkpoint every 25 epochs.",
            "FP8 options and H2D-only block swap are not supported for full fine-tuning.",
            "When an unconditional DiT is set, fine-tuning samples use it automatically for asymmetric CFG.",
        ],
        "vram": {
            "lora": "LoRA: 19.5 GB without block swap (24 and 32 GB), 13.2 GB with block swap 26 and H2D-only streaming (16 GB), down to 6.3 GB at 512 px, rank 32 (8 GB); up to 33 blocks swap with H2D-only streaming, 32 without.",
            "full": "The fine-tuning preset is not VRAM-measured; fine-tuning swaps at most 32 blocks.",
        },
        "dataset": "1024x1024 images, each with a same-name .txt caption, ideally holding structured JSON text.",
    },
    "hidream_o1": {
        "about": "HiDream-O1, a pixel-space Qwen3-VL based image model in one checkpoint (full or dev) for text-to-image and image-to-image.",
        "all": [
            "No VAE and no separate text encoder: caching only stores pixel patches and tokenized captions.",
            "Image sides must be multiples of 32 (32x32 pixel patches); uniform timesteps match the official SFT recipe.",
            "Model Type must match the checkpoint and fills in its settings: full (the presets) sets Noise Scale Start and End 8.0, Noise Clip Std 0.0 and CFG Scale 5; dev sets 7.5, 7.5, 2.5 and CFG 0.",
            "Switching to dev also selects the hidream_o1_image_dev_bf16 checkpoint: download HiDream-O1 Dev first, otherwise the checkpoint field is cleared.",
            "Task i2i turns on Use control/reference images (t2i turns it off); add a control_images sub-folder and rebuild the pixel and text caches; i2i also trains the visual encoder.",
            "🔍 Check dataset in 📁 Dataset shows every target beside its references and lists images without one or misnamed references.",
            "With block swap, the i2i task or FlashAttention requires a dataset batch size of 1.",
        ],
        "lora": [
            "LoRA presets: rank 128, AdaFactor LR 4e-5, 1024 px, 200 epochs, a checkpoint every 25 epochs.",
            "The 32 GB preset turns off gradient checkpointing for speed (about 0.56 s/step on an RTX 5090).",
            "For i2i LoRAs, add network args conv_dim=4 conv_alpha=1 if the visual conv layers should adapt too.",
        ],
        "full": [
            "Fine-tuning presets: full BF16, fused-backward AdaFactor at LR 5e-6, 1024 px.",
            "Freeze Text Token Embeddings keeps the 0.6B token table frozen (its update alone peaks at 15.5 GB); every transformer layer still trains.",
            "The 16 GB and smaller fine-tuning presets freeze the token table; the 24 and 32 GB presets train it too.",
        ],
        "vram": {
            "lora": "LoRA: 19.2 GB without block swap (24 GB tier) down to 4.3 GB at 768 px rank 32 (6 GB tier); max block swap 34 of 36 layers.",
            "full": "Fine-tuning at 1024 px: 27.9 GB without block swap (32 GB tier) down to 4.3 GB with block swap 34 and a frozen token table (6 GB tier); max block swap 34 of 36 layers.",
        },
        "dataset": "Images with sides in multiples of 32 (presets use 1024x1024) and .txt captions; i2i adds a control_images sub-folder.",
    },
    "ernie_image": {
        "about": "Baidu ERNIE-Image, an 8B text-to-image DiT with the Ministral 3 text encoder and the FLUX.2 VAE (LoRA training only).",
        "all": [
            "Flow shift 4.0 matches the official scheduler and is the recommended starting point (timestep sampling shift).",
            "Training samples default to 50 steps at CFG 4.0 with the negative prompt low quality, blurry.",
        ],
        "lora": [
            "Presets: rank 128, AdaFactor LR 6e-5, 1024 px, 200 epochs, a checkpoint every 25 epochs.",
            "8 GB and 6 GB tiers drop to rank 64 / 32 (6 GB at 768 px) and cache the text encoder on the CPU.",
        ],
        "vram": "24 GB trains without block swap (20.8 GB peak); 16, 12 and 10 GB tiers swap 18, 28 and 33 blocks (H2D-only); max swap 34.",
        "dataset": "1024x1024 images in folders like 1_name with same-name .txt captions (768 px on the 6 GB tier).",
    },
    "flux3_klein": {
        "about": "Black Forest Labs FLUX.3 Klein, a 5.4B text-to-image and editing DiT with the Qwen3-VL 4B text encoder and the FLUX.2 VAE (LoRA training only).",
        "all": [
            "Train on the undistilled Base checkpoint (flux3-image-klein-base-bf16); the LoRA also loads on the 16-step distilled model in ComfyUI and SwarmUI.",
            "Timestep sampling shift with flow shift 6.93 is BFL's own sampler shift, the same at every resolution.",
            "Training samples default to 50 steps at CFG 3.5 with an empty negative prompt, the base model's reference settings.",
            "Turn on reference images to train an editing LoRA: each target takes one or more ordered references from its control folder.",
            "Up to 26 blocks can be swapped; two joint blocks always stay on the GPU.",
        ],
        "lora": [
            "Presets: DoRA rank 128, AdaFactor LR 6e-5, 1024 px, 200 epochs, a checkpoint every 25 epochs; the 8 GB tier trains 768 px and the 6 GB tier 512 px with an FP8 scaled base, both at rank 32.",
            "The 32 GB preset turns off gradient checkpointing (1.33 s/step on an RTX 5090; 1.46 s/step with it on the 24 GB preset).",
            "10 GB and smaller tiers cache the text encoder on the CPU.",
            "The presets were measured with a text-to-image dataset: reference images (editing) add tokens and keep the reference stream on the GPU, so pick a lower tier for editing LoRAs.",
        ],
        "vram": "24 GB trains without block swap (14.5 GB peak); 16, 12 and 10 GB swap 2, 16 and 25 blocks (H2D-only); 8 and 6 GB swap 22 and 24 blocks with activation CPU offload (6.2 and 4.3 GB).",
        "dataset": "1024x1024 images in folders like 1_name with same-name .txt captions; for editing, a control_images sub-folder with the references.",
    },
    "moss_image": {
        "about": "OpenMOSS MOSS-Image, a 5B single-stream text-to-image DiT with the Qwen3.5-9B text encoder and the FLUX.2 VAE (LoRA training only).",
        "all": [
            "Train on the BF16 Base model (moss_image_base_bf16); the LoRA also loads on the SFT model and on the INT8 ConvRot files in ComfyUI and SwarmUI.",
            "Timestep sampling shift with flow shift 5 is the vendor's own sampler shift.",
            "Training samples default to 50 steps at CFG 3 with an empty negative prompt; their negative branch skips block 22, as the vendor's sampler does.",
            "The text encoder is the INT8 ConvRot Qwen3.5-9B file SwarmUI downloads, read in BF16 for caching.",
            "Up to 40 blocks can be swapped; two blocks always stay on the GPU.",
        ],
        "lora": [
            "No presets yet: the defaults load rank 128 with AdaFactor LR 6e-5, 1024 px, 200 epochs, a checkpoint every 25.",
        ],
        "vram": "No presets yet, so no measured VRAM; the defaults train without block swap.",
        "dataset": "1024x1024 images in folders like 1_name with same-name .txt captions.",
    },
    "mage_flow": {
        "about": "Microsoft Mage-Flow (4B, MIT license), a native-resolution text-to-image model (LoRA training only).",
        "all": [
            "The trainer uses uniform_shift timesteps with static shift 6, matching the released inference scheduler.",
            "Only SDPA and FlashAttention attention backends are supported.",
            "Plain FP8 Base is rejected (enable Scaled FP8 with it), and torch.compile fullgraph mode is not supported.",
            "Training samples default to 30 steps at CFG 5.0.",
        ],
        "lora": [
            "Presets: rank 128, AdaFactor LR 6e-5, 1024 px, 200 epochs, a checkpoint every 25 epochs.",
            "The 24 and 32 GB Fast presets turn off gradient checkpointing (about 0.5 s/step on an RTX 5090).",
            "10 GB and smaller tiers cache the text encoder on the CPU; the 6 GB tier uses rank 64 with activation CPU offload.",
        ],
        "vram": "16 GB trains without block swap (11.2 GB peak); at most 10 blocks swap, and even the 6 GB tier keeps 1024 px.",
        "dataset": "1024x1024 images in folders like 1_name with same-name .txt captions.",
    },
    "mage_flow_edit": {
        "about": "Mage-Flow-Edit, the instruction-editing Mage-Flow model, learns edits from 1-3 ordered reference images per target (LoRA training only).",
        "all": [
            "Every dataset folder needs a control_images sub-folder; the tab refuses to start without one.",
            "Name references after their target with an index from 0: photo1.png -> control_images/photo1_0.png, photo1_1.png, photo1_2.png.",
            "The target image is the edited result and the references are the source images, used in index order.",
            "🔍 Check dataset in 📁 Dataset shows every target beside its references and lists images without one or misnamed references.",
            "Reference images add tokens, so Edit needs more VRAM than text-to-image Mage-Flow at the same settings.",
            "It uses its own Edit DiT (mage_flow_edit_base_bf16) with uniform_shift timesteps at shift 6.",
        ],
        "lora": [
            "Presets: rank 128, AdaFactor LR 6e-5, 1024 px, 200 epochs; the 6 GB tier trains 768 px at rank 32.",
            "The 32 GB Fast preset turns off gradient checkpointing (28.6 GB peak, 1.12 s/step on an RTX 5090).",
        ],
        "vram": "16 and 24 GB train without block swap (12.6 GB peak); 12, 10 and 8 GB swap 7, 9 and 10 blocks (the maximum).",
        "dataset": "1024x1024 target images with same-name .txt captions plus 1-3 indexed references per target in control_images.",
    },
    # ------------------------------------------------------------------ video models
    "wan22": {
        "about": "Wan 2.2 A14B dual-model video generator: a high-noise DiT for early denoising and a low-noise DiT for detail (T2V, I2V and image LoRAs).",
        "all": [
            "Fill both DiT paths to train high- and low-noise LoRAs together; leaving the high-noise path empty trains one model only.",
            "DoRA with both DiT paths writes <name>.safetensors for the low-noise model and <name>_high_noise.safetensors for the high-noise model (shared LoRA matrices, one DoRA magnitude per expert): load each file on its own model; a LoRA file loads on both.",
            "Timestep Boundary 0 uses the model default switch point: 0.875 for T2V and 0.9 for I2V.",
            "Use standard block swap for dual-DiT training; H2D-only block swap is not meant for Wan 2.2's two models.",
            "Auto Normalize Target Frames clamps the frame count to your shortest video so no clip is skipped.",
        ],
        "lora": [
            "Researched presets (64+ trainings): rank 128, AdaFactor LR 6e-5, 960x960, sigma timesteps, flow shift 5, 200 epochs.",
            "Text-to-Image presets train only the low-noise T2V DiT (high-noise path empty) for image LoRAs.",
            "Image datasets work well with Wan 2.2 and the presets were measured on images; video clips need more VRAM.",
            "FP8_Scaled_Low_RAM presets use FP8 Scaled weights; the picker prefers the regular BF16 preset of a tier.",
        ],
        "vram": "Presets from 8 GB (6 GB for T2I) up to 192 GB; 24 GB: 21.5 GB in BF16 with block swap 31 or 22.2 GB with FP8 Scaled; max swap 38.",
        "dataset": "960x960 images, 16 fps videos with 4n+1 frames (81 = about 5 s), or both (a folder with images and videos trains both), each with a same-name .txt caption.",
    },
    "wan21": {
        "about": "Wan 2.1 14B / 1.3B single-DiT video models for text-to-video, image-to-video, first-last-frame, text-to-image and Fun-Control.",
        "all": [
            "Pick the Task that matches your checkpoint; 1.3B tasks swap at most 28 blocks, 14B tasks 38.",
            "CLIP Vision is needed only for i2v-14B, flf2v-14B and i2v-14B-FC; every task uses the Wan 2.1 VAE.",
            "Fun-Control tasks need each training video paired with a matching control video.",
            "Auto Normalize Target Frames clamps the frame count to your shortest video so no clip is skipped.",
            "Wan trains LoRAs only; the installed backend has no full-DiT Wan script.",
        ],
        "lora": [
            "T2V 14B presets: rank 128, AdaFactor LR 6e-5, 960x960, sigma timesteps, flow shift 5, 200 epochs, a checkpoint every 10.",
            "The presets were measured on image datasets; video clips need more VRAM than the listed peaks.",
            "10 GB and 8 GB tiers drop to rank 64 / 32 with activation CPU offload (8 GB also trains at 768 px).",
        ],
        "vram": "T2V 14B: 29.2 GB with block swap 9 on 32 GB down to 6.5 GB (768 px, rank 32, block swap 36) on 8 GB.",
        "dataset": "Images, 16 fps videos or both at 960x960 (or 1280x720 / 720x1280), 4n+1 frames (81 = about 5 s), each with a same-name .txt caption; image-only folders work too.",
    },
    "ltx23": {
        "about": "Lightricks LTX 2.3 (22B) video model whose single checkpoint also holds the video VAE, audio VAE and vocoder, for video or joint audio+video.",
        "all": [
            "Audio + Video learns sound from each clip's audio track or a same-stem .wav file; image samples give no audio signal.",
            "Latent caching, text caching and training must use the same modality (Video only or Audio + Video).",
            "With the single-file FP8 Gemma 3 12B text encoder, keep the text cache device on cuda (CPU caching is not supported).",
            "First-frame conditioning probability 0.1 (default) keeps frame 0 clean on some steps for image-to-video use; 0 trains pure text-to-video.",
        ],
        "lora": [
            "LoRA presets: rank 128, AdaFactor LR 1e-4, 768x512, 33 frames, INT8 ConvRot base, 200 epochs, a checkpoint every 25; no block swap on 24 and 32 GB.",
            "Choose one base quantization: INT8 ConvRot (the presets' choice), FP8 Scaled, or NF4 (lowest VRAM and lowest fidelity).",
            "Video only builds a video-only transformer, so its LoRAs leave audio untouched; in Audio + Video, video_* target presets do the same.",
            "LoRAs are saved in musubi and ComfyUI (.comfy.safetensors) formats by default.",
        ],
        "full": [
            "Fine-tuning presets: dense BF16 weights, fused-backward Adafactor at LR 1e-5 with max grad norm 0, 768x512, 33 frames, 200 epochs, a checkpoint every 25; they need 64 GB+ system RAM.",
            "Turn off every base quantization and use the BF16 ltx-2.3-22b-dev checkpoint; mixed precision must be bf16.",
            "The trainable block-swap ring is the fastest block swap for fine-tuning; H2D-only block swap is LoRA-only.",
            "By default the final checkpoint is merged after training into a single-file LTX bundle (about 43 GB on disk, as Windows shows it).",
        ],
        "vram": {
            "lora": "LoRA at 768x512, 33 frames, rank 128: 18.3 GB peak without block swap (24 and 32 GB), 14.2 GB with block swap 18 (16 GB), 10.3 GB with block swap 34 (12 GB), 8.3 GB with block swap 42 (10 GB), 5.9 GB at rank 64 with block swap 42, activation CPU offload and Low-VRAM Text Caching (8 GB); up to 47 of the 48 blocks can be swapped.",
            "full": "Fine-tuning at 768x512, 33 frames: 30.4 GB with block swap 1 (32 GB), 21.9 GB with block swap 18 (24 GB), 13.9 GB with block swap 34 (16 GB), 9.9 GB with block swap 42 (12 GB), 7.9 GB with block swap 46 (10 GB); up to 47 of the 48 blocks can be swapped.",
        },
        "dataset": "Images (1-frame samples), videos or both at 768x512 in 32 px steps, 8k+1 frames (rounded down; presets: 33) resampled to 25 fps, .txt captions; Audio + Video uses each folder's videos and needs their sound described.",
    },
    "minimax_h3": {
        "about": "MiniMax H3, a CFG-distilled joint video+audio model, with LoRAs for text-to-video-with-audio (T2VA), first/last-frame (FL2VA) and reference (Ref2VA).",
        "all": [
            "Frame counts must be 17n+5 (other counts are rounded down); the released range is 124-345 frames (5-15 s at 24 fps), shorter needs Allow Experimental Duration.",
            "Dataset batch size is fixed to 1 by the architecture; raise gradient accumulation instead.",
            "FP8 is not supported; pre-quantized ConvRot INT8 checkpoints are detected automatically, so leave the ConvRot INT8 box off for them.",
            "T2VA and FL2VA use the FL2VA transformer; Ref2VA needs the Ref2VA transformer and a JSONL dataset via Use TOML File.",
        ],
        "lora": [
            "Keep the guidance loss on (scale 4.0): it stops the CFG-distilled base from washing out, at about 50% more step time.",
            "Presets: rank 128, AdamW8bit LR 1e-4, max grad norm 1.0, 124 frames, 200 epochs, a checkpoint every 25 epochs; 12 and 10 GB drop to rank 64 with activation CPU offload.",
            "Clips with an audio track or a same-stem .wav also train audio; a video-only LoRA can degrade H3's audio quality.",
            "Teacher matching (T2VA only) replaces the guidance loss; the two cannot be enabled together.",
        ],
        "vram": "At 124 frames: 512 px on 32 GB (30 GB peak), 384 px on 24 GB (21.7 GB), 256 px on 16 GB (14.4 GB) with H2D-only block swap; 256 px at rank 64 on 12 and 10 GB (9.5 and 8.2 GB, block swap 45 and 48); up to 48 of 50 blocks swap; 64 GB system RAM recommended.",
        "dataset": "Videos only (image folders are skipped), 17n+5 frames (default 124), sides in 32 px steps, normalized to 24 fps, one .txt caption per clip.",
    },
    "kandinsky6": {
        "about": "Kandinsky 6 Lite generates video together with audio at 24 fps from text (t2av) or text plus a first frame (ti2av).",
        "all": [
            "Frame counts must be 4n+1 (81, 97, 121; other counts are rounded down); 121 frames is about 5 seconds at 24 fps.",
            "Batch size is fixed to 1; raise gradient accumulation instead.",
            "Clips with sound also train audio (video track or same-stem audio file); image folders train appearance as silent one-frame videos.",
            "ti2av needs a JSONL dataset with a first-frame control_path per clip, loaded with Use TOML File.",
            "Train the regular Lite or Pro transformer; distilled checkpoints cannot be trained, and attention is SDPA only.",
        ],
        "lora": [
            "Presets: rank 32, AdamW8bit LR 1e-4, max grad norm 1.0, 100 epochs, a checkpoint every 10 epochs.",
            "Video Only turns the audio loss off; Audio Loss Weight scales it for clips with real audio.",
            "ConvRot INT8 base (needs Triton) and FP8 Base are mutually exclusive; choose one base quantization.",
        ],
        "vram": "24/32 GB: 864x480 at 121 frames (22.4 GB peak); 16 GB: 864x480 with block swap 17 (14.3 GB); 12 and 10 GB: 640x352 at 121 frames; 8 GB: 512x288 at 97 frames; 6 GB: 512x288 at 49 frames (4.3 GB); Lite swaps at most 30 blocks.",
        "dataset": "Clips with sound at 864x480 (16 px steps), 4n+1 frames (121 = about 5 s at 24 fps), images, or both in one folder, each with a same-name .txt caption.",
    },
    # ------------------------------------------------------------------ audio models
    "ltx23_audio": {
        "about": "Audio-only LoRA or fine-tune of the LTX 2.3 audio branch for speech, music and sound effects, with no video target.",
        "all": [
            "Audio clips are used as they are, and videos with a sound track count too: Start extracts their audio to WAV (cache folder, extracted_audio) with the videos' captions; silent videos are skipped.",
            "Latent caching, text caching and training all run in audio mode, and training samples are audio-only previews.",
            "Caption each clip with a same-stem .txt that describes the sound: speech, music or ambient noise.",
            "The LTX 2.3 checkpoint supplies the audio VAE and vocoder; the Gemma 3 12B text encoder is still needed for text caching.",
        ],
        "lora": [
            "No presets yet: the defaults load rank 128 with AdaFactor LR 1e-4 (the pair the LTX 2.3 video LoRA presets use), INT8 ConvRot base, block swap 20, 100 epochs, a checkpoint every 10.",
            "Keep the LoRA target preset on audio: it trains only the audio attention and FFN layers.",
        ],
        "full": [
            "No presets yet: the defaults follow the LTX 2.3 fine-tuning recipe (dense BF16 weights, fused-backward Adafactor at LR 1e-5, max grad norm 0) with block swap 20, 100 epochs, a checkpoint every 10; it needs 64 GB+ system RAM.",
            "Keep every base quantization off (the defaults do) and use the BF16 ltx-2.3-22b-dev checkpoint; mixed precision must be bf16.",
            "By default the final checkpoint is merged after training into a single-file LTX bundle (about 43 GB on disk, as Windows shows it).",
        ],
        "vram": {
            "lora": "No presets yet, so no measured VRAM; the defaults swap 20 of the 48 blocks (up to 47 can be swapped, and LoRA can stream them H2D-only).",
            "full": "No presets yet, so no measured VRAM; the defaults swap 20 of the 48 blocks on the trainable ring (up to 47 can be swapped).",
        },
        "dataset": "Sub-folders such as 1_my_voice with audio clips (.wav, .flac, .mp3, .ogg, .m4a, .aac, .opus, .wma) and/or videos with sound, each with a same-stem .txt caption describing the sound.",
    },
}
