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
            "LoRA presets: rank 128, AdaFactor LR 2e-4, 1024 px, 200 epochs, a checkpoint every 25 epochs; 12 GB and below lower the rank (and the resolution on 10 and 8 GB).",
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
    "krea2_turbo": {
        "about": (
            "Krea 2 Turbo, the distilled 8-step Krea 2: LoRAs train on it through ostris' training adapter, as ai-toolkit trains "
            "Krea 2 Turbo, and the LoRA runs on Turbo at 8 steps without CFG in ComfyUI."
        ),
        "all": [
            "The training adapter (krea2_turbo_training_adapter_v1 from the Krea 2 Turbo bundle) de-distills Turbo while the LoRA trains: it is merged into the DiT for training only, left out of the training samples and not part of the saved LoRA, so the samples show Turbo with your LoRA.",
            "Training samples use Turbo's own sampler: a fixed shift (mu 1.15), 8 steps, CFG 1, as Krea's reference code and ComfyUI's Krea-2 Turbo templates run it.",
            "Keep runs short (styles, concepts, characters): the adapter's author notes that a long run still wears the distillation down, and artifacts can then appear without the adapter.",
            "DoRA learns its magnitudes on the merged weights: the adapter moves their column norms by 0.003% on average (at most 1.5% in a few columns of the last block), so the LoRA loads on plain Turbo as trained.",
            "Up to 26 blocks can be swapped; lowering block swap is where most of the speed comes from when VRAM allows.",
        ],
        "lora": [
            "LoRA presets: DoRA rank 128 on the ConvRot INT8 DiT, AdaFactor LR 1e-4, 1024 px, 200 epochs, a checkpoint every 25 epochs; 12 GB lowers the rank to 64 with activation CPU offload, 10 and 8 GB train 768 and 512 px at rank 32.",
            "ConvRot INT8 (every preset) is faster than FP8 and closer to BF16; FP8 Base and Scaled FP8 must be enabled together, and only one base quantization runs at a time. The adapter is merged before either quantization.",
            "About 6.1 s/step on an RTX A6000 without block swap (32 GB) and with 4 blocks H2D-only (24 GB), 12.05 with 24 (16 GB); activation CPU offload on 8-12 GB takes 20-25 s/step.",
            "10 GB and smaller tiers cache the text encoder on the CPU.",
        ],
        "vram": "LoRA at 1024 px, rank 128: 23.4 GB without block swap (32 GB), 22.3 GB with block swap 4 and 14.1 GB with block swap 24, both H2D-only (24 / 16 GB); 10.1 GB at rank 64 with activation CPU offload (12 GB); 8.1 and 6.1 GB at 768 and 512 px, rank 32 (10 / 8 GB); no preset fits 6 GB.",
        "dataset": "1024x1024 images in Kohya folders such as 1_ohwx, each with a same-name .txt caption.",
    },
    "krea2_edit": {
        "about": (
            "Krea 2 editing LoRAs: the model learns an edit from each target image (the result) and its reference images (the source), "
            "trained on Raw; the LoRA also runs on Turbo. The references reach the model the way ComfyUI's Krea 2 edit workflow feeds "
            "them."
        ),
        "all": [
            "Every dataset folder needs a control_images sub-folder with each target's reference: photo.png -> control_images/photo.png or photo_0.png (more references: photo_1.png ...); the tab refuses to start without it.",
            "As in ComfyUI's Image Style Reference template: the text encoder reads each reference (about 384 x 384 pixels) after a Picture N marker with the Text Encode Qwen Image Edit Plus template, and the DiT takes its latents at timestep 0 (Edit Model Reference Method index_timestep_zero).",
            "Each reference keeps its own framing at the training pixel count, so a batch holds one target: raise Gradient Accumulation Steps instead of the batch size.",
            "Sample prompts take their reference images with --ci (one per reference, in order); a sample's references are sized to its pixel count.",
            "In ComfyUI: Text Encode Qwen Image Edit Plus with the references and the VAE, Edit Model Reference Method index_timestep_zero, then the Krea 2 Raw or Turbo model with the LoRA.",
            "🔍 Check dataset in 📁 Dataset shows every target beside its references and lists images without one.",
            "Up to 26 blocks can be swapped; lowering block swap is where most of the speed comes from when VRAM allows.",
        ],
        "lora": [
            "Presets: one reference per target, DoRA rank 128 on the ConvRot INT8 DiT, AdaFactor LR 1e-4, 1024 px, 200 epochs, a checkpoint every 25 epochs; 16 GB adds activation CPU offload, 12 and 10 GB train 768 and 512 px at rank 32. No configuration fits 8 GB or less.",
            "Each reference adds its latent tokens to the image and its vision tokens to the text, so editing needs more VRAM than text-to-image; more references per target need more than the listed peak: pick a lower tier.",
            "11.53 s/step without block swap on an RTX A6000 (32 GB), 15.42 with 21 blocks H2D-only (24 GB); activation CPU offload makes the smaller tiers much slower (16 GB 42.26 s/step).",
        ],
        "vram": "One reference per target at 1024 px, rank 128: 30.2 GB without block swap (32 GB), 22.1 GB with block swap 21, H2D-only (24 GB), 14.2 GB with block swap 26 and activation CPU offload (16 GB); 10.1 GB at 768 px, rank 32 (12 GB); 10 GB trains 512 px; nothing fits 8 GB or less.",
        "dataset": "1024x1024 target images with same-name .txt captions (the edit instruction) plus one reference per target in control_images (photo_0.png).",
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
            "LoRA presets: rank 128, AdaFactor LR 3e-5, 1024 px, FP8 Base with Scaled FP8, 200 epochs, a checkpoint every 25 epochs; 16 GB trains 768 px at rank 32 with activation CPU offload.",
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
            "full": "Fine-tuning at 1024 px: 23.3 GB peak with block swap 27 (32 GB), 21.5 GB with block swap 30, the dev maximum (24 GB), both about 90 s/step on an RTX A6000; no preset fits 16 GB or less.",
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
            "LoRA presets: AdaFactor, rank 128, LR 1.2e-4 for 200 epochs, 3e-4 for 100 or 50. As DoRA, the earlier 6e-5 (200) and 1.5e-4 (100) learned only a partial likeness by the last save; at the doubled rates the subject arrived at epoch 125-175 of 200 and 80-100 of 100 on two training seeds.",
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
            "Each checkpoint also gets <name>.comfy.safetensors: load that file in ComfyUI or SwarmUI, which apply the trainer's own file without 2 of the 7 trained layers per block (the fused MLP); convert older LoRAs in Tools with Qwen-Image 2.1 -> ComfyUI / SwarmUI.",
            "In ComfyUI load the LoRA with Load LoRA or LoRA Loader Model Only: Load LoRA (Bypass) drops the adapters on the fused MLP halves (ComfyUI pull request 17247, open) and ignores DoRA.",
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
    "qwen_image_21_turbo": {
        "about": (
            "Qwen-Image 2.1 Turbo, the distilled 8-step Qwen-Image 2.1: LoRAs train on it through ostris' training adapter, as "
            "ai-toolkit trains it, and run on Turbo at 8 steps with CFG 1 in ComfyUI."
        ),
        "all": [
            "The training adapter (qwen_image_2_turbo_training_adapter_v1 from the Turbo bundle) de-distills Turbo while the LoRA trains: it is merged into the DiT for training only, left out of the training samples and not part of the saved LoRA, so the samples show Turbo with your LoRA.",
            "Training samples use the 8-step schedule the Turbo checkpoint ships, at CFG 1, as ComfyUI's Turbo templates and Diffusers run it; the step count does not change it.",
            "Editing works as on Qwen-Image 2.1: turn on control images and add ordered references in control_images (image1.png -> image1_0.png).",
            "DoRA learns its magnitudes on the merged weights: the adapter moves their column norms by 0.004% on average (at most 1% in a few columns of the last block), so the LoRA loads on plain Turbo as trained.",
            "Keep runs short (styles, concepts, characters): the adapter's author notes that a long run still wears the distillation down, and artifacts can then appear without the adapter.",
            "Up to 30 blocks can be swapped; image sides snap to multiples of 32.",
        ],
        "lora": [
            "LoRA presets: DoRA rank 128, AdaFactor LR 6e-5, 1328 px, 200 epochs, a checkpoint every 25 epochs; 8 GB lowers the rank to 64 with activation CPU offload, 6 GB trains 768 px at rank 32.",
            "Each checkpoint also gets <name>.comfy.safetensors for ComfyUI and SwarmUI, on qwen_image_2.1_turbo_bf16 or its INT8 ConvRot file.",
            "Tiers of 16 GB and below cache the Qwen3-VL text encoder on the CPU in full precision (slower, lossless); on 24 and 32 GB text caching on the GPU peaks at 16.2 GB.",
            "6.53-6.58 s/step without block swap on an RTX A6000 (24 and 32 GB), 7.13 and 8.70 with 17 and 27 blocks H2D-only (16 / 12 GB); 10 GB and below take 19-28 s/step.",
            "Measured with a text-to-image dataset: editing (control images) adds every reference's tokens to each step, so pick a lower tier for editing LoRAs.",
        ],
        "vram": "LoRA at 1328 px, rank 128: 20.3 GB without block swap (24 and 32 GB), 14.2 GB with block swap 17 and 10.2 GB with block swap 27, both H2D-only (16 / 12 GB), 8.1 GB with block swap 30 (10 GB); 6.3 GB at rank 64 with activation CPU offload (8 GB); 4.2 GB at 768 px, rank 32 (6 GB).",
        "dataset": "1328x1328 images in folders like 1_name with same-name .txt captions; for editing, a control_images sub-folder with each target's references.",
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
            "LoRA presets: rank 128, AdaFactor LR 3e-5, unquantized BF16 base, a checkpoint every 25 epochs; the 6 GB preset trains rank 32 at 768 px.",
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
            "LoRA presets: rank 128, AdaFactor LR 5e-5 (Base uses 3e-5), unquantized BF16 base, 200 epochs, a checkpoint every 25; 1536 px on 32 to 12 GB, 1024 px with activation CPU offload on 10 and 8 GB, 512 px at rank 32 on 6 GB.",
        ],
        "full": [
            "No measured Z-Image Turbo fine-tuning preset yet: the defaults load AdamW8bit at LR 1e-4 with Fused Backward Pass and Full BF16 Training off; the DiT, VAE and text encoder come from Training_Models_Z_Image once the Turbo bundle is downloaded.",
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
            "LoRA presets: rank 128, AdaFactor LR 1e-4, 1024 px, 200 epochs, a checkpoint every 25 epochs; 12 GB and below lower the rank (and the resolution on 10 and 8 GB).",
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
            "LoRA presets: rank 128, AdaFactor LR 1e-4, 1024 px, 200 epochs, a checkpoint every 25 epochs.",
            "The 32 GB preset turns off gradient checkpointing for speed (about 0.43 s/step on an RTX 5090).",
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
            "Presets: rank 128, AdaFactor LR 9e-5, 1024 px, 200 epochs, a checkpoint every 25 epochs.",
            "8 GB and 6 GB tiers drop to rank 64 / 32 (6 GB at 768 px) and cache the text encoder on the CPU.",
        ],
        "vram": "24 GB trains without block swap (20.8 GB peak); 16, 12 and 10 GB tiers swap 18, 28 and 33 blocks (H2D-only); max swap 34.",
        "dataset": "1024x1024 images in folders like 1_name with same-name .txt captions (768 px on the 6 GB tier).",
    },
    "flux3_klein": {
        "about": "Black Forest Labs FLUX.3 Klein, a 5.4B text-to-image DiT with the Qwen3-VL 4B text encoder and the FLUX.2 VAE (LoRA training only); editing LoRAs train in the FLUX.3 Klein Edit card.",
        "all": [
            "Train on the undistilled Base checkpoint (flux3-image-klein-base-bf16); the LoRA also loads on the 16-step distilled model in ComfyUI (v0.41.0 or newer) and SwarmUI.",
            "In ComfyUI load the LoRA with Load LoRA or LoRA Loader Model Only: Load LoRA (Bypass) drops the adapters on the fused q / k / v (ComfyUI pull request 17247, open) and ignores DoRA.",
            "Timestep sampling shift with flow shift 6.93 is BFL's own sampler shift, the same at every resolution.",
            "Training samples default to 50 steps at CFG 3.5 with an empty negative prompt, the base model's reference settings.",
            "Up to 26 blocks can be swapped; two joint blocks always stay on the GPU.",
        ],
        "lora": [
            "Presets: DoRA rank 128, AdaFactor LR 1.2e-4, 1024 px, 200 epochs, a checkpoint every 25 epochs; the 8 GB tier trains 768 px and the 6 GB tier 512 px with an FP8 scaled base, both at rank 32.",
            "The 32 GB preset turns off gradient checkpointing (1.59 s/step on an RTX 5090; 1.78 s/step with it on the 24 GB preset).",
            "10 GB and smaller tiers cache the text encoder on the CPU.",
            "These presets are for text-to-image datasets: reference images (editing) add their tokens and keep the reference stream on the GPU, so train editing LoRAs in the FLUX.3 Klein Edit card, whose presets were measured with one reference per image.",
        ],
        "vram": "24 GB trains without block swap (14.5 GB peak); 16, 12 and 10 GB swap 2, 16 and 25 blocks (H2D-only); 8 and 6 GB swap 22 and 24 blocks with activation CPU offload (6.2 and 4.3 GB).",
        "dataset": "1024x1024 images in folders like 1_name with same-name .txt captions.",
    },
    "flux3_klein_edit": {
        "about": "FLUX.3 Klein editing LoRAs: the model learns an edit from each target image (the result) and its reference images (the source), LoRA training only.",
        "all": [
            "Every dataset folder needs a control_images sub-folder with each target's reference: photo.png -> control_images/photo_0.png (more references: photo_1.png ...); the tab refuses to start without it.",
            "Each reference keeps its own framing at the training resolution's pixel count (not cropped to the target), so the dataset batch size stays 1.",
            "🔍 Check dataset in 📁 Dataset shows every target beside its references and lists images without one.",
            "Train on the undistilled Base checkpoint (flux3-image-klein-base-bf16); the LoRA also loads on the 16-step distilled model in ComfyUI (v0.41.0 or newer) and SwarmUI.",
            "In ComfyUI load the LoRA with Load LoRA or LoRA Loader Model Only: Load LoRA (Bypass) drops the adapters on the fused q / k / v (ComfyUI pull request 17247, open) and ignores DoRA.",
            "Sample prompts take their reference images with --ci (one per reference, in order); timestep sampling shift 6.93 is BFL's own.",
            "Up to 26 blocks can be swapped; two joint blocks always stay on the GPU.",
        ],
        "lora": [
            "Presets: one reference per image, DoRA rank 128, AdaFactor LR 1.2e-4, 1024 px, 200 epochs, a checkpoint every 25 epochs; the 10 GB tier trains 768 px and the 8 GB tier 512 px with an FP8 scaled base, both at rank 32. No configuration fits 6 GB.",
            "More references per target add their tokens and need more VRAM than the listed peak: pick a lower tier.",
            "5.96 s/step without block swap on an RTX A6000 (32 and 24 GB presets), 6.59 s/step with 13 blocks streamed H2D-only (16 GB).",
        ],
        "vram": "24 and 32 GB train without block swap (16.9 GB peak); 16 GB streams 13 blocks H2D-only (14.3 GB); 12 GB swaps 24 blocks with activation CPU offload (10.3 GB); 10 GB trains 768 px (8.3 GB) and 8 GB 512 px with an FP8 base (6.3 GB); nothing fits 6 GB.",
        "dataset": "1024x1024 target images with same-name .txt captions plus one reference per target in control_images (photo_0.png).",
    },
    "moss_image": {
        "about": "OpenMOSS MOSS-Image, a 5B single-stream text-to-image DiT with the Qwen3.5-9B text encoder and the FLUX.2 VAE (LoRA or full fine-tuning).",
        "all": [
            "Train on the BF16 Base model (moss_image_base_bf16); the LoRA also loads on the SFT model and on the INT8 ConvRot files in ComfyUI (v0.42.0 or newer) and SwarmUI.",
            "In ComfyUI load the LoRA with Load LoRA or LoRA Loader Model Only: Load LoRA (Bypass) drops the adapters on the fused q / k / v (ComfyUI pull request 17247, open) and ignores DoRA.",
            "Training uses timestep sampling shift with flow shift 3 (about MOSS's pre-training distribution): it kept the prompts one checkpoint longer than shift 5 on two training seeds; samples keep the vendor's shift 5.",
            "Training samples default to 50 steps at CFG 3 with an empty negative prompt; their negative branch skips block 22, as the vendor's sampler does.",
            "The text encoder is the INT8 ConvRot Qwen3.5-9B file SwarmUI downloads, read in BF16 for caching. The unquantized BF16 file (Model Files: MOSS-Image BF16 text encoder, 19 GB on disk) loaded in 5 s instead of 49 s on an A6000 server, and its caption embeddings differ from the INT8 file's by about 3% (cosine 0.999): cache with the file you generate with.",
            "Up to 40 blocks can be swapped; two blocks always stay on the GPU.",
            "A caption file can hold several captions, separated by lines starting with ---===--- (Caption Separator Line): text caching encodes them all and each step trains one of them.",
            "Caption Dropout Rate (Advanced) trains that share of steps with the empty caption; while it is 0 a control adapter keeps its own Caption Dropout.",
            "Timestep sampling truncated_logit_normal is the MOSS-Image pre-training schedule: with Logit Mean -1.09, Logit Std 0.95 (the defaults here), Discrete Flow Shift 1, Timestep Cutoff 0.95 and Uniform Share 0.01 it reproduces it.",
            "Training samples draw their noise on the CPU, as ComfyUI does: a sample's seed gives the same starting noise as ComfyUI's KSampler with that seed.",
        ],
        "lora": [
            "Presets: DoRA rank 128, AdaFactor LR 6e-5, 1024 px, 200 epochs, a checkpoint every 25 epochs; the 6 GB tier trains 768 px at rank 32 with activation CPU offload.",
            "The 24 and 32 GB presets cache the text encoder on the GPU (15.4 GB caching peak); 16 GB and smaller cache it on the CPU.",
            "The trainer measures the CPU's BF16 speed first: where it is far slower than FP32 (an AVX2-only server CPU took 17 minutes per caption), text caching and sample prompts run the text encoder on the GPU one layer at a time instead (under 1 GB of VRAM, about 12 seconds per caption there).",
            "2.02 s/step on an RTX 5090 without block swap; 2.22, 2.32 and 2.46 s/step with 20, 30 and 39 swapped blocks.",
            "The _INT8 presets keep the DiT's block projections in ConvRot INT8, half the BF16 block size, so the tiers that swap blocks train faster: back to back on one RTX A6000, 12 GB took 3.64-3.73 s/step against 4.43 for the BF16 preset and 8 GB 4.16 against 6.37. The INT8 DiT's predictions are 1.5-4.1% from BF16's (FP8 Scaled 2.9-7.3%), and a corgi LoRA trained both ways from the same seeds came out alike save for save, with the same best epoch. On 8-12 GB the INT8 preset comes first (Quality 1 INT8, pre-selected) and the BF16 one follows as Quality 2; on 6 GB BF16 stays first (Quality 3) and INT8 follows as Quality 4 INT8 with the same 768 px settings. ConvRot INT8 Base (Model) turns INT8 on for any preset.",
            "Use Control Images trains a colour / depth control adapter for ComfyUI's MOSS-Image Color/Depth Control node (ComfyUI v0.42.2 or newer) instead of a subject LoRA: every image needs <image>_0.png (one flat colour per region) and <image>_1.png (depth, near = white) in its control_images sub-folder, and Make Missing Colour / Depth Maps creates them before caching (Mask2Former regions, Depth Anything V2 depth). An adapter learns layouts, not a subject: give it a large, varied image set.",
            "The control adapter follows the published recipe: the patch embedder's new colour / depth columns train at 10x the learning rate, each step keeps the colour map only, the depth map only, both or neither (35 / 35 / 20 / 10 %), 10% of the captions are dropped, and the text refiner's two blocks train with the LoRA. Every checkpoint is the file ComfyUI's node loads (load it with LoRA Loader Model Only).",
            "Control adapter sample prompts take the maps with --ci: the colour map, then the depth map; --ci none leaves one out (--ci none --ci depth.png samples with depth only).",
            "LoRA on text_embedder.proj (Advanced) also trains the projection of the text encoder's states into the DiT; ComfyUI loads it with the rest of the LoRA.",
            "Replica GPUs (Advanced) trains the LoRA on several GPUs from one process, without DDP, also on Windows: each GPU holds its own BF16 DiT and each step trains one batch per GPU, so with 2 GPUs an epoch has half the steps, the same steps and learning-rate schedule as a single-GPU run with Gradient Accumulation Steps x 2. It saves time only where the GPUs, not the CPU, set the pace: on two RTX A6000 of a busy Linux server a MOSS-Image DoRA step took 5.2 s against 5.6 s on one GPU, and 3.8 s on both with torch.compile, so compare the s/step of a short run first.",
        ],
        "full": [
            "Presets: every DiT weight trains in BF16 with fused-backward AdaFactor, LR 1e-5 (max grad norm 0), 1024 px, 200 epochs, a checkpoint every 25 epochs with only the newest training state kept; the VAE and the text encoder stay frozen.",
            "Triton Adafactor is on (the presets and the default): the same update in fused kernels, 1.4 GB less VRAM and 2.67 instead of 3.5 s/step than the standard update on an RTX A6000. Where Triton cannot build its kernels, the trainer warns and uses the standard update.",
            "With block swap, the Trainable Block-Swap Ring streams the swapped blocks through reusable GPU buffers from pinned CPU copies, written back after each update (it turns Pinned Memory for Block Swap on; about 10 GB of system RAM).",
            "The fine-tuned DiT is saved in the released layout (fused qkv), so it loads as the DiT for more training, in ComfyUI and in SwarmUI.",
            "2.64 s/step on an RTX A6000 without block swap (16 to 32 GB); 4.85, 6.49 and 7.93 s/step with 14, 24 and 33 swapped blocks (12, 10 and 8 GB), 11.10 s/step for the 6 GB preset with activation CPU offload.",
        ],
        "vram": {
            "lora": "24 and 32 GB train without block swap (14.0 GB); 16 GB too, with text caching on the CPU (14.1 GB). On 12, 10 and 8 GB the INT8 presets come first: 12 GB without block swap (9.9 GB), 10 and 8 GB with 17 and 36 blocks H2D-only (8.4 and 6.3 GB); the BF16 presets there swap 20, 30 and 39 blocks (H2D-only, 10.3, 8.2 and 6.3 GB). 6 GB swaps 35 blocks at 768 px (4.3 GB), its INT8 preset 29 blocks (4.3 GB).",
            "full": "16, 24 and 32 GB train without block swap (13.6 GB; 15.7 GB while the text encoder caches on the GPU on 24 and 32 GB); 12, 10 and 8 GB swap 14, 24 and 33 blocks through the ring with text caching on the CPU (10.4, 8.3 and 6.4 GB); 6 GB swaps 37 with activation CPU offload (4.2 GB).",
        },
        "dataset": "1024x1024 images in folders like 1_name with same-name .txt captions; for a control adapter, a control_images sub-folder with <image>_0.png colour and <image>_1.png depth maps (made when missing).",
    },
    "moss_image_edit": {
        "about": "MOSS-Image editing LoRAs: the model learns an edit from each target image (the result) and its reference images (the source), trained on MOSS-Image Base, LoRA training only.",
        "all": [
            "Every dataset folder needs a control_images sub-folder with each target's reference: photo.png -> control_images/photo_0.png (more references: photo_1.png ...); the tab refuses to start without it. Each reference keeps its own framing at the training resolution's pixel count, so the dataset batch size stays 1.",
            "The text encoder sees every reference before the caption ('Picture 1: <image>' and your instruction, through Qwen3.5's vision tower at up to 384 x 384 pixels), and the DiT sees its latent beside the image it denoises, as ai-toolkit trains MOSS-Image edit LoRAs.",
            "In ComfyUI load the LoRA with LoRA Loader Model Only on MOSS-Image Base or SFT and encode the prompt and the negative prompt with Text Encode MOSS-Image Edit (with the VAE and the same reference images): edit_format 'ai-toolkit' when References Attend Only to Each Other is on (the default, the references are computed once per image), 'ai-toolkit (kv_cache off)' when it is off.",
            "ChenkinLab's Moss-Image-Edit is another layout (its references sit at the text start and the text encoder does not see them): it is not a base for these LoRAs, and they do not load with its edit_format.",
            "Sample prompts take their reference images with --ci (one per reference, in order); each is fitted to the sample size as the training references are.",
            "A caption file can hold several captions (lines starting with ---===--- separate them); Caption Dropout Rate keeps the references and drops only the instruction, as the negative prompt does in ComfyUI.",
            "Up to 40 blocks can be swapped; two blocks always stay on the GPU.",
        ],
        "lora": [
            "Presets: one reference per target, DoRA rank 128, AdaFactor LR 6e-5, 1024 px, 200 epochs, a checkpoint every 25 epochs, timestep sampling shift 3; 8 GB lowers the rank to 64 with activation CPU offload, 6 GB trains 768 px at rank 32.",
            "Two families as on MOSS-Image: the _INT8 presets (ConvRot INT8 block projections) come first on 8-12 GB (pre-selected), the BF16 presets on the other tiers. INT8 swaps fewer blocks, so it steps faster wherever BF16 swaps (RTX A6000: 10 GB 6.65 against 13.08 s/step, 8 GB 15.01 against 21.41, 6 GB 9.42 against 15.72).",
            "24 and 32 GB cache the text encoder on the GPU (16.5 GB caching peak); 16 GB and smaller cache it on the CPU.",
            "More references per target add their tokens to each step and need more VRAM than the listed peak: pick a lower tier.",
            "🔍 Check dataset in 📁 Dataset shows every target beside its references and lists images without one.",
        ],
        "vram": "One reference per target at 1024 px, rank 128: 16.6 GB without block swap on 24 and 32 GB (text caching; training 16.4 GB); 16 GB 14.2 GB with block swap 13, H2D-only (its INT8 preset 12.1 GB without block swap); on 12 and 10 GB the INT8 presets come first with 20 and 38 blocks H2D-only (10.3 and 8.3 GB); 8 GB 6.3 GB at rank 64 with activation CPU offload; 6 GB 4.3 GB at 768 px, rank 32.",
        "dataset": "Target images with same-name .txt captions (the edit instruction) plus each target's reference in control_images (photo_0.png).",
    },
    "fibo2": {
        "about": (
            "Bria FIBO 2.0, an 8.1B text-to-image DiT (44 blocks, a Perceiver over the text) with the Qwen3-VL 4B text encoder and the "
            "FLUX.2 VAE (LoRA or full fine-tuning); editing LoRAs train in the FIBO 2.0 Edit card."
        ),
        "all": [
            "Train on the BF16 Dev model (fibo2_dev_bf16 from Comfy-Org/Fibo-2.0); the LoRA also loads on Edit-Turbo and on the INT8 ConvRot files in ComfyUI.",
            "Captions: plain text is encoded as written, Bria's structured JSON captions are cleaned and compacted as in Bria's training, an empty caption is the null caption.",
            "Timestep sampling flux2_shift is Bria's own shift by image size (mu 0.5 at 256 tokens of 16 px up to 1.15 at 4096, which is 1024 x 1024).",
            "Training samples use ComfyUI's sampler: shift 1.15 at every size, 30 steps, CFG 5 against the null caption (an empty negative prompt); a negative prompt you write is put into the chat template as it is.",
            "Up to 42 blocks can be swapped; two of the 44 blocks, the noise refiner and the Perceiver always stay on the GPU.",
        ],
        "lora": [
            "LoRA presets: DoRA rank 128, AdaFactor LR 1e-4, 1024 px, 200 epochs, a checkpoint every 25 epochs; below 12 GB most presets lower the rank and add activation CPU offload (the 8 GB BF16 and the 6 GB INT8 preset also train 768 px at rank 32).",
            "Two families share the folder: the BF16 presets train the DiT as released, the _INT8 presets keep its block linears in ConvRot INT8 (quantized while loading). From 16 GB up BF16 leads its tier and INT8 follows one rung lower; on 8-12 GB the INT8 preset comes first (pre-selected) and BF16 follows; 6 GB has only the INT8 preset.",
            "INT8 swaps fewer blocks, so it steps faster wherever BF16 swaps (RTX A6000): 16 GB 5.09 against 6.57 s/step, 12 GB 5.60 against 9.18; without block swap (24 and 32 GB) both take about 5.1 s/step.",
            "ConvRot INT8 Base (Model) turns INT8 on for any preset (Comfy-Org's fibo2_dev_int8_convrot file holds the same values and also trains as it is).",
            "10 GB and smaller tiers cache the text encoder on the CPU.",
        ],
        "full": [
            "Fine-tuning presets: every DiT weight in BF16 with fused-backward AdaFactor at LR 1e-5 (Triton kernels), 1024 px, 200 epochs, a checkpoint every 25 epochs with only the newest training state kept.",
            "With block swap, the Trainable Block-Swap Ring streams the swapped blocks through reusable GPU buffers from pinned CPU copies, written back after each update.",
            "The fine-tuned DiT is saved in the released layout (fused qkv), so it loads as the DiT for more training and in ComfyUI.",
            "4.17-4.18 s/step without block swap on an RTX A6000 (24 and 32 GB); 8.56, 11.46, 12.69 and 17.16 s/step with 20, 33, 39 and 42 swapped blocks (16, 12, 10 and 8 GB).",
        ],
        "vram": {
            "lora": "LoRA at 1024 px, rank 128: BF16 21.9 GB without block swap (24 and 32 GB) and 14.2 GB with block swap 28, H2D-only (16 GB); INT8 14.4 GB without block swap, 10.3 GB with block swap 29, H2D-only (12 GB) and 8.3 GB with block swap 42 (10 GB); 6.3 GB at rank 64 with activation CPU offload (8 GB, INT8); 4.4 GB at 768 px, rank 32 (6 GB, INT8).",
            "full": "Fine-tuning at 1024 px: 20.2 GB without block swap (24 and 32 GB), 14.1 GB with block swap 20 (16 GB), 10.2 GB with 33 (12 GB), 8.3 GB with 39 (10 GB) and 6.2 GB with 42 and activation CPU offload (8 GB); no preset fits 6 GB.",
        },
        "dataset": "1024x1024 images in folders like 1_name with same-name .txt captions (plain text or Bria's JSON captions).",
    },
    "fibo2_edit": {
        "about": (
            "FIBO 2.0 editing LoRAs: the model learns an edit from each target image (the result) and its reference images (the source), "
            "on the Dev model with the references as ai-toolkit trains it (LoRA training only)."
        ),
        "all": [
            "Every dataset folder needs a control_images sub-folder with each target's reference: photo.png -> control_images/photo_0.png (more references: photo_1.png ...); the tab refuses to start without it.",
            "A caption is the edit instruction (or Bria's JSON caption of the result with an edit_instruction key); Qwen3-VL reads it together with the reference images, as Bria's pipeline encodes an edit.",
            "A single reference is cropped to its target's size (the edit lines up with it pixel for pixel); several references each keep their own aspect ratio at the target's pixel count, Bria's rules.",
            "🔍 Check dataset in 📁 Dataset shows every target beside its references and lists images without one.",
            "Train on the BF16 Dev model (fibo2_dev_bf16); the LoRA also loads on Edit-Turbo, the step-distilled edit model, in ComfyUI.",
            "Sample prompts take their reference images with --ci (one per reference, in order): one reference is cropped to the sample's size, several to their own 1 MP sizes, as ComfyUI's TextEncodeFibo2 node does.",
            "Up to 42 blocks can be swapped; two of the 44 blocks, the noise refiner and the Perceiver always stay on the GPU.",
        ],
        "lora": [
            "Presets: one reference per target, DoRA rank 128, AdaFactor LR 1e-4, 1024 px, 200 epochs, a checkpoint every 25 epochs; 12 GB and below add activation CPU offload, and the lower tiers step down in rank and resolution (8 GB: INT8 768 px, BF16 512 px, both rank 32). No configuration fits 6 GB.",
            "Two families as on FIBO 2.0: from 16 GB up the BF16 preset leads its tier and the _INT8 preset (ConvRot INT8 block linears) follows one rung lower; on 8-12 GB the INT8 preset comes first (pre-selected).",
            "Each reference adds its latent tokens to the image and its vision tokens to the text, so editing needs more VRAM than text-to-image; more references per target need more than the listed peak: pick a lower tier.",
            "On an RTX A6000 about 9.7-10 s/step on 24 and 32 GB, 9.65 with INT8 and 11.61 with BF16 on 16 GB; activation CPU offload on 8-12 GB takes 15-31 s/step.",
            "ConvRot INT8 Base (Model) turns INT8 on for any preset (Comfy-Org's fibo2_dev_int8_convrot file holds the same values and also trains as it is).",
        ],
        "vram": "One reference per target at 1024 px, rank 128: BF16 25.1 GB without block swap (32 GB), 22.1 GB with block swap 12 and 14.3 GB with block swap 38, both H2D-only (24 / 16 GB); INT8 17.4 GB without block swap (24 and 32 GB), 14.3 GB with block swap 23, H2D-only (16 GB) and 10.2 GB with activation CPU offload (12 GB); 8.3 and 6.4 GB on 10 and 8 GB (INT8); nothing fits 6 GB.",
        "dataset": "1024x1024 target images with same-name .txt captions (the edit instruction) plus one reference per target in control_images (photo_0.png).",
    },
    "anima": {
        "about": (
            "CircleStone Labs Anima, a 2B anime and illustration text-to-image DiT (NVIDIA Cosmos-Predict2 with an LLM adapter) "
            "with the Qwen3-0.6B text encoder and the Qwen-Image VAE (LoRA or full fine-tuning)."
        ),
        "all": [
            "Train on Anima-Base (anima-base-v1.0, the model card's choice for LoRAs); the LoRA also loads on Anima-Aesthetic and Anima-Turbo in ComfyUI.",
            "The LLM adapter between the text encoder and the DiT stays frozen: the model card warns that training it degrades the model. Network Arguments train_llm_adapter=True adds it to a LoRA, Train the LLM Adapter to a fine-tune.",
            "Captions can be danbooru-style tags, natural language or both; the model card orders tags as quality / meta / year / safety, 1girl / 1boy, character, series, @artist, then general tags.",
            "Training samples follow ComfyUI's default Anima workflow: er_sde on the simple schedule (shift 3), 30 steps, CFG 4 with its negative prompt.",
            "Up to 26 blocks can be swapped; two blocks always stay on the GPU.",
        ],
        "lora": [
            "LoRA presets: DoRA rank 128, AdaFactor LR 5e-5 (picked from 200-epoch tests of 2e-5 to 2e-4; the model card starts at 2e-5), 1024 px, 200 epochs, a checkpoint every 25 epochs, on every tier.",
            "BF16 trains without block swap from 10 GB (2.54-2.70 s/step on an RTX A6000); 8 GB streams 10 blocks H2D-only at the same speed (2.64 s/step), 6 GB 26 blocks (3.05 s/step).",
            "The _INT8 presets (ConvRot INT8 DiT) follow BF16 one rung lower: Anima's blocks are small, so INT8 saves VRAM but costs time, about a quarter slower (3.21-3.45 against 2.54-2.70 s/step without block swap).",
        ],
        "full": [
            "Fine-tuning presets: every DiT weight in BF16 with fused-backward AdaFactor at LR 2e-6 (max grad norm 0, Triton kernels), 1024 px, 200 epochs, a checkpoint every 25 epochs with only the newest training state kept.",
            "Every weight moves at once: at LR 1e-5 a short test on photos turned the samples blurry within 4 steps, at 2e-6 they stayed clean, so start low.",
            "The fine-tuned DiT is saved in the released layout (anima-base-v1.0's tensors, BF16), so it loads as the DiT for more training, in ComfyUI and in SwarmUI.",
            "1.04-1.15 s/step without block swap on an RTX A6000 (8 GB and up); the 6 GB preset swaps 12 blocks through the Trainable Block-Swap Ring (2.24 s/step).",
        ],
        "vram": {
            "lora": "LoRA at 1024 px, rank 128: 7.4 GB without block swap (10-32 GB), 6.4 GB with block swap 10 and 4.3 GB with block swap 26, both H2D-only (8 / 6 GB); INT8 5.9 GB without block swap (8-32 GB), 4.4 GB with block swap 22, H2D-only (6 GB).",
            "full": "Fine-tuning at 1024 px: 5.9 GB without block swap (8-32 GB), 4.4 GB with block swap 12 (6 GB).",
        },
        "dataset": "1024x1024 images in folders like 1_name with same-name .txt captions (tags, natural language or both).",
    },
    "mage_flow": {
        "about": "Microsoft Mage-Flow (4B), a native-resolution text-to-image model (LoRA training only).",
        "all": [
            "The trainer uses uniform_shift timesteps with static shift 6, matching the released inference scheduler.",
            "Only SDPA and FlashAttention attention backends are supported.",
            "Plain FP8 Base is rejected (enable Scaled FP8 with it), and torch.compile fullgraph mode is not supported.",
            "Training samples default to 30 steps at CFG 5.0.",
        ],
        "lora": [
            "Presets: rank 128, AdaFactor LR 1e-4, 1024 px, 200 epochs, a checkpoint every 25 epochs.",
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
            "Presets: rank 128, AdaFactor LR 1e-4, 1024 px, 200 epochs; the 6 GB tier trains 768 px at rank 32.",
            "The 32 GB Fast preset turns off gradient checkpointing (28.6 GB peak, 1.24 s/step on an RTX 5090).",
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
            "10 GB and 8 GB tiers drop to rank 64 / 32 with activation CPU offload (8 GB also trains at 512 px).",
        ],
        "vram": "T2V 14B: 29.2 GB with block swap 9 on 32 GB down to 4.9 GB (512 px, rank 32, block swap 38) on 8 GB.",
        "dataset": "Images, 16 fps videos or both at 960x960 (or 1280x720 / 720x1280), 4n+1 frames (81 = about 5 s), each with a same-name .txt caption; image-only folders work too.",
    },
    "ltx25": {
        "about": "Lightricks LTX 2.5 (22B) video model in separate files: the transformer, the Gemma 4 12B text encoder with the LTX text projections, the video VAE and the audio VAE with its vocoder, for video or joint audio+video.",
        "all": [
            "LTX 2.5 needs its own Gemma 4 text encoder and VAEs: the LTX 2.3 checkpoint and the Gemma 3 files do not work with it, and caches made for LTX 2.3 are encoded again.",
            "Audio + Video learns sound from each clip's audio track or a same-stem .wav file; image samples give no audio signal.",
            "Latent caching, text caching and training must use the same modality (Video only or Audio + Video).",
            "Training samples run the dev transformer's full schedule (Sampling Preset ltx25_full, with CFG and STG) and decode with the diffusion video VAE, whose tiles overlap by 160 px and 40 frames.",
            "Use Lightricks' Gemma 4 text encoder file of 17 August 2026 or later: the trainer refuses the copy published before, whose encoder does not match the released transformer.",
            "Every preset caches text with Low-VRAM Text Caching: the 24.5 GB Gemma 4 streams through the GPU one decoder layer at a time (5.7 GB caching peak, identical embeddings) and then sits in system RAM; the sample prompts are encoded in the same step, so training never loads the text encoder.",
            "Transformer Variant Pre-Trained trains Lightricks' LTX-2.5 Pre-Trained checkpoint, the model before supervised fine-tuning and post-training, meant for specialized fine-tunes; it shares the Dev text encoder, VAEs and presets, and only its transformer is a separate download (LTX 2.5 Pre-Trained transformer). A 200-epoch DoRA test on it (4.5e-5 to 3.6e-4) kept the presets' 9e-5.",
        ],
        "lora": [
            "Presets: DoRA rank 128, AdaFactor LR 9e-5, 768x512, 33 frames, 200 epochs, a checkpoint every 25 epochs; the 8 GB tier trains rank 64 with activation CPU offload.",
            "The BF16 presets train on the transformer as released, the best quality: Quality 1, pre-selected on 16-32 GB. The _INT8 presets quantize it to ConvRot INT8 while loading and are pre-selected on 8-12 GB (Quality 1 INT8; 8 GB: Quality 2 INT8), with the BF16 preset one rung below; from 16 GB up they sit one rung below BF16. From 24 GB down they swap fewer blocks and need 27-59% less time per step on an RTX A6000 (24 GB 6.14 against 8.98 s/step, 10 GB 12.34 against 16.85, 8 GB 32.73 against 79.60); at 32 GB BF16 is the faster one (4.25 against 5.72). In a 50-epoch check on the same clips both learned the subject at the same pace, with the scenes intact.",
            "Training samples are slow on the presets that swap blocks: 10 sampling steps took 2-5 minutes on the 8-12 GB INT8 presets and 5.5-10 minutes on the BF16 ones (RTX A6000 with other jobs running; samples default to 30 steps), so sample with few prompts and not too often.",
            "Video only builds a video-only transformer, so its LoRAs leave audio untouched; in Audio + Video, video_* target presets do the same.",
        ],
        "full": [
            "Presets: every transformer weight trains in BF16 with fused-backward AdaFactor, LR 1e-5 (max grad norm 0), 768x512, 33 frames, 200 epochs, a checkpoint every 25 with only the newest training state kept; they need 64 GB+ system RAM.",
            "With block swap the swapped blocks stream through the Trainable Block-Swap Ring from pinned CPU copies, written back after each update.",
            "The fine-tuned transformer is saved as its own file, in the form of the released LTX 2.5 transformer (no bundle merge).",
            "6.44 s/step on an RTX A6000 with block swap 1 (32 GB); 12.79, 19.24, 22.41 and 24.42 s/step with 17, 33, 41 and 45 blocks on the ring (24, 16, 12 and 10 GB), measured one preset at a time.",
        ],
        "vram": {
            "lora": "BF16 presets: 32 GB without block swap (29.7 GB), 24, 16, 12 and 10 GB with 17, 33, 41 and 45 blocks H2D-only (22.2, 14.2, 10.2 and 8.2 GB), 8 GB at rank 64 with 46 blocks and activation CPU offload (6.4 GB). INT8 presets: 24 and 32 GB without block swap (17.8 GB), 16, 12 and 10 GB with 16, 32 and 40 blocks H2D-only (14.3, 10.3 and 8.2 GB), 8 GB at rank 64 with 40 blocks and activation CPU offload (6.4 GB). Text caching alone peaks at 5.7 GB, so no preset fits 6 GB; up to 47 of the 48 blocks can be swapped.",
            "full": "32 GB with block swap 1 (30.2 GB); 24, 16, 12 and 10 GB swap 17, 33, 41 and 45 blocks through the trainable ring (22.2, 14.2, 10.2 and 8.2 GB); nothing fits 8 GB; up to 47 of the 48 blocks can be swapped on the trainable ring.",
        },
        "dataset": "Images (1-frame samples), videos or both at 768x512 in 32 px steps, 8k+1 frames resampled to 25 fps, .txt captions; Audio + Video uses each folder's videos and needs their sound described.",
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
            "LoRA presets: rank 128, AdaFactor LR 2e-4, 768x512, 33 frames, INT8 ConvRot base, 200 epochs, a checkpoint every 25; no block swap on 24 and 32 GB.",
            "The 10, 12 and 16 GB presets stream their swapped blocks H2D-only (frozen, so never copied back): the same VRAM as swapping both ways, 1.9-2.4x faster per step and about 6x faster training samples with Sample With Offloading on an RTX A6000.",
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
            "lora": "LoRA at 768x512, 33 frames, rank 128: 18.3 GB peak without block swap (24 and 32 GB), 14.2 GB with block swap 18 H2D-only (16 GB), 10.3 GB with block swap 34 H2D-only (12 GB), 8.3 GB with block swap 42 H2D-only (10 GB), 5.9 GB at rank 64 with block swap 42, activation CPU offload and Low-VRAM Text Caching (8 GB); up to 47 of the 48 blocks can be swapped.",
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
        "vram": "At 124 frames: 512 px on 32 GB (30 GB peak), 384 px on 24 GB (21.7 GB), 256 px on 16 GB (14.4 GB) with H2D-only block swap; 256 px at rank 64 on 12 and 10 GB (9.5 and 8.4 GB, block swap 45 and 48); up to 48 of 50 blocks swap; 64 GB system RAM recommended.",
        "dataset": "Videos only (image folders are skipped), 17n+5 frames (default 124), sides in 32 px steps, normalized to 24 fps, one .txt caption per clip.",
    },
    "kandinsky6": {
        "about": "Kandinsky 6 Lite generates video together with audio at 24 fps from text (t2av) or text plus a first frame (ti2av).",
        "all": [
            "Frame counts must be 4n+1 (81, 97, 121; other counts are rounded down); 121 frames is about 5 seconds at 24 fps.",
            "Batch size is fixed to 1; raise gradient accumulation instead.",
            "Clips with sound also train audio (video track or same-stem audio file); image folders train appearance as silent one-frame videos.",
            "ti2av needs a JSONL dataset with a first-frame control_path per clip, loaded with Use TOML File.",
            "r2av (multi-reference): each clip comes with up to 8 reference images that its caption names @Image 1, @Image 2 ... in order; the JSONL lists them per clip (\"references\": [{\"type\": \"image\", \"path\": ...}]), loaded with Use TOML File. A released checkpoint has no reference slots: train Fine-Tuning with Task r2av first, then LoRAs on the DiT it saves. Sample prompts add each reference with --ref <image>.",
            "Train the regular Lite or Pro transformer; distilled checkpoints cannot be trained, and attention is SDPA only.",
        ],
        "lora": [
            "Presets: rank 32, AdamW8bit LR 1e-4, max grad norm 1.0, 100 epochs, a checkpoint every 10 epochs.",
            "Video Only turns the audio loss off; Audio Loss Weight scales it for clips with real audio.",
            "ConvRot INT8 base (needs Triton) and FP8 Base are mutually exclusive; choose one base quantization.",
        ],
        "full": [
            "Fine-tuning presets (Lite): every DiT weight in BF16 with fused-backward AdaFactor at LR 1e-5, max grad norm 0, 100 epochs, a checkpoint every 25 (about 7.5 GB each); the video and audio VAEs and the text encoders stay frozen.",
            "With block swap the presets use the Trainable Block-Swap Ring: the swapped blocks stream through reusable GPU buffers from pinned CPU copies (about 7.5 GB of system RAM), written back after each update; it turns Pinned Memory for Block Swap on. Frozen Blocks keeps the visual blocks you list unchanged.",
            "ConvRot INT8 and FP8 bases are for LoRA: fine-tuning loads the BF16 transformer.",
            "The fine-tuned DiT is saved as one .safetensors file with the model's own keys, which loads as the DiT for more training or for samples.",
        ],
        "vram": {
            "lora": "32 GB: 864x480 at 121 frames without block swap (22.4 GB peak), 24 GB with block swap 10 (22.3 GB); 16 GB: 864x480 with block swap 25 and activation CPU offload (14.2 GB); 12 GB: 640x352 at 121 frames; 10 and 8 GB: 512x288 at 97 frames; 6 GB: 512x288 at 49 frames (4.3 GB); Lite swaps at most 30 blocks.",
            "full": "Fine-tuning at 864x480, 121 frames: 24 GB without block swap (32 GB tier), 22.2 GB with block swap 10 (24 GB), 14.3 GB with block swap 25 and activation CPU offload (16 GB); 12 and 10 GB train 640x352, 8 GB 512x288 at 97 frames, 6 GB 512x288 at 49 frames (4.3 GB). Lite swaps at most 30 of its 32 blocks, Pro 58 of 60.",
        },
        "dataset": "Clips with sound at 864x480 (16 px steps), 4n+1 frames (121 = about 5 s at 24 fps), images, or both in one folder, each with a same-name .txt caption.",
    },
    # ------------------------------------------------------------------ audio models
    "ltx25_audio": {
        "about": "Audio-only DoRA / LoRA or fine-tune of the LTX 2.5 audio branch for speech, music and sound effects, with no video target.",
        "all": [
            "Audio clips are used as they are, and videos with a sound track count too: Start extracts their audio to WAV (cache folder, extracted_audio) with the videos' captions; silent videos are skipped.",
            "Latent caching, text caching and training all run in audio mode, and training samples are audio-only previews.",
            "Caption each clip with a same-stem .txt that describes the sound: speech, music or ambient noise.",
            "LTX 2.5 needs its own audio VAE with vocoder and the Gemma 4 12B text encoder: the LTX 2.3 files do not work with it.",
            "Transformer Variant Pre-Trained trains Lightricks' LTX-2.5 Pre-Trained checkpoint (before supervised fine-tuning and post-training) with the same audio VAE and text encoder; only its transformer is a separate download.",
        ],
        "lora": [
            "No presets yet: the defaults load rank 128 with AdaFactor LR 1e-4, INT8 ConvRot base, block swap 20, Low-VRAM Text Caching, 100 epochs, a checkpoint every 10.",
            "Keep the LoRA target preset on audio: it trains only the audio attention and FFN layers.",
        ],
        "full": [
            "No presets yet: the defaults follow the LTX fine-tuning recipe (dense BF16 weights, fused-backward Adafactor at LR 1e-5, max grad norm 0) with block swap 20, 100 epochs, a checkpoint every 10; it needs 64 GB+ system RAM.",
            "Keep every base quantization off (the defaults do) and use the BF16 LTX 2.5 dev transformer; mixed precision must be bf16.",
            "The fine-tuned transformer is saved as its own file, in the form of the released LTX 2.5 transformer (no bundle merge).",
        ],
        "vram": {
            "lora": "No presets yet, so no measured VRAM; the defaults swap 20 of the 48 blocks (up to 47 can be swapped, and LoRA can stream them H2D-only).",
            "full": "No presets yet, so no measured VRAM; the defaults swap 20 of the 48 blocks on the trainable ring (up to 47 can be swapped).",
        },
        "dataset": "Sub-folders such as 1_my_voice with audio clips (.wav, .flac, .mp3, .ogg, .m4a, .aac, .opus, .wma) and/or videos with sound, each with a same-stem .txt caption describing the sound.",
    },
    "ltx23_audio": {
        "about": "Audio-only LoRA or fine-tune of the LTX 2.3 audio branch for speech, music and sound effects, with no video target.",
        "all": [
            "Audio clips are used as they are, and videos with a sound track count too: Start extracts their audio to WAV (cache folder, extracted_audio) with the videos' captions; silent videos are skipped.",
            "Latent caching, text caching and training all run in audio mode, and training samples are audio-only previews.",
            "Caption each clip with a same-stem .txt that describes the sound: speech, music or ambient noise.",
            "The LTX 2.3 checkpoint supplies the audio VAE and vocoder; the Gemma 3 12B text encoder is still needed for text caching.",
        ],
        "lora": [
            "No presets yet: the defaults load rank 128 with AdaFactor LR 1e-4, INT8 ConvRot base, block swap 20, 100 epochs, a checkpoint every 10.",
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
