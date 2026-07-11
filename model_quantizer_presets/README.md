# Model Quantizer Presets

Preset review date: 2026-07-11. The GUI favors a proven model recipe over applying one quantization format to every architecture.

## Quality Presets

| Preset | Intended use | Runtime |
|---|---|---|
| INT8 ConvRot Balanced | Default. Simple row-wise INT8 with fixed ConvRot group 256 | Native current ComfyUI, Turing+ |
| INT8 ConvRot Learned | Higher reconstruction quality with Prodigy and 2,000 iterations | Native current ComfyUI, Turing+ |
| INT8 Tensorwise | Models whose dimensions or public recipes do not support fixed ConvRot well | Native current ComfyUI, Turing+ |
| FP8 Simple | Fast, low-memory FP8 conversion | Native current ComfyUI; best acceleration on Ada+ |
| FP8 Learned | Balanced or high-quality learned FP8 | Native current ComfyUI; best acceleration on Ada+ |
| FP8 Compatibility | Full-precision matmul metadata for known quality/compatibility exceptions | Native current ComfyUI fallback path |
| INT8 Blockwise | Experimental block layout only | QuantOps required |
| MXFP8 / NVFP4 | Expert Blackwell formats | Native current ComfyUI on Blackwell |

`full_precision_matrix_mult` is false for native compute presets. It is true only for the compatibility preset and the Qwen Image quality-first recipe. The removed aggressive Z-Image NVFP4 preset is mapped to FP8 Compatibility when an old config is loaded.

## Model Defaults

| Model preset | Default | Important protected behavior |
|---|---|---|
| FLUX.2 | INT8 ConvRot Balanced | Modulation, guidance, time, final, and input layers |
| FLUX 2 Klein | INT8 ConvRot Balanced | Flux-sensitive layers except the absent guidance input |
| Krea 2 Raw/Turbo | INT8 ConvRot Balanced | First/last, time, input, final, and text-fusion layers; no layer-config ConvRot bypass |
| Ideogram 4 | INT8 ConvRot Balanced | Image indicator, time, condition, AdaLN, final, and input projections |
| WAN 2.1/2.2 | INT8 ConvRot Balanced | Embeddings, condition/audio paths, packers, heads, and adapters |
| LTX 2/2.3 | INT8 ConvRot Balanced | Boundary blocks, connectors, VAE, vocoder, and gates |
| Gemma 4 | INT8 ConvRot Balanced | Embeddings, K/V projections, audio/vision, and multimodal projectors |
| Qwen3.5 | INT8 ConvRot Balanced | Layers 0 and 23/31/63, embeddings, MTP, and complete visual stack |
| T5-XXL | INT8 ConvRot Balanced | Norm/bias exclusions, decoder removal, and input scales |
| Radiance / distilled / NeRF | INT8 ConvRot Balanced | Their upstream high-precision boundary and guidance layers |
| Boogu-Image | INT8 Tensorwise | Embeddings plus `norm1.linear` and `norm_out`; avoids incompatible block/ConvRot widths |
| Qwen Image / Edit | FP8 Learned Reference | Learned rounding and full-precision matmul; Qwen-sensitive norms and boundary layers |
| Z-Image / Refiner | FP8 Compatibility | Z-Image norms, embeddings, final layers, and refiner blocks |
| Anima Base/Turbo | FP8 Compatibility | Initial blocks, AdaLN, adapters, time/input/final layers |
| FLUX.1 | FP8 Learned Reference | Flux-sensitive layers and input-scale tensors |
| ERNIE Image | FP8 Learned Reference | Author-recommended time, AdaLN, first attention/MLP, final, and text projection layers |
| Hunyuan Video / LENS | FP8 Learned Balanced | Existing upstream sensitive-layer filters; no public ConvRot validation yet |
| Generic Text Encoder | FP8 Learned Balanced | Input-scale behavior only; no architecture-specific exclusion claim |

## Safety Rules

- Fixed ConvRot group 256 is the stable default. Dynamic groups remain an explicit experiment.
- The GUI blocks detected quantized inputs from the Quantize workflow unless the expert override is enabled.
- W4A4 ConvRot is not listed yet: ComfyUI can load the new layout, but convert_to_quant cannot currently create it.
- Choose the model preset before changing the quality preset so high-precision exclusions remain active.

Upstream references: [convert_to_quant](https://github.com/silveroxides/convert_to_quant), [model filter PR 27](https://github.com/silveroxides/convert_to_quant/pull/27), [ComfyUI](https://github.com/comfyanonymous/ComfyUI), and [comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen).
