import gc
import glob
import os
import subprocess
import sys
from typing import List, Optional, Tuple

import gradio as gr
import psutil
import torch

from .class_gui_config import GUIConfig
from .common_gui import (
    get_folder_path,
    get_model_file_path,
    get_saveasfilename_path,
    scriptdir,
    setup_environment,
)
from .custom_logging import setup_logging

PYTHON = sys.executable
EXTRACT_SCRIPT = os.path.join(scriptdir, "musubi-tuner", "qwen_extract_lora.py")

log = setup_logging()


def _normalize_extensions(ext_text: str, fallback: List[str]) -> List[str]:
    if not ext_text:
        return fallback
    result = []
    for part in ext_text.split(","):
        value = part.strip()
        if not value:
            continue
        if not value.startswith("."):
            value = "." + value
        result.append(value)
    return result or fallback


class LoRAExtractor:
    def __init__(self, headless: bool, config: Optional[GUIConfig]) -> None:
        self.headless = headless
        self.config = config
        self.single_process: Optional[subprocess.Popen] = None
        self.batch_process: Optional[subprocess.Popen] = None
        self.single_cancel_requested = False
        self.batch_cancel_requested = False

    @staticmethod
    def _resolve_device(choice: str) -> Optional[str]:
        if not choice:
            return None
        normalized = choice.lower()
        if "auto" in normalized:
            return "cuda" if torch.cuda.is_available() else "cpu"
        if normalized == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            log.warning("CUDA requested but not available, falling back to CPU.")
            return "cpu"
        if normalized == "cpu":
            return "cpu"
        return None

    @staticmethod
    def _map_precision(choice: str) -> Optional[str]:
        mapping = {
            "fp16": "fp16",
            "bf16": "bf16",
            "float32": "float",
            "auto (preserve tensors)": None,
        }
        if not choice:
            return mapping["auto (preserve tensors)"]
        normalized = choice.lower()
        return mapping.get(normalized, mapping["auto (preserve tensors)"])

    @staticmethod
    def _summarize_output(output: str) -> str:
        if not output:
            return " Check the console logs for additional details."
        tail = "\n".join(output.strip().splitlines()[-8:])
        return f"\nLast output:\n{tail}"

    @staticmethod
    def _is_running(process: Optional[subprocess.Popen]) -> bool:
        return process is not None and process.poll() is None

    @staticmethod
    def _ensure_script_exists() -> None:
        if not os.path.isfile(EXTRACT_SCRIPT):
            raise FileNotFoundError(
                f"LoRA extractor script not found at '{EXTRACT_SCRIPT}'. "
                "Ensure the musubi-tuner repository is present."
            )

    def _terminate_process(self, process: Optional[subprocess.Popen]) -> bool:
        if not self._is_running(process):
            return False
        try:
            parent = psutil.Process(process.pid)
            for child in parent.children(recursive=True):
                try:
                    child.kill()
                except psutil.Error:
                    continue
            parent.kill()
            return True
        except psutil.Error:
            return False

    def _build_command(
        self,
        model_org: str,
        model_tuned: str,
        save_to: str,
        dim: int,
        clamp_quantile: float,
        device: Optional[str],
        save_precision: Optional[str],
        mem_eff_safe_open: bool,
        keep_metadata: bool,
    ) -> List[str]:
        self._ensure_script_exists()
        command = [
            PYTHON,
            EXTRACT_SCRIPT,
            "--model_org",
            model_org,
            "--model_tuned",
            model_tuned,
            "--save_to",
            save_to,
            "--dim",
            str(int(dim)),
            "--clamp_quantile",
            str(float(clamp_quantile)),
        ]
        if device:
            command.extend(["--device", device])
        if mem_eff_safe_open:
            command.append("--mem_eff_safe_open")
        if save_precision:
            command.extend(["--save_precision", save_precision])
        if not keep_metadata:
            command.append("--no_metadata")
        return command

    def _run_command_with_logging(
        self,
        command: List[str],
        process_attr: str,
    ) -> Tuple[int, str]:
        env = setup_environment()
        log.info("Executing LoRA extraction command: %s", " ".join(command))
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        setattr(self, process_attr, process)
        lines: List[str] = []
        try:
            if process.stdout:
                for raw_line in process.stdout:
                    line = raw_line.rstrip("\r\n")
                    lines.append(line)
                    if line:
                        log.info(line)
            process.wait()
            rc = process.returncode if process.returncode is not None else 0
            output = "\n".join(lines)
            return rc, output
        finally:
            setattr(self, process_attr, None)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def extract_single(
        self,
        model_org: str,
        model_tuned: str,
        save_to: str,
        dim: float,
        clamp_quantile: float,
        device_choice: str,
        save_precision_choice: str,
        mem_eff_safe_open: bool,
        keep_metadata: bool,
        overwrite_existing: bool,
    ) -> str:
        if self._is_running(self.single_process):
            return "⚠️ A LoRA extraction is already running. Please cancel it before starting another."

        if not model_org or not os.path.isfile(model_org):
            return f"❌ Base/original model not found: {model_org}"

        if not model_tuned or not os.path.isfile(model_tuned):
            return f"❌ Tuned model not found: {model_tuned}"

        if not save_to:
            return "❌ Output path is required."

        output_dir = os.path.dirname(save_to)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
            except OSError as exc:
                return f"❌ Unable to create output directory '{output_dir}': {exc}"

        if os.path.exists(save_to) and not overwrite_existing:
            return f"⊘ Skipped: Output already exists -> {save_to}"

        device = self._resolve_device(device_choice)
        save_precision = self._map_precision(save_precision_choice)

        try:
            command = self._build_command(
                model_org=model_org,
                model_tuned=model_tuned,
                save_to=save_to,
                dim=int(dim),
                clamp_quantile=clamp_quantile,
                device=device,
                save_precision=save_precision,
                mem_eff_safe_open=mem_eff_safe_open,
                keep_metadata=keep_metadata,
            )
        except FileNotFoundError as exc:
            return f"❌ {exc}"

        log.info("🚀 Executing LoRA extraction command:")
        log.info(" ".join(command))

        self.single_cancel_requested = False

        try:
            returncode, output = self._run_command_with_logging(command, "single_process")
        except Exception as exc:
            log.error("Failed to start LoRA extraction: %s", exc)
            return f"❌ Failed to start extraction: {exc}"

        was_cancelled = self.single_cancel_requested
        self.single_cancel_requested = False

        if was_cancelled:
            return "⛔ Extraction cancelled."

        if returncode == 0:
            return (
                "✅ LoRA extraction complete!\n"
                f"- Base Model: {model_org}\n"
                f"- Tuned Model: {model_tuned}\n"
                f"- Saved To: {save_to}\n"
                f"- Rank (dim): {int(dim)}\n"
                f"- Device: {device or 'cpu'}\n"
                f"- Clamp Quantile: {clamp_quantile}\n"
                f"- Save Precision: {save_precision or 'match tensors'}"
            )

        return (
            f"❌ Extraction failed (exit code {returncode})."
            f"{self._summarize_output(output)}"
        )

    def batch_extract(
        self,
        base_model_path: str,
        tuned_folder: str,
        output_folder: str,
        tuned_suffix: str,
        output_suffix: str,
        tuned_extensions: str,
        output_extension: str,
        recursive: bool,
        dim: float,
        clamp_quantile: float,
        device_choice: str,
        save_precision_choice: str,
        mem_eff_safe_open: bool,
        keep_metadata: bool,
        overwrite_existing: bool,
    ) -> str:
        if self._is_running(self.batch_process):
            return "⚠️ Batch extraction already running. Please cancel it before starting a new one."

        if not base_model_path or not os.path.isfile(base_model_path):
            return f"❌ Base/original model does not exist: {base_model_path}"

        if not tuned_folder or not os.path.isdir(tuned_folder):
            return f"❌ Tuned models folder does not exist: {tuned_folder}"

        tuned_exts = _normalize_extensions(
            tuned_extensions, [".safetensors", ".pt", ".bin"]
        )

        output_ext = output_extension.strip() or ".safetensors"
        if not output_ext.startswith("."):
            output_ext = "." + output_ext

        search_pattern = "**/*" if recursive else "*"
        tuned_files: List[str] = []
        for ext in tuned_exts:
            pattern = os.path.join(tuned_folder, f"{search_pattern}{ext}")
            tuned_files.extend(glob.glob(pattern, recursive=recursive))

        tuned_files = sorted(set(tuned_files))

        if not tuned_files:
            return f"⚠️ No tuned model files found in {tuned_folder} with extensions {', '.join(tuned_exts)}"

        successes = 0
        skips = 0
        failures = 0
        log_lines: List[str] = []
        device = self._resolve_device(device_choice)
        save_precision = self._map_precision(save_precision_choice)

        self.batch_cancel_requested = False

        for tuned_path in tuned_files:
            if self.batch_cancel_requested:
                log_lines.append("⛔ Cancellation requested. Stopping batch extraction.")
                break

            rel_dir = os.path.relpath(os.path.dirname(tuned_path), tuned_folder)
            rel_dir = "" if rel_dir == "." else rel_dir
            tuned_name, tuned_ext = os.path.splitext(os.path.basename(tuned_path))

            if output_suffix and tuned_name.endswith(output_suffix):
                skips += 1
                log_lines.append(f"⊘ {tuned_name}{tuned_ext} (already looks like LoRA, skipped)")
                continue

            base_stem = tuned_name
            if tuned_suffix and base_stem.endswith(tuned_suffix):
                base_stem = base_stem[: -len(tuned_suffix)]

            output_root = output_folder.strip() or tuned_folder
            output_dir = os.path.join(output_root, rel_dir)
            output_name = (
                f"{base_stem}{output_suffix}{output_ext}"
                if output_suffix
                else f"{base_stem}{output_ext}"
            )
            output_path = os.path.join(output_dir, output_name)

            if os.path.exists(output_path) and not overwrite_existing:
                skips += 1
                log_lines.append(
                    f"⊘ {tuned_name}{tuned_ext} (output already exists, skipped)"
                )
                continue

            try:
                os.makedirs(output_dir, exist_ok=True)
            except OSError as exc:
                failures += 1
                log_lines.append(
                    f"✗ {os.path.relpath(tuned_path, tuned_folder)} (error: {exc})"
                )
                continue

            try:
                command = self._build_command(
                    model_org=base_model_path,
                    model_tuned=tuned_path,
                    save_to=output_path,
                    dim=int(dim),
                    clamp_quantile=clamp_quantile,
                    device=device,
                    save_precision=save_precision,
                    mem_eff_safe_open=mem_eff_safe_open,
                    keep_metadata=keep_metadata,
                )
            except FileNotFoundError as exc:
                return f"❌ {exc}"

            log.info(
                "🚧 Extracting %s ...",
                os.path.relpath(tuned_path, tuned_folder),
            )

            try:
                returncode, output = self._run_command_with_logging(command, "batch_process")
            except Exception as exc:
                failures += 1
                log_lines.append(
                    f"✗ {os.path.relpath(tuned_path, tuned_folder)} (error: {exc})"
                )
                continue

            if self.batch_cancel_requested:
                log_lines.append("⛔ Cancellation requested. Stopping batch extraction.")
                break

            if returncode == 0:
                successes += 1
                log_lines.append(
                    f"✓ {os.path.relpath(tuned_path, tuned_folder)} → "
                    f"{os.path.relpath(output_path, output_root)}"
                )
            else:
                failures += 1
                log_lines.append(
                    f"✗ {os.path.relpath(tuned_path, tuned_folder)} "
                    f"(exit code {returncode})"
                )
                summary = self._summarize_output(output)
                if summary.strip():
                    log_lines.append(summary.strip())

        was_cancelled = self.batch_cancel_requested
        self.batch_cancel_requested = False

        summary_prefix = (
            "Batch extraction cancelled by user.\n"
            if was_cancelled
            else "Batch extraction complete!\n"
        )

        summary = (
            summary_prefix
            + f"- Success: {successes}\n"
            + f"- Skipped: {skips}\n"
            + f"- Failed: {failures}\n\n"
            + "\n".join(log_lines)
        )
        return summary

    def cancel_single(self) -> str:
        if not self._is_running(self.single_process):
            return "⚠️ No single extraction is currently running."
        self.single_cancel_requested = True
        if self._terminate_process(self.single_process):
            return "⛔ Cancellation requested. The extraction process is stopping..."
        return "⚠️ Unable to cancel the extraction. It may have already finished."

    def cancel_batch(self) -> str:
        if self._is_running(self.batch_process):
            self.batch_cancel_requested = True
            if self._terminate_process(self.batch_process):
                return "⛔ Cancellation requested. The current extraction will stop soon."
            return "⚠️ Unable to cancel the running extraction. It may have already finished."

        # If no process is currently running, flag future iterations to stop.
        if not self.batch_cancel_requested:
            self.batch_cancel_requested = True
            return "⛔ Cancellation requested. Batch extraction will stop after the current file."
        return "⚠️ Batch extraction is not running."


def lora_extractor_tab(headless: bool, config: Optional[GUIConfig]) -> None:
    extractor = LoRAExtractor(headless, config)

    gr.Markdown("# LoRA Extractor")
    gr.Markdown(
        "Extract LoRA adapters from fine-tuned checkpoints by comparing them with their original weights. "
        "Uses the Musubi Tuner SVD-based extractor for Qwen Image and compatible diffusion-style models."
    )

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.Tab("Single Extraction"):
                    with gr.Row():
                        base_model_path = gr.Textbox(
                            label="Base Model (Original)",
                            placeholder="Path to original checkpoint (e.g., ./models/base/qwen_image.safetensors)",
                        )
                        base_model_button = gr.Button("📄", size="lg")

                    with gr.Row():
                        tuned_model_path = gr.Textbox(
                            label="Tuned Model (Fine-Tuned)",
                            placeholder="Path to fine-tuned checkpoint (e.g., ./models/tuned/qwen_image_person.safetensors)",
                        )
                        tuned_model_button = gr.Button("📄", size="lg")

                    with gr.Row():
                        output_lora_path = gr.Textbox(
                            label="Output LoRA Path",
                            placeholder="Where to save the LoRA (e.g., ./loras/qwen_person_LoRA.safetensors)",
                        )
                        output_path_button = gr.Button("💾", size="lg")

                    single_status = gr.Textbox(
                        label="Extraction Log",
                        lines=14,
                        max_lines=40,
                        interactive=False,
                    )

                    extract_button = gr.Button(
                        "🔍 Extract LoRA",
                        variant="primary",
                    )
                    cancel_single_button = gr.Button(
                        "⛔ Cancel Extraction",
                        variant="secondary",
                    )

                with gr.Tab("Batch Extraction"):
                    with gr.Row():
                        batch_base_model_input = gr.Textbox(
                            label="Base Model (Original)",
                            placeholder="Path to the original/base checkpoint used for fine-tuning",
                            info="This model will be compared against every tuned checkpoint in the folder.",
                        )
                        batch_base_model_button = gr.Button("📄", size="lg")

                    with gr.Row():
                        tuned_folder_input = gr.Textbox(
                            label="Tuned Models Folder",
                            placeholder="Folder containing fine-tuned checkpoints",
                        )
                        tuned_folder_button = gr.Button("📂", size="lg")

                    with gr.Row():
                        output_folder_input = gr.Textbox(
                            label="Output Folder",
                            info="Leave empty to write LoRAs next to tuned models.",
                            placeholder="Folder to save extracted LoRAs",
                        )
                        output_folder_button = gr.Button("📂", size="lg")

                    with gr.Row():
                        tuned_suffix_input = gr.Textbox(
                            label="Suffix to Remove from Tuned Names",
                            value="_finetuned",
                            placeholder="e.g., _finetuned",
                            info="Removed before naming the resulting LoRA file.",
                        )

                    with gr.Row():
                        output_suffix_input = gr.Textbox(
                            label="Output Filename Suffix",
                            value="_LoRA",
                            placeholder="e.g., _LoRA",
                            info="Appended to the tuned name (after removing suffix) for the saved LoRA file.",
                        )
                        output_extension_input = gr.Textbox(
                            label="Output Extension",
                            value=".safetensors",
                            placeholder=".safetensors",
                            info="Extension used for saved LoRA files.",
                        )

                    with gr.Row():
                        tuned_extensions_input = gr.Textbox(
                            label="Tuned File Extensions",
                            value=".safetensors,.pt,.bin",
                            placeholder=".safetensors,.pt",
                            info="Comma-separated list of extensions to scan in tuned folder.",
                        )

                    recursive_checkbox = gr.Checkbox(
                        label="Search Tuned Folder Recursively",
                        value=True,
                    )

                    overwrite_batch_checkbox = gr.Checkbox(
                        label="Overwrite Existing LoRAs",
                        value=False,
                        info="Enable to regenerate LoRAs even when output already exists.",
                    )

                    batch_status = gr.Textbox(
                        label="Batch Extraction Log",
                        lines=18,
                        max_lines=60,
                        interactive=False,
                    )

                    batch_button = gr.Button(
                        "🚀 Start Batch Extraction",
                        variant="primary",
                    )
                    cancel_batch_button = gr.Button(
                        "⛔ Cancel Batch",
                        variant="secondary",
                    )

        with gr.Column(scale=1):
            with gr.Accordion("Advanced Extraction Settings", open=True):
                dim_slider = gr.Slider(
                    label="LoRA Rank (dim)",
                    minimum=1,
                    maximum=512,
                    step=1,
                    value=128,
                    info="Higher rank captures more detail but increases LoRA size.",
                )
                clamp_slider = gr.Slider(
                    label="Clamp Quantile",
                    minimum=0.90,
                    maximum=1.0,
                    step=0.0005,
                    value=1.0,
                    info="Clamp outliers in SVD factors to stabilize weights. 1.0 disables clamping.",
                )
                device_radio = gr.Radio(
                    label="SVD Compute Device",
                    choices=["Auto (Prefer CUDA)", "CUDA", "CPU"],
                    value="Auto (Prefer CUDA)",
                    info="Auto selects CUDA when available, otherwise falls back to CPU.",
                )
                precision_radio = gr.Radio(
                    label="Save Precision",
                    choices=["FP16", "BF16", "Float32", "Auto (Preserve tensors)"],
                    value="BF16",
                    info="Target dtype for saved LoRA tensors. Auto keeps original precision.",
                )
                mem_eff_checkbox = gr.Checkbox(
                    label="Use Memory-Efficient SafeOpen",
                    value=True,
                    info="Enable memory-efficient safetensors reader to reduce RAM spikes.",
                )
                with gr.Row():
                    keep_metadata_checkbox = gr.Checkbox(
                        label="Preserve Metadata",
                        value=True,
                        info="Keep training metadata (title, timestamps) in the output LoRA file.",
                    )
                    overwrite_single_checkbox = gr.Checkbox(
                        label="Overwrite Existing LoRA",
                        value=False,
                        info="Enable to replace an existing LoRA file at the target location.",
                    )

    def start_single_extraction(*inputs):
        return extractor.extract_single(*inputs)

    def start_batch_extraction(*inputs):
        return extractor.batch_extract(*inputs)

    def request_single_cancel():
        return extractor.cancel_single()

    def request_batch_cancel():
        return extractor.cancel_batch()

    base_model_button.click(
        fn=lambda current: get_model_file_path(current),
        inputs=[base_model_path],
        outputs=[base_model_path],
        show_progress=False,
    )

    tuned_model_button.click(
        fn=lambda current: get_model_file_path(current),
        inputs=[tuned_model_path],
        outputs=[tuned_model_path],
        show_progress=False,
    )

    output_path_button.click(
        fn=lambda current: get_saveasfilename_path(
            current, extensions="*.safetensors", extension_name="LoRA files"
        ),
        inputs=[output_lora_path],
        outputs=[output_lora_path],
        show_progress=False,
    )

    extract_button.click(
        fn=start_single_extraction,
        inputs=[
            base_model_path,
            tuned_model_path,
            output_lora_path,
            dim_slider,
            clamp_slider,
            device_radio,
            precision_radio,
            mem_eff_checkbox,
            keep_metadata_checkbox,
            overwrite_single_checkbox,
        ],
        outputs=[single_status],
        show_progress=True,
    )
    cancel_single_button.click(
        fn=request_single_cancel,
        inputs=[],
        outputs=[single_status],
        show_progress=False,
    )

    batch_base_model_button.click(
        fn=lambda current: get_model_file_path(current),
        inputs=[batch_base_model_input],
        outputs=[batch_base_model_input],
        show_progress=False,
    )

    tuned_folder_button.click(
        fn=lambda current: get_folder_path(current),
        inputs=[tuned_folder_input],
        outputs=[tuned_folder_input],
        show_progress=False,
    )

    output_folder_button.click(
        fn=lambda current: get_folder_path(current),
        inputs=[output_folder_input],
        outputs=[output_folder_input],
        show_progress=False,
    )

    batch_button.click(
        fn=start_batch_extraction,
        inputs=[
            batch_base_model_input,
            tuned_folder_input,
            output_folder_input,
            tuned_suffix_input,
            output_suffix_input,
            tuned_extensions_input,
            output_extension_input,
            recursive_checkbox,
            dim_slider,
            clamp_slider,
            device_radio,
            precision_radio,
            mem_eff_checkbox,
            keep_metadata_checkbox,
            overwrite_batch_checkbox,
        ],
        outputs=[batch_status],
        show_progress=True,
    )
    cancel_batch_button.click(
        fn=request_batch_cancel,
        inputs=[],
        outputs=[batch_status],
        show_progress=False,
    )

