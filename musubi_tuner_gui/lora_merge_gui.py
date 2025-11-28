import json
import os
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

import gradio as gr
import psutil

from .class_gui_config import GUIConfig
from .common_gui import (
    get_folder_path,
    get_model_file_path,
    get_saveasfilename_path,
    scriptdir,
    setup_environment,
    save_executed_script,
    generate_script_content,
)
from .custom_logging import setup_logging

PYTHON = sys.executable
MERGE_WORKER_SCRIPT = os.path.join(scriptdir, "musubi_tuner_gui", "lora_merge_worker.py")

log = setup_logging()

MERGE_MODE_LABELS = {
    "lora_to_lora": "LoRA ➜ LoRA (new adapter)",
    "lora_to_dit": "LoRA ➜ DiT (bake into base DiT)",
}
LABEL_TO_MODE = {label: key for key, label in MERGE_MODE_LABELS.items()}
DEFAULT_MERGE_MODE_LABEL = MERGE_MODE_LABELS["lora_to_lora"]


class LoRAMerger:
    def __init__(self, headless: bool, config: Optional[GUIConfig]) -> None:
        self.headless = headless
        self.config = config
        self.single_process: Optional[subprocess.Popen] = None
        self.batch_process: Optional[subprocess.Popen] = None
        self.single_cancel_requested = False
        self.batch_cancel_requested = False

    @staticmethod
    def _parse_multipliers(multiplier_inputs: List[float], count: int) -> List[float]:
        multipliers = []
        for value in multiplier_inputs:
            multipliers.append(1.0 if value is None else float(value))

        if not multipliers:
            multipliers = [1.0] * count

        if len(multipliers) < count:
            last = multipliers[-1] if multipliers else 1.0
            multipliers.extend([last] * (count - len(multipliers)))
        elif len(multipliers) > count:
            multipliers = multipliers[:count]

        return multipliers

    @staticmethod
    def _is_running(process: Optional[subprocess.Popen]) -> bool:
        return process is not None and process.poll() is None

    @staticmethod
    def _summarize_output(output: str) -> str:
        if not output:
            return " No worker output was captured."
        tail = "\n".join(output.strip().splitlines()[-8:])
        return f"\nWorker output:\n{tail}"

    @staticmethod
    def _terminate_process(process: Optional[subprocess.Popen]) -> bool:
        if not process or process.poll() is not None:
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

    @staticmethod
    def _ensure_worker_exists() -> None:
        if not os.path.isfile(MERGE_WORKER_SCRIPT):
            raise FileNotFoundError(
                f"LoRA merge worker script not found at '{MERGE_WORKER_SCRIPT}'. "
                "Please ensure the file exists."
            )

    @staticmethod
    def _write_payload(payload: Dict[str, object]) -> str:
        fd, path = tempfile.mkstemp(suffix=".json", prefix="lora_merge_payload_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
        except Exception:
            os.remove(path)
            raise
        return path

    @staticmethod
    def _read_result(path: str) -> Optional[Dict[str, str]]:
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None

    def _run_worker(
        self,
        mode: str,
        payload: Dict[str, object],
        process_attr: str,
    ) -> Tuple[str, str]:
        self._ensure_worker_exists()
        payload_path = self._write_payload(payload)
        result_fd, result_path = tempfile.mkstemp(suffix=".json", prefix="lora_merge_result_")
        os.close(result_fd)

        command = [
            PYTHON,
            MERGE_WORKER_SCRIPT,
            "--mode",
            mode,
            "--input",
            payload_path,
            "--output",
            result_path,
        ]

        log.info("Executing LoRA merge worker: %s", " ".join(command))
        
        # Save the merge command
        merge_script = generate_script_content(command, "LoRA merge")
        save_executed_script(
            script_content=merge_script,
            config_name=None,
            script_type="lora_merge"
        )
        
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=setup_environment(),
        )
        setattr(self, process_attr, process)
        output_lines: List[str] = []
        try:
            if process.stdout:
                for raw_line in process.stdout:
                    line = raw_line.rstrip("\r\n")
                    output_lines.append(line)
                    if line:
                        log.info("[LoRA Merge Worker] %s", line)
            process.wait()
            result = self._read_result(result_path)
            if result is None:
                summary = self._summarize_output("\n".join(output_lines))
                return "error", (
                    "LoRA merge worker did not produce a result file."
                    f"{summary}"
                )
            return result.get("status", "error"), result.get("message", "Worker returned no message.")
        finally:
            setattr(self, process_attr, None)
            for path in (payload_path, result_path):
                try:
                    os.remove(path)
                except OSError:
                    pass

    def merge_single(
        self,
        dit_path: str,
        lora1_path: str,
        lora1_mult: Optional[float],
        lora2_path: str,
        lora2_mult: Optional[float],
        lora3_path: str,
        lora3_mult: Optional[float],
        dit_in_channels: float,
        model_type: str,
        device_choice: str,
        output_path: str,
        overwrite_existing: bool,
        merge_target_choice: str,
        merged_lora_dtype_choice: str,
    ) -> str:
        if self._is_running(self.single_process):
            return "⚠️ A merge is already running. Please cancel it before starting another."

        lora_paths = [path for path in [lora1_path, lora2_path, lora3_path] if path]
        if not lora_paths:
            return "❌ Please provide at least one LoRA file to merge."

        if not output_path:
            return "❌ Output path must be specified."

        multipliers = self._parse_multipliers(
            [lora1_mult, lora2_mult, lora3_mult],
            len(lora_paths),
        )

        payload = {
            "merge_mode": LABEL_TO_MODE.get(merge_target_choice, "lora_to_lora"),
            "lora_paths": lora_paths,
            "multipliers": multipliers,
            "dit_path": dit_path,
            "output_path": output_path,
            "overwrite_existing": bool(overwrite_existing),
            "dit_in_channels": int(dit_in_channels),
            "model_type": model_type,
            "device_choice": device_choice,
            "merged_lora_dtype_choice": merged_lora_dtype_choice,
        }

        self.single_cancel_requested = False
        status, message = self._run_worker("single", payload, "single_process")
        was_cancelled = self.single_cancel_requested
        self.single_cancel_requested = False

        if was_cancelled:
            return "⛔ Merge cancelled."
        if status == "skip":
            return f"⊘ {message}"
        if status == "error":
            return f"❌ {message}"
        return message

    def batch_merge(
        self,
        dit_path: str,
        lora_folder: str,
        output_folder: str,
        output_suffix: str,
        output_extension: str,
        dit_in_channels: float,
        model_type: str,
        device_choice: str,
        multiplier: float,
        recursive: bool,
        overwrite_existing: bool,
        lora_extensions: str,
    ) -> str:
        if self._is_running(self.batch_process):
            return "⚠️ Batch merge is already running. Please cancel it before starting another."

        payload = {
            "dit_path": dit_path,
            "lora_folder": lora_folder,
            "output_folder": output_folder,
            "output_suffix": output_suffix,
            "output_extension": output_extension,
            "dit_in_channels": int(dit_in_channels),
            "model_type": model_type,
            "device_choice": device_choice,
            "multiplier": float(multiplier),
            "recursive": bool(recursive),
            "overwrite_existing": bool(overwrite_existing),
            "lora_extensions": lora_extensions,
        }

        self.batch_cancel_requested = False
        status, message = self._run_worker("batch", payload, "batch_process")
        was_cancelled = self.batch_cancel_requested
        self.batch_cancel_requested = False

        if was_cancelled:
            return "⛔ Merge cancelled."
        if status == "error":
            return f"❌ {message}"
        return message

    def cancel_single(self) -> str:
        if not self._is_running(self.single_process):
            return "⚠️ No single merge is currently running."
        self.single_cancel_requested = True
        if self._terminate_process(self.single_process):
            return "⛔ Cancellation requested. The merge process is stopping..."
        return "⚠️ Unable to cancel the merge. It may have already finished."

    def cancel_batch(self) -> str:
        if not self._is_running(self.batch_process):
            return "⚠️ No batch merge is currently running."
        self.batch_cancel_requested = True
        if self._terminate_process(self.batch_process):
            return "⛔ Cancellation requested. The batch merge is stopping..."
        return "⚠️ Unable to cancel the batch merge. It may have already finished."


def lora_merge_tab(headless: bool, config: Optional[GUIConfig]) -> None:
    merger = LoRAMerger(headless, config)

    gr.Markdown("# LoRA Merger")
    gr.Markdown(
        "Merge LoRA adapters into a base DiT checkpoint using Musubi Tuner's merge utilities. "
        "Supports single-shot merges with up to three LoRAs or batch merging an entire folder."
    )

    model_profiles: List[Dict[str, object]] = [
        {"label": "Qwen Image (16 channels)", "type": "qwen", "channels": 16},
        {"label": "Skyreels / DiT-I2V (32 channels)", "type": "hunyuan", "channels": 32},
        {"label": "Wan / VideoXL (48 channels)", "type": "hunyuan", "channels": 48},
        {"label": "Custom (manual channels)", "type": "custom", "channels": 16},
    ]
    profile_labels = [profile["label"] for profile in model_profiles]
    profile_map = {profile["label"]: profile for profile in model_profiles}
    default_profile = "Qwen Image (16 channels)"

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.Tab("Single Merge"):
                    with gr.Row():
                        merge_target_radio = gr.Radio(
                            label="Merge Target",
                            choices=list(MERGE_MODE_LABELS.values()),
                            value=DEFAULT_MERGE_MODE_LABEL,
                            info="Choose whether to combine LoRAs into another LoRA (default) or bake them into a DiT checkpoint.",
                        )
                        merged_lora_dtype_dropdown = gr.Dropdown(
                            label="Merged LoRA Dtype",
                            choices=[
                                "Auto",
                                "FP16",
                                "BF16",
                                "Float32",
                            ],
                            value="Float32",
                            info="Only used for LoRA ➜ LoRA merges. Forces all tensors to save in the selected dtype.",
                        )

                    with gr.Row():
                        dit_path_input = gr.Textbox(
                            label="Base DiT Checkpoint",
                            placeholder="Path to DiT .safetensors/.pt model",
                            info="Required only for LoRA ➜ DiT merges.",
                        )
                        dit_path_button = gr.Button("📄", size="lg")

                    with gr.Row():
                        lora1_input = gr.Textbox(
                            label="LoRA #1",
                            placeholder="First LoRA weights file",
                        )
                        lora1_button = gr.Button("📄", size="lg")
                        lora1_multiplier = gr.Number(
                            label="Multiplier #1",
                            value=1.0,
                            precision=3,
                        )

                    with gr.Row():
                        lora2_input = gr.Textbox(
                            label="LoRA #2 (optional)",
                            placeholder="Second LoRA weights file",
                        )
                        lora2_button = gr.Button("📄", size="lg")
                        lora2_multiplier = gr.Number(
                            label="Multiplier #2",
                            value=1.0,
                            precision=3,
                        )

                    with gr.Row():
                        lora3_input = gr.Textbox(
                            label="LoRA #3 (optional)",
                            placeholder="Third LoRA weights file",
                        )
                        lora3_button = gr.Button("📄", size="lg")
                        lora3_multiplier = gr.Number(
                            label="Multiplier #3",
                            value=1.0,
                            precision=3,
                        )

                    with gr.Row():
                        output_path_input = gr.Textbox(
                            label="Merged Output Path",
                            placeholder="Where to save merged checkpoint",
                        )
                        output_path_button = gr.Button("💾", size="lg")

                    overwrite_single_checkbox = gr.Checkbox(
                        label="Overwrite Existing Output",
                        value=False,
                    )

                    single_status = gr.Textbox(
                        label="Merge Log",
                        lines=14,
                        max_lines=40,
                        interactive=False,
                    )

                    merge_button = gr.Button(
                        "🔗 Merge LoRA(s)",
                        variant="primary",
                    )
                    cancel_single_button = gr.Button(
                        "⛔ Cancel Merge",
                        variant="secondary",
                    )

                with gr.Tab("Batch Merge"):
                    with gr.Row():
                        batch_dit_path_input = gr.Textbox(
                            label="Base DiT Checkpoint",
                            placeholder="Path to DiT .safetensors/.pt model",
                        )
                        batch_dit_button = gr.Button("📄", size="lg")

                    with gr.Row():
                        lora_folder_input = gr.Textbox(
                            label="LoRA Folder",
                            placeholder="Folder containing LoRA weights",
                        )
                        lora_folder_button = gr.Button("📂", size="lg")

                    with gr.Row():
                        output_folder_input = gr.Textbox(
                            label="Output Folder (optional)",
                            info="Leave empty to write merged outputs next to LoRA files.",
                            placeholder="Folder to save merged checkpoints",
                        )
                        output_folder_button = gr.Button("📂", size="lg")

                    with gr.Row():
                        output_suffix_input = gr.Textbox(
                            label="Output Suffix",
                            value="_merged",
                            placeholder="_merged",
                        )
                        output_extension_input = gr.Textbox(
                            label="Output Extension",
                            value=".safetensors",
                            placeholder=".safetensors",
                        )

                    with gr.Row():
                        multiplier_input = gr.Number(
                            label="LoRA Multiplier",
                            value=1.0,
                            precision=3,
                            info="Multiplier applied to every LoRA in the folder.",
                        )
                        lora_extensions_input = gr.Textbox(
                            label="LoRA Extensions",
                            value=".safetensors",
                            placeholder=".safetensors,.pt",
                        )
                        recursive_checkbox = gr.Checkbox(
                            label="Search Recursively",
                            value=True,
                        )

                    overwrite_batch_checkbox = gr.Checkbox(
                        label="Overwrite Existing Outputs",
                        value=False,
                    )

                    batch_status = gr.Textbox(
                        label="Batch Merge Log",
                        lines=18,
                        max_lines=60,
                        interactive=False,
                    )

                    batch_merge_button = gr.Button(
                        "🚀 Start Batch Merge",
                        variant="primary",
                    )
                    cancel_batch_button = gr.Button(
                        "⛔ Cancel Batch Merge",
                        variant="secondary",
                    )

        with gr.Column(scale=1):
            with gr.Accordion("Shared Settings", open=True):
                model_profile_dropdown = gr.Dropdown(
                    label="Model Type",
                    choices=profile_labels,
                    value=default_profile,
                    info="Select the base DiT architecture so we can set the correct input channels automatically. This is NOT the LoRA rank.",
                )
                dit_in_channels_value = gr.Number(
                    label="Custom DiT Input Channels",
                    value=profile_map[default_profile]["channels"],
                    precision=0,
                    interactive=True,
                    visible=False,
                    info="Only used when Model Type is set to Custom. This must match the in_channels value the DiT model was trained with.",
                )
                model_type_state = gr.State(value=profile_map[default_profile]["type"])
                device_radio = gr.Radio(
                    label="Merge Device",
                    choices=["Auto (Prefer CUDA)", "CUDA", "CPU"],
                    value="CPU",
                    info="Device used for applying LoRA weights when baking into a DiT.",
                )

    # Event bindings
    dit_path_button.click(
        fn=lambda current: get_model_file_path(current),
        inputs=[dit_path_input],
        outputs=[dit_path_input],
        show_progress=False,
    )

    for button, textbox in [
        (lora1_button, lora1_input),
        (lora2_button, lora2_input),
        (lora3_button, lora3_input),
    ]:
        button.click(
            fn=lambda current: get_model_file_path(current),
            inputs=[textbox],
            outputs=[textbox],
            show_progress=False,
        )

    output_path_button.click(
        fn=lambda current: get_saveasfilename_path(
            current, extensions="*.safetensors", extension_name="Checkpoint files"
        ),
        inputs=[output_path_input],
        outputs=[output_path_input],
        show_progress=False,
    )

    merge_button.click(
        fn=merger.merge_single,
        inputs=[
            dit_path_input,
            lora1_input,
            lora1_multiplier,
            lora2_input,
            lora2_multiplier,
            lora3_input,
            lora3_multiplier,
            dit_in_channels_value,
            model_type_state,
            device_radio,
            output_path_input,
            overwrite_single_checkbox,
            merge_target_radio,
            merged_lora_dtype_dropdown,
        ],
        outputs=[single_status],
        show_progress=True,
    )
    cancel_single_button.click(
        fn=merger.cancel_single,
        inputs=[],
        outputs=[single_status],
        show_progress=False,
    )

    batch_dit_button.click(
        fn=lambda current: get_model_file_path(current),
        inputs=[batch_dit_path_input],
        outputs=[batch_dit_path_input],
        show_progress=False,
    )

    lora_folder_button.click(
        fn=lambda current: get_folder_path(current),
        inputs=[lora_folder_input],
        outputs=[lora_folder_input],
        show_progress=False,
    )

    output_folder_button.click(
        fn=lambda current: get_folder_path(current),
        inputs=[output_folder_input],
        outputs=[output_folder_input],
        show_progress=False,
    )

    batch_merge_button.click(
        fn=merger.batch_merge,
        inputs=[
            batch_dit_path_input,
            lora_folder_input,
            output_folder_input,
            output_suffix_input,
            output_extension_input,
            dit_in_channels_value,
            model_type_state,
            device_radio,
            multiplier_input,
            recursive_checkbox,
            overwrite_batch_checkbox,
            lora_extensions_input,
        ],
        outputs=[batch_status],
        show_progress=True,
    )
    cancel_batch_button.click(
        fn=merger.cancel_batch,
        inputs=[],
        outputs=[batch_status],
        show_progress=False,
    )

    def update_dit_channels(selection, current_custom):
        profile = profile_map.get(selection, profile_map[default_profile])
        profile_type = profile["type"]
        is_custom = profile_type == "custom"
        new_value = current_custom if is_custom else profile["channels"]
        return (
            gr.Number(
                value=new_value,
                visible=is_custom,
                interactive=is_custom,
            ),
            gr.State(value=profile_type),
        )

    model_profile_dropdown.change(
        fn=update_dit_channels,
        inputs=[model_profile_dropdown, dit_in_channels_value],
        outputs=[dit_in_channels_value, model_type_state],
        show_progress=False,
    )

