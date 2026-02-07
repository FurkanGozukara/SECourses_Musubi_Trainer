import glob
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gradio as gr
import psutil

from .class_gui_config import GUIConfig
from .common_gui import (
    generate_script_content,
    get_folder_path,
    get_model_file_path,
    save_executed_script,
    setup_environment,
)
from .custom_logging import setup_logging

log = setup_logging()

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MUSUBI_TUNER_ROOT = os.path.join(REPO_ROOT, "musubi-tuner")
MUSUBI_SRC_ROOT = os.path.join(MUSUBI_TUNER_ROOT, "src")

GENERIC_CONVERTER_SCRIPT = os.path.join(MUSUBI_TUNER_ROOT, "convert_lora.py")
HV15_CONVERTER_SCRIPT = os.path.join(
    MUSUBI_TUNER_ROOT, "src", "musubi_tuner", "networks", "convert_hunyuan_video_1_5_lora_to_comfy.py"
)
ZIMAGE_CONVERTER_SCRIPT = os.path.join(
    MUSUBI_TUNER_ROOT, "src", "musubi_tuner", "networks", "convert_z_image_lora_to_comfy.py"
)


@dataclass(frozen=True)
class ConversionProfile:
    key: str
    label: str
    script_path: str
    supported_models: str
    direction: str
    converter_type: str
    target: Optional[str] = None
    reverse: bool = False
    notes: str = ""


PROFILE_DEFINITIONS: List[ConversionProfile] = [
    ConversionProfile(
        key="generic_to_other",
        label="Musubi Default -> Other (Diffusers / ComfyUI)",
        script_path=GENERIC_CONVERTER_SCRIPT,
        supported_models="HunyuanVideo, FramePack, FLUX/FLUX Kontext/FLUX.2, Wan2.1, Z-Image, Qwen-Image",
        direction="Musubi default LoRA format to external format",
        converter_type="generic",
        target="other",
        notes="Uses musubi-tuner convert_lora.py --target other.",
    ),
    ConversionProfile(
        key="generic_to_default",
        label="Other (Diffusers / ComfyUI) -> Musubi Default",
        script_path=GENERIC_CONVERTER_SCRIPT,
        supported_models="HunyuanVideo, FramePack, FLUX/FLUX Kontext/FLUX.2, Wan2.1, Z-Image, Qwen-Image",
        direction="External format back to Musubi default LoRA format",
        converter_type="generic",
        target="default",
        notes="Uses musubi-tuner convert_lora.py --target default.",
    ),
    ConversionProfile(
        key="hv15_to_comfy",
        label="HunyuanVideo 1.5 -> ComfyUI",
        script_path=HV15_CONVERTER_SCRIPT,
        supported_models="HunyuanVideo 1.5",
        direction="HunyuanVideo 1.5 LoRA to ComfyUI format",
        converter_type="hv15",
        reverse=False,
        notes="Uses convert_hunyuan_video_1_5_lora_to_comfy.py.",
    ),
    ConversionProfile(
        key="comfy_to_hv15",
        label="ComfyUI -> HunyuanVideo 1.5",
        script_path=HV15_CONVERTER_SCRIPT,
        supported_models="HunyuanVideo 1.5",
        direction="ComfyUI-format LoRA back to HunyuanVideo 1.5 format",
        converter_type="hv15",
        reverse=True,
        notes="Uses convert_hunyuan_video_1_5_lora_to_comfy.py --reverse.",
    ),
    ConversionProfile(
        key="zimage_to_comfy",
        label="Z-Image -> ComfyUI (Dedicated Converter)",
        script_path=ZIMAGE_CONVERTER_SCRIPT,
        supported_models="Z-Image",
        direction="Z-Image LoRA to ComfyUI format",
        converter_type="zimage",
        reverse=False,
        notes="Uses convert_z_image_lora_to_comfy.py.",
    ),
    ConversionProfile(
        key="comfy_to_zimage",
        label="ComfyUI -> Z-Image (Dedicated Converter)",
        script_path=ZIMAGE_CONVERTER_SCRIPT,
        supported_models="Z-Image",
        direction="ComfyUI-format LoRA back to Z-Image format",
        converter_type="zimage",
        reverse=True,
        notes="Uses convert_z_image_lora_to_comfy.py --reverse.",
    ),
]

PROFILE_BY_LABEL: Dict[str, ConversionProfile] = {
    profile.label: profile for profile in PROFILE_DEFINITIONS
}
DEFAULT_PROFILE_LABEL = PROFILE_DEFINITIONS[0].label


def _normalize_extensions(ext_text: str, fallback: List[str]) -> List[str]:
    if not ext_text:
        return fallback
    result: List[str] = []
    for part in ext_text.split(","):
        ext = part.strip()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = "." + ext
        result.append(ext)
    return result or fallback


def _format_profile_markdown(profile_label: str) -> str:
    profile = PROFILE_BY_LABEL.get(profile_label, PROFILE_BY_LABEL[DEFAULT_PROFILE_LABEL])
    script_display = os.path.relpath(profile.script_path, REPO_ROOT).replace("\\", "/")
    return (
        f"**Script**: `{script_display}`\n\n"
        f"**Direction**: {profile.direction}\n\n"
        f"**Supported Models**: {profile.supported_models}\n\n"
        f"**Notes**: {profile.notes}"
    )


def _is_diffusers_prefix_supported(profile_label: str) -> bool:
    profile = PROFILE_BY_LABEL.get(profile_label, PROFILE_BY_LABEL[DEFAULT_PROFILE_LABEL])
    return profile.converter_type == "generic" and profile.target == "other"


class LoRAConverter:
    def __init__(self, headless: bool, config: Optional[GUIConfig]) -> None:
        self.headless = headless
        self.config = config
        self.single_process: Optional[subprocess.Popen] = None
        self.batch_process: Optional[subprocess.Popen] = None
        self.single_cancel_requested = False
        self.batch_cancel_requested = False

    @staticmethod
    def _is_running(process: Optional[subprocess.Popen]) -> bool:
        return process is not None and process.poll() is None

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
    def _tail_output(output: str, max_lines: int = 12) -> str:
        if not output:
            return ""
        lines = output.strip().splitlines()
        return "\n".join(lines[-max_lines:])

    def _resolve_python(self) -> str:
        venv_python = os.path.join(REPO_ROOT, "venv", "Scripts", "python.exe")
        if os.path.isfile(venv_python):
            return venv_python
        return sys.executable

    def _build_env(self) -> Dict[str, str]:
        env = setup_environment()
        existing = env.get("PYTHONPATH", "")
        paths = [p for p in existing.split(os.pathsep) if p]
        src_norm = os.path.normpath(MUSUBI_SRC_ROOT)
        if src_norm not in [os.path.normpath(p) for p in paths]:
            paths.insert(0, src_norm)
        env["PYTHONPATH"] = os.pathsep.join(paths)
        return env

    @staticmethod
    def _get_profile(profile_label: str) -> Optional[ConversionProfile]:
        return PROFILE_BY_LABEL.get(profile_label)

    @staticmethod
    def _ensure_script_exists(profile: ConversionProfile) -> Optional[str]:
        if os.path.isfile(profile.script_path):
            return None
        return f"Converter script not found: {profile.script_path}"

    def _build_command(
        self,
        profile: ConversionProfile,
        input_path: str,
        output_path: str,
        diffusers_prefix: str,
    ) -> List[str]:
        python_exec = self._resolve_python()

        if profile.converter_type == "generic":
            command = [
                python_exec,
                profile.script_path,
                "--input",
                input_path,
                "--output",
                output_path,
                "--target",
                str(profile.target),
            ]
            if profile.target == "other":
                prefix = (diffusers_prefix or "").strip()
                if prefix:
                    command.extend(["--diffusers_prefix", prefix])
            return command

        command = [python_exec, profile.script_path, input_path, output_path]
        if profile.reverse:
            command.append("--reverse")
        return command

    def _run_command(self, command: List[str], process_attr: str, run_label: str) -> Tuple[int, str]:
        log.info("Executing LoRA conversion command: %s", " ".join(command))
        script_content = generate_script_content(command, run_label)
        save_executed_script(
            script_content=script_content,
            config_name=None,
            script_type="lora_convert",
        )

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=self._build_env(),
            cwd=REPO_ROOT,
        )
        setattr(self, process_attr, process)
        captured_lines: List[str] = []
        try:
            if process.stdout:
                for raw_line in process.stdout:
                    line = raw_line.rstrip("\r\n")
                    captured_lines.append(line)
                    if line:
                        log.info("[LoRA Convert] %s", line)
            process.wait()
            return int(process.returncode or 0), "\n".join(captured_lines)
        finally:
            setattr(self, process_attr, None)

    @staticmethod
    def _normalize_output_filename(output_file_name: str) -> str:
        cleaned = (output_file_name or "").strip()
        if not cleaned:
            return ""
        if not os.path.splitext(cleaned)[1]:
            cleaned = f"{cleaned}.safetensors"
        return cleaned

    def convert_single(
        self,
        profile_label: str,
        input_path: str,
        output_folder: str,
        output_file_name: str,
        diffusers_prefix: str,
        overwrite_existing: bool,
    ) -> str:
        if self._is_running(self.single_process):
            return "A single conversion is already running. Cancel it before starting another."

        profile = self._get_profile(profile_label)
        if profile is None:
            return f"Unknown conversion profile: {profile_label}"

        script_error = self._ensure_script_exists(profile)
        if script_error:
            return script_error

        if not input_path or not os.path.isfile(input_path):
            return f"Input LoRA file not found: {input_path}"

        if not output_folder:
            return "Output folder is required."

        output_folder = os.path.abspath(output_folder)
        try:
            os.makedirs(output_folder, exist_ok=True)
        except OSError as exc:
            return f"Unable to create output folder '{output_folder}': {exc}"

        if not os.path.isdir(output_folder):
            return f"Output folder does not exist: {output_folder}"

        normalized_name = self._normalize_output_filename(output_file_name)
        if not normalized_name:
            return "Output file name is required."

        output_path = os.path.join(output_folder, normalized_name)
        if os.path.exists(output_path) and not overwrite_existing:
            return f"Skipped: Output already exists -> {output_path}"

        command = self._build_command(profile, input_path, output_path, diffusers_prefix)

        self.single_cancel_requested = False
        try:
            return_code, output = self._run_command(command, "single_process", "LoRA single conversion")
        except Exception as exc:
            return f"Failed to start conversion: {exc}"

        was_cancelled = self.single_cancel_requested
        self.single_cancel_requested = False
        if was_cancelled:
            return "Single conversion cancelled."

        if return_code == 0:
            return (
                "Single conversion complete.\n"
                f"- Profile: {profile.label}\n"
                f"- Input: {input_path}\n"
                f"- Output: {output_path}"
            )

        tail = self._tail_output(output)
        if tail:
            return f"Single conversion failed (exit code {return_code}).\n\nLast output:\n{tail}"
        return f"Single conversion failed (exit code {return_code})."

    def convert_batch(
        self,
        profile_label: str,
        input_folder: str,
        output_folder: str,
        extensions: str,
        recursive: bool,
        diffusers_prefix: str,
        overwrite_existing: bool,
    ) -> str:
        if self._is_running(self.batch_process):
            return "A batch conversion is already running. Cancel it before starting another."

        profile = self._get_profile(profile_label)
        if profile is None:
            return f"Unknown conversion profile: {profile_label}"

        script_error = self._ensure_script_exists(profile)
        if script_error:
            return script_error

        if not input_folder or not os.path.isdir(input_folder):
            return f"Input folder not found: {input_folder}"

        if not output_folder:
            return "Output folder is required."

        output_folder = os.path.abspath(output_folder)
        try:
            os.makedirs(output_folder, exist_ok=True)
        except OSError as exc:
            return f"Unable to create output folder '{output_folder}': {exc}"

        if not os.path.isdir(output_folder):
            return f"Output folder does not exist: {output_folder}"

        exts = _normalize_extensions(extensions, [".safetensors"])
        search_pattern = "**/*" if recursive else "*"
        input_files: List[str] = []
        for ext in exts:
            pattern = os.path.join(input_folder, f"{search_pattern}{ext}")
            input_files.extend(glob.glob(pattern, recursive=recursive))

        input_files = sorted(set(path for path in input_files if os.path.isfile(path)))
        if not input_files:
            return (
                "No LoRA files found for batch conversion.\n"
                f"- Folder: {input_folder}\n"
                f"- Extensions: {', '.join(exts)}"
            )

        success = 0
        skipped = 0
        failed = 0
        logs: List[str] = []

        self.batch_cancel_requested = False

        for input_path in input_files:
            if self.batch_cancel_requested:
                logs.append("Cancellation requested. Stopping batch conversion.")
                break

            rel_dir = os.path.relpath(os.path.dirname(input_path), input_folder)
            rel_dir = "" if rel_dir == "." else rel_dir
            output_subdir = os.path.join(output_folder, rel_dir)
            try:
                os.makedirs(output_subdir, exist_ok=True)
            except OSError as exc:
                failed += 1
                logs.append(f"FAIL {os.path.relpath(input_path, input_folder)} (create dir error: {exc})")
                continue

            # Keep the same file name as the source file in batch mode.
            output_path = os.path.join(output_subdir, os.path.basename(input_path))

            if os.path.exists(output_path) and not overwrite_existing:
                skipped += 1
                logs.append(f"SKIP {os.path.relpath(input_path, input_folder)} (already exists)")
                continue

            command = self._build_command(profile, input_path, output_path, diffusers_prefix)

            try:
                return_code, output = self._run_command(command, "batch_process", "LoRA batch conversion")
            except Exception as exc:
                failed += 1
                logs.append(f"FAIL {os.path.relpath(input_path, input_folder)} (start error: {exc})")
                continue

            if self.batch_cancel_requested:
                logs.append("Cancellation requested. Stopping batch conversion.")
                break

            rel_input = os.path.relpath(input_path, input_folder)
            rel_output = os.path.relpath(output_path, output_folder)
            if return_code == 0:
                success += 1
                logs.append(f"OK   {rel_input} -> {rel_output}")
            else:
                failed += 1
                tail = self._tail_output(output)
                if tail:
                    logs.append(f"FAIL {rel_input} (exit code {return_code})\n{tail}")
                else:
                    logs.append(f"FAIL {rel_input} (exit code {return_code})")

        was_cancelled = self.batch_cancel_requested
        self.batch_cancel_requested = False

        summary_header = "Batch conversion cancelled." if was_cancelled else "Batch conversion complete."
        return (
            f"{summary_header}\n"
            f"- Profile: {profile.label}\n"
            f"- Input folder: {input_folder}\n"
            f"- Output folder: {output_folder}\n"
            f"- Success: {success}\n"
            f"- Skipped: {skipped}\n"
            f"- Failed: {failed}\n\n"
            + "\n".join(logs)
        )

    def cancel_single(self) -> str:
        if not self._is_running(self.single_process):
            return "No single conversion is currently running."
        self.single_cancel_requested = True
        if self._terminate_process(self.single_process):
            return "Cancellation requested. Single conversion is stopping."
        return "Unable to cancel single conversion. It may have already finished."

    def cancel_batch(self) -> str:
        if not self._is_running(self.batch_process):
            return "No batch conversion is currently running."
        self.batch_cancel_requested = True
        if self._terminate_process(self.batch_process):
            return "Cancellation requested. Batch conversion is stopping."
        return "Unable to cancel batch conversion. It may have already finished."


def lora_convert_tab(headless: bool, config: Optional[GUIConfig]) -> None:
    converter = LoRAConverter(headless, config)

    gr.Markdown("# LoRA Converter")
    gr.Markdown(
        "Convert LoRA files with all conversion paths currently provided by musubi-tuner. "
        "Includes generic Musubi <-> external conversion plus dedicated HunyuanVideo 1.5 and Z-Image converters."
    )

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.Tab("Single Convert"):
                    single_profile = gr.Dropdown(
                        label="Conversion Profile",
                        choices=[profile.label for profile in PROFILE_DEFINITIONS],
                        value=DEFAULT_PROFILE_LABEL,
                        info="Select the exact converter route and model family.",
                    )

                    single_profile_info = gr.Markdown(
                        value=_format_profile_markdown(DEFAULT_PROFILE_LABEL)
                    )

                    with gr.Row():
                        single_input_path = gr.Textbox(
                            label="Input LoRA File",
                            placeholder="Path to source LoRA (.safetensors/.pt/.ckpt/.pth)",
                        )
                        single_input_button = gr.Button("Browse File", size="lg")

                    with gr.Row():
                        single_output_folder = gr.Textbox(
                            label="Output Folder",
                            placeholder="Folder where converted LoRA will be saved",
                        )
                        single_output_folder_button = gr.Button("Browse Folder", size="lg")

                    single_output_name = gr.Textbox(
                        label="Output File Name",
                        value="converted_lora.safetensors",
                        placeholder="example_converted.safetensors",
                        info="You provide the file name for single conversion.",
                    )

                    single_diffusers_prefix = gr.Textbox(
                        label="Diffusers Prefix (optional)",
                        value="",
                        placeholder="diffusion_model or transformer",
                        info="Only used by 'Musubi Default -> Other'. Leave empty to use converter default.",
                        visible=True,
                    )

                    single_overwrite = gr.Checkbox(
                        label="Overwrite Existing Output",
                        value=False,
                    )

                    single_status = gr.Textbox(
                        label="Single Conversion Log",
                        lines=14,
                        max_lines=50,
                        interactive=False,
                    )

                    with gr.Row():
                        run_single_button = gr.Button(
                            "Convert Single LoRA",
                            variant="primary",
                        )
                        cancel_single_button = gr.Button(
                            "Cancel Single Conversion",
                            variant="secondary",
                        )

                with gr.Tab("Batch Convert"):
                    batch_profile = gr.Dropdown(
                        label="Conversion Profile",
                        choices=[profile.label for profile in PROFILE_DEFINITIONS],
                        value=DEFAULT_PROFILE_LABEL,
                        info="Select the exact converter route and model family.",
                    )

                    batch_profile_info = gr.Markdown(
                        value=_format_profile_markdown(DEFAULT_PROFILE_LABEL)
                    )

                    with gr.Row():
                        batch_input_folder = gr.Textbox(
                            label="Batch Input Folder",
                            placeholder="Folder containing source LoRA files",
                        )
                        batch_input_folder_button = gr.Button("Browse Folder", size="lg")

                    with gr.Row():
                        batch_output_folder = gr.Textbox(
                            label="Batch Output Folder",
                            placeholder="Folder to save converted LoRA files",
                            info="Batch output keeps the same file names as input files.",
                        )
                        batch_output_folder_button = gr.Button("Browse Folder", size="lg")

                    with gr.Row():
                        batch_extensions = gr.Textbox(
                            label="Input Extensions",
                            value=".safetensors,.pt,.ckpt,.pth",
                            placeholder=".safetensors,.pt",
                        )
                        batch_recursive = gr.Checkbox(
                            label="Search Recursively",
                            value=True,
                        )

                    batch_diffusers_prefix = gr.Textbox(
                        label="Diffusers Prefix (optional)",
                        value="",
                        placeholder="diffusion_model or transformer",
                        info="Only used by 'Musubi Default -> Other'.",
                        visible=True,
                    )

                    batch_overwrite = gr.Checkbox(
                        label="Overwrite Existing Outputs",
                        value=False,
                    )

                    batch_status = gr.Textbox(
                        label="Batch Conversion Log",
                        lines=18,
                        max_lines=70,
                        interactive=False,
                    )

                    with gr.Row():
                        run_batch_button = gr.Button(
                            "Start Batch Conversion",
                            variant="primary",
                        )
                        cancel_batch_button = gr.Button(
                            "Cancel Batch Conversion",
                            variant="secondary",
                        )

        with gr.Column(scale=1):
            gr.Markdown(
                "### Supported Converters\n"
                "- Generic `convert_lora.py` (Musubi default <-> other)\n"
                "- `convert_hunyuan_video_1_5_lora_to_comfy.py`\n"
                "- `convert_z_image_lora_to_comfy.py`\n\n"
                "Batch mode validates input/output folders and preserves source file names in the output folder."
            )

    def _update_single_profile(profile_label: str, current_prefix: str):
        return (
            _format_profile_markdown(profile_label),
            gr.Textbox(visible=_is_diffusers_prefix_supported(profile_label), value=current_prefix),
        )

    def _update_batch_profile(profile_label: str, current_prefix: str):
        return (
            _format_profile_markdown(profile_label),
            gr.Textbox(visible=_is_diffusers_prefix_supported(profile_label), value=current_prefix),
        )

    single_profile.change(
        fn=_update_single_profile,
        inputs=[single_profile, single_diffusers_prefix],
        outputs=[single_profile_info, single_diffusers_prefix],
        show_progress=False,
    )

    batch_profile.change(
        fn=_update_batch_profile,
        inputs=[batch_profile, batch_diffusers_prefix],
        outputs=[batch_profile_info, batch_diffusers_prefix],
        show_progress=False,
    )

    single_input_button.click(
        fn=lambda current: get_model_file_path(current),
        inputs=[single_input_path],
        outputs=[single_input_path],
        show_progress=False,
    )

    single_output_folder_button.click(
        fn=lambda current: get_folder_path(current),
        inputs=[single_output_folder],
        outputs=[single_output_folder],
        show_progress=False,
    )

    batch_input_folder_button.click(
        fn=lambda current: get_folder_path(current),
        inputs=[batch_input_folder],
        outputs=[batch_input_folder],
        show_progress=False,
    )

    batch_output_folder_button.click(
        fn=lambda current: get_folder_path(current),
        inputs=[batch_output_folder],
        outputs=[batch_output_folder],
        show_progress=False,
    )

    run_single_button.click(
        fn=converter.convert_single,
        inputs=[
            single_profile,
            single_input_path,
            single_output_folder,
            single_output_name,
            single_diffusers_prefix,
            single_overwrite,
        ],
        outputs=[single_status],
        show_progress=True,
    )

    cancel_single_button.click(
        fn=converter.cancel_single,
        inputs=[],
        outputs=[single_status],
        show_progress=False,
    )

    run_batch_button.click(
        fn=converter.convert_batch,
        inputs=[
            batch_profile,
            batch_input_folder,
            batch_output_folder,
            batch_extensions,
            batch_recursive,
            batch_diffusers_prefix,
            batch_overwrite,
        ],
        outputs=[batch_status],
        show_progress=True,
    )

    cancel_batch_button.click(
        fn=converter.cancel_batch,
        inputs=[],
        outputs=[batch_status],
        show_progress=False,
    )

