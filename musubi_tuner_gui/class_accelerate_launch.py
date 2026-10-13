import gradio as gr
import os
import re
import shlex

from .class_gui_config import GUIConfig
from .custom_logging import setup_logging

# Set up logging
log = setup_logging()


def normalize_gpu_ids(value) -> str:
    """accelerate's --gpu_ids: "0", "0,1". Spaces, ';' or a list / brackets as typed are accepted; anything that is not a
    GPU number is refused (accelerate would otherwise fail after the run started, or use other GPUs)."""
    if isinstance(value, (list, tuple)):
        value = ",".join(str(item) for item in value)
    text = str(value or "").strip().strip("'\"[](){}").strip()
    if not text:
        return ""
    parts = [part for part in re.split(r"[\s,;]+", text) if part]
    bad = [part for part in parts if not part.isdigit() or int(part) > 128]
    if bad:
        raise ValueError(f"GPU IDs must be GPU numbers separated by commas (for example 0 or 0,1), got: {value}")
    return ",".join(dict.fromkeys(str(int(part)) for part in parts))


def split_launch_args(value) -> list:
    """Extra accelerate launch arguments as separate list items (quotes group a value with spaces)."""
    text = str(value or "").strip()
    if not text:
        return []
    try:
        tokens = shlex.split(text, posix=os.name != "nt")  # Windows: backslashes in paths are not escapes
    except ValueError:
        tokens = text.split()
    cleaned = []
    for token in tokens:
        token = token.strip()
        if os.name == "nt" and len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'":
            token = token[1:-1]
        if token:
            cleaned.append(token)
    return cleaned


class AccelerateLaunch:
    def __init__(
        self,
        config: GUIConfig = {},
    ) -> None:
        self.config = config

        with gr.Accordion("Resource Selection", open=True):
            with gr.Row():
                self.mixed_precision = gr.Dropdown(
                    label="Mixed precision",
                    choices=["no", "fp16", "bf16"],
                    value=self.config.get("mixed_precision", "bf16"),
                    info="Use mixed precision for training. bf16 is generally recommended for FLUX/Z-Image on supported GPUs; fp16 may be less stable.",
                )
                self.num_processes = gr.Number(
                    label="Number of processes",
                    value=self.config.get("num_processes", 1),
                    # precision=0,
                    step=1,
                    minimum=1,
                    info="The total number of processes to be launched in parallel.",
                )
                self.num_machines = gr.Number(
                    label="Number of machines",
                    value=self.config.get("num_machines", 1),
                    # precision=0,
                    step=1,
                    minimum=1,
                    info="The total number of machines used in this training.",
                )
                self.num_cpu_threads_per_process = gr.Slider(
                    minimum=1,
                    maximum=os.cpu_count(),
                    step=1,
                    label="Number of CPU threads per core",
                    value=self.config.get(
                        "num_cpu_threads_per_process", 2
                    ),
                    info="CPU threads per process used by accelerate (data loading and preprocessing).",
                )
            with gr.Row():
                self.dynamo_backend = gr.Dropdown(
                    label="Dynamo backend",
                    choices=[
                        "no",
                        "eager",
                        "aot_eager",
                        "inductor",
                        "aot_ts_nvfuser",
                        "nvprims_nvfuser",
                        "cudagraphs",
                        "ofi",
                        "fx2trt",
                        "onnxrt",
                        "tensorrt",
                        "ipex",
                        "tvm",
                    ],
                    value=self.config.get("dynamo_backend", "no"),
                    info="Backend for torch dynamo JIT. Use 'no' to disable (recommended for stability), or 'inductor' for PyTorch 2.x optimizations.",
                )
                self.dynamo_mode = gr.Dropdown(
                    label="Dynamo mode",
                    choices=[
                        "",  # Empty = use default
                        "default",
                        "reduce-overhead",
                        "max-autotune",
                    ],
                    value=self.config.get("dynamo_mode", ""),
                    info="Choose a mode to optimize your training with dynamo. Leave empty for default.",
                )
                self.dynamo_use_fullgraph = gr.Checkbox(
                    label="Dynamo use fullgraph",
                    value=self.config.get(
                        "dynamo_use_fullgraph", False
                    ),
                    info="Whether to use full graph mode for dynamo or it is ok to break model into several subgraphs",
                )
                self.dynamo_use_dynamic = gr.Checkbox(
                    label="Dynamo use dynamic",
                    value=self.config.get(
                        "dynamo_use_dynamic", False
                    ),
                    info="Whether to enable dynamic shape tracing.",
                )

        with gr.Accordion("Hardware Selection & Distributed GPUs", open=True):
            with gr.Row():
                self.multi_gpu = gr.Checkbox(
                    label="Multi GPU",
                    value=self.config.get("multi_gpu", False),
                    info="Whether or not this should launch a distributed GPU training.",
                )
                self.gpu_ids = gr.Textbox(
                    label="GPU IDs",
                    value=self.config.get("gpu_ids", ""),
                    placeholder="example: 0,1",
                    info="Comma-separated GPU IDs to use on this machine (e.g., 0,1). Leave empty for auto.",
                )

                def validate_gpu_ids(value):
                    if value == "":
                        return
                    
                    # Handle both single GPU ID and comma-separated GPU IDs
                    gpu_ids = value.split(",")
                    for gpu_id in gpu_ids:
                        gpu_id = gpu_id.strip()  # Remove whitespace
                        if not gpu_id.isdigit() or int(gpu_id) < 0 or int(gpu_id) > 128:
                            log.error(
                                f"GPU ID '{gpu_id}' must be an integer between 0 and 128"
                            )
                            return

                self.gpu_ids.blur(fn=validate_gpu_ids, inputs=self.gpu_ids)

                self.main_process_port = gr.Number(
                    label="Main process port",
                    value=self.config.get("main_process_port", 0),
                    # precision=1,
                    step=1,
                    minimum=0,
                    maximum=65535,
                    info="Port of the main process for multi-GPU runs (0 = accelerate's default 29500).",
                )
        with gr.Row():
            self.extra_accelerate_launch_args = gr.Textbox(
                label="Extra accelerate launch arguments",
                value=self.config.get(
                    "extra_accelerate_launch_args", ""
                ),
                placeholder="example: --same_network --machine_rank 4",
                info="List of extra parameters to pass to accelerate launch",
            )

    def run_cmd(run_cmd: list, **kwargs):
        if "dynamo_backend" in kwargs and kwargs.get("dynamo_backend"):
            run_cmd.append("--dynamo_backend")
            run_cmd.append(kwargs["dynamo_backend"])

        if "dynamo_mode" in kwargs and kwargs.get("dynamo_mode"):
            run_cmd.append("--dynamo_mode")
            run_cmd.append(kwargs["dynamo_mode"])

        if "dynamo_use_fullgraph" in kwargs and kwargs.get("dynamo_use_fullgraph"):
            run_cmd.append("--dynamo_use_fullgraph")

        if "dynamo_use_dynamic" in kwargs and kwargs.get("dynamo_use_dynamic"):
            run_cmd.append("--dynamo_use_dynamic")

        # run_cmd is started as an argument list (no shell): every item reaches accelerate exactly as written, so
        # nothing may be shell-quoted here (a quoted '0, 1' or a quoted path would arrive with its quotes)
        run_cmd.extend(split_launch_args(kwargs.get("extra_accelerate_launch_args")))

        # Only pass gpu_ids if it's actually specified (not empty)
        gpu_ids = normalize_gpu_ids(kwargs.get("gpu_ids"))
        if gpu_ids:
            run_cmd.append("--gpu_ids")
            run_cmd.append(gpu_ids)

        main_process_port = int(float(kwargs.get("main_process_port") or 0))
        if main_process_port > 0:
            run_cmd.append("--main_process_port")
            run_cmd.append(str(main_process_port))

        if "mixed_precision" in kwargs and kwargs.get("mixed_precision"):
            run_cmd.append("--mixed_precision")
            run_cmd.append(str(kwargs["mixed_precision"]).strip())

        if "multi_gpu" in kwargs and kwargs.get("multi_gpu"):
            run_cmd.append("--multi_gpu")

        if int(kwargs.get("num_processes") or 0) > 0:
            run_cmd.append("--num_processes")
            run_cmd.append(str(int(kwargs["num_processes"])))

        if int(kwargs.get("num_machines") or 0) > 0:
            run_cmd.append("--num_machines")
            run_cmd.append(str(int(kwargs["num_machines"])))

        if (
            int(kwargs.get("num_cpu_threads_per_process") or 0) > 0
        ):
            run_cmd.append("--num_cpu_threads_per_process")
            run_cmd.append(str(int(kwargs["num_cpu_threads_per_process"])))

        return run_cmd
