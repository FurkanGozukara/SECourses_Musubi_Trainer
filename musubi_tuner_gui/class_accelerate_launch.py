import gradio as gr
import os
import shlex

from .class_gui_config import GUIConfig
from .custom_logging import setup_logging

# Set up logging
log = setup_logging()


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
                    info="✅ bf16 RECOMMENDED for Qwen Image. fp16 may cause instability. fp8 not supported in mixed_precision.",
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
                    info="The number of CPU threads per process.",
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
                    info="Backend for dynamo JIT compiler. no = disabled (recommended), inductor = PyTorch 2.0+ optimization",
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

        with gr.Accordion("Hardware Selection", open=True):
            with gr.Row():
                self.multi_gpu = gr.Checkbox(
                    label="Multi GPU",
                    value=self.config.get("multi_gpu", False),
                    info="Whether or not this should launch a distributed GPU training.",
                )
        with gr.Accordion("Distributed GPUs", open=True):
            with gr.Row():
                self.gpu_ids = gr.Textbox(
                    label="GPU IDs",
                    value=self.config.get("gpu_ids", ""),
                    placeholder="example: 0,1",
                    info=" What GPUs (by id) should be used for training on this machine as a comma-separated list",
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
                    info="The port to use to communicate with the machine of rank 0.",
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

        if (
            "extra_accelerate_launch_args" in kwargs
            and kwargs["extra_accelerate_launch_args"] != ""
        ):
            extra_accelerate_launch_args = kwargs[
                "extra_accelerate_launch_args"
            ].replace('"', "")
            for arg in extra_accelerate_launch_args.split():
                run_cmd.append(shlex.quote(arg))

        # Only pass gpu_ids if it's actually specified (not empty)
        if "gpu_ids" in kwargs and kwargs.get("gpu_ids", "").strip() != "":
            run_cmd.append("--gpu_ids")
            run_cmd.append(shlex.quote(kwargs["gpu_ids"]))

        if "main_process_port" in kwargs and kwargs.get("main_process_port", 0) > 0:
            run_cmd.append("--main_process_port")
            run_cmd.append(str(int(kwargs["main_process_port"])))

        if "mixed_precision" in kwargs and kwargs.get("mixed_precision"):
            run_cmd.append("--mixed_precision")
            run_cmd.append(shlex.quote(kwargs["mixed_precision"]))

        if "multi_gpu" in kwargs and kwargs.get("multi_gpu"):
            run_cmd.append("--multi_gpu")

        if "num_processes" in kwargs and int(kwargs.get("num_processes", 0)) > 0:
            run_cmd.append("--num_processes")
            run_cmd.append(str(int(kwargs["num_processes"])))

        if "num_machines" in kwargs and int(kwargs.get("num_machines", 0)) > 0:
            run_cmd.append("--num_machines")
            run_cmd.append(str(int(kwargs["num_machines"])))

        if (
            "num_cpu_threads_per_process" in kwargs
            and int(kwargs.get("num_cpu_threads_per_process", 0)) > 0
        ):
            run_cmd.append("--num_cpu_threads_per_process")
            run_cmd.append(str(int(kwargs["num_cpu_threads_per_process"])))

        return run_cmd
