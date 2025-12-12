import gradio as gr

from .class_gui_config import GUIConfig


class MetaData:
    def __init__(
        self,
        config: GUIConfig = {},
    ) -> None:
        self.config = config

        with gr.Row():
            self.metadata_title = gr.Textbox(
                label="Metadata title",
                placeholder="(optional) title for model metadata (default is output_name)",
                interactive=True,
                value=self.config.get("metadata_title", ""),
            )
            self.metadata_author = gr.Textbox(
                label="Metadata author",
                placeholder="(optional) author name for model metadata",
                interactive=True,
                value=self.config.get("metadata_author", ""),
            )
        self.metadata_description = gr.Textbox(
            label="Metadata description",
            placeholder="(optional) description for model metadata",
            interactive=True,
            value=self.config.get("metadata_description", ""),
        )
        with gr.Row():
            self.metadata_license = gr.Textbox(
                label="Metadata license",
                placeholder="(optional) license for model metadata",
                interactive=True,
                value=self.config.get("metadata_license", ""),
            )
            self.metadata_tags = gr.Textbox(
                label="Metadata tags",
                placeholder="(optional) tags for model metadata, separated by comma",
                interactive=True,
                value=self.config.get("metadata_tags", ""),
            )
        
        with gr.Row():
            self.metadata_reso = gr.Textbox(
                label="Metadata Resolution",
                placeholder="(optional) resolution for model metadata, e.g., 1024,1024 or 720,1280",
                interactive=True,
                value=self.config.get("metadata_reso", ""),
                info="Format: width,height (e.g., 1024,1024)"
            )
            self.metadata_arch = gr.Textbox(
                label="Metadata Architecture",
                placeholder="(optional) custom architecture string for model metadata",
                interactive=True,
                value=self.config.get("metadata_arch", ""),
                info="Custom architecture identifier"
            )

    def run_cmd(run_cmd: list, **kwargs):
        if "metadata_title" in kwargs and kwargs.get("metadata_title") != "":
            run_cmd.append("--metadata_title")
            run_cmd.append(kwargs["metadata_title"])

        if "metadata_author" in kwargs and kwargs.get("metadata_author") != "":
            run_cmd.append("--metadata_author")
            run_cmd.append(kwargs["metadata_author"])

        if "metadata_description" in kwargs and kwargs.get("metadata_description") != "":
            run_cmd.append("--metadata_description")
            run_cmd.append(kwargs["metadata_description"])

        if "metadata_license" in kwargs and kwargs.get("metadata_license") != "":
            run_cmd.append("--metadata_license")
            run_cmd.append(kwargs["metadata_license"])

        if "metadata_tags" in kwargs and kwargs.get("metadata_tags") != "":
            run_cmd.append("--metadata_tags")
            run_cmd.append(kwargs["metadata_tags"])

        if "metadata_reso" in kwargs:
            reso_value = kwargs.get("metadata_reso")
            # Handle None, empty strings, and validate format
            if reso_value and isinstance(reso_value, str):
                reso_value = reso_value.strip()
                if reso_value:
                    try:
                        # Validate format: should be comma-separated integers like "1024,1024"
                        parts = reso_value.split(",")
                        if len(parts) in [1, 2] and all(p.strip() for p in parts):
                            # Try to convert to integers to validate - will raise ValueError if invalid
                            validated_parts = [int(p.strip()) for p in parts]
                            # Ensure positive integers
                            if all(v > 0 for v in validated_parts):
                                run_cmd.append("--metadata_reso")
                                run_cmd.append(reso_value)
                    except (ValueError, AttributeError, TypeError):
                        # Invalid format, skip this parameter to let training use architecture-appropriate default
                        pass

        if "metadata_arch" in kwargs:
            arch_value = kwargs.get("metadata_arch")
            # Handle None, empty strings
            if arch_value and isinstance(arch_value, str):
                arch_value = arch_value.strip()
                if arch_value:
                    run_cmd.append("--metadata_arch")
                    run_cmd.append(arch_value)

        return run_cmd
