import gradio as gr
import toml
from .class_gui_config import GUIConfig

class HuggingFace:
    def __init__(
        self,
        config: GUIConfig,
    ) -> None:
        self.config = config

        # Initialize the UI components
        self.initialize_ui_components()

    def initialize_ui_components(self) -> None:
        # --huggingface_repo_id HUGGINGFACE_REPO_ID
        #                         huggingface repo name to upload / huggingfaceにアップロードするリポジトリ名
        # --huggingface_repo_type HUGGINGFACE_REPO_TYPE
        #                         huggingface repo type to upload / huggingfaceにアップロードするリポジトリの種類
        # --huggingface_path_in_repo HUGGINGFACE_PATH_IN_REPO
        #                         huggingface model path to upload files / huggingfaceにアップロードするファイルのパス
        # --huggingface_token HUGGINGFACE_TOKEN
        #                         huggingface token / huggingfaceのトークン
        # --huggingface_repo_visibility HUGGINGFACE_REPO_VISIBILITY
        #                         huggingface repository visibility ('public' for public, 'private' or None for private) / huggingfaceにアップロードするリポジトリの公開設定（'public'で公開、'private'またはNoneで非公開）
        # --save_state_to_huggingface
        #                         save state to huggingface / huggingfaceにstateを保存する
        # --resume_from_huggingface
        #                         resume from huggingface (ex: --resume {repo_id}/{path_in_repo}:{revision}:{repo_type}) / huggingfaceから学習を再開する(例: --resume {repo_id}/{path_in_repo}:{revision}:{repo_type})
        # --async_upload        upload to huggingface asynchronously / huggingfaceに非同期でアップロードする
        with gr.Row():
            self.huggingface_repo_id = gr.Textbox(
                label="Huggingface repo id",
                placeholder="huggingface repo id",
                value=self.config.get("huggingface_repo_id", ""),
                info="Target repo id, e.g. username/model-name.",
            )

            self.huggingface_token = gr.Textbox(
                label="Huggingface token",
                placeholder="huggingface token",
                value=self.config.get("huggingface_token", ""),
                info="Access token with write permissions for the repo.",
            )

        with gr.Row():
            # Repository settings
            self.huggingface_repo_type = gr.Textbox(
                label="Huggingface repo type",
                placeholder="huggingface repo type",
                value=self.config.get("huggingface_repo_type", ""),
                info="Repo type (e.g., model or dataset). Leave empty for default.",
            )

            self.huggingface_repo_visibility = gr.Textbox(
                label="Huggingface repo visibility",
                placeholder="huggingface repo visibility",
                value=self.config.get("huggingface_repo_visibility", ""),
                info="Visibility setting: public or private (leave empty to keep repo default).",
            )

        with gr.Row():
            # File location in the repository
            self.huggingface_path_in_repo = gr.Textbox(
                label="Huggingface path in repo",
                placeholder="huggingface path in repo",
                value=self.config.get("huggingface_path_in_repo", ""),
                info="Optional subfolder path inside the repo to upload files to.",
            )

        with gr.Row():
            # Functions
            self.save_state_to_huggingface = gr.Checkbox(
                label="Save state to huggingface",
                value=self.config.get("save_state_to_huggingface", False),
                info="Upload training state/checkpoints to Hugging Face after save.",
            )

            self.resume_from_huggingface = gr.Textbox(
                label="Resume from huggingface",
                placeholder="resume from huggingface",
                value=self.config.get("resume_from_huggingface", ""),
                info="Resume string: {repo_id}/{path}:{revision}:{repo_type}.",
            )

            self.async_upload = gr.Checkbox(
                label="Async upload",
                value=self.config.get("async_upload", False),
                info="Upload to Hugging Face asynchronously to avoid blocking training.",
            )
