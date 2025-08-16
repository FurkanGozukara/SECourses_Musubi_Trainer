import os
import toml
from .custom_logging import setup_logging
from .class_gui_config import GUIConfig

log = setup_logging()

class TabConfigManager:
    """Manages configuration loading based on active tabs with automatic defaults"""
    
    def __init__(self, config_file_path: str = "./config.toml"):
        self.config_file_path = config_file_path
        self.current_tab = None
        self.user_loaded_config = False
        self.configs = {
            "qwen_image": None,
            "musubi_tuner": None,
            "image_captioning": None
        }
        
        # Initialize base config
        self.base_config = GUIConfig(config_file_path)
        
        # Check if user loaded a config file
        self.user_loaded_config = self.base_config.is_config_loaded()
        
    def get_config_for_tab(self, tab_name: str) -> GUIConfig:
        """Get configuration for specific tab, loading defaults if needed"""
        if tab_name not in self.configs:
            log.error(f"Unknown tab: {tab_name}")
            return self.base_config
            
        # If user has loaded their own config, use it for all tabs
        if self.user_loaded_config:
            return self.base_config
            
        # If no user config and we haven't initialized this tab's config yet
        if self.configs[tab_name] is None:
            self.configs[tab_name] = self._load_tab_defaults(tab_name)
            
        return self.configs[tab_name]
    
    def _load_tab_defaults(self, tab_name: str) -> GUIConfig:
        """Load default configuration for specific tab"""
        default_files = {
            "qwen_image": "qwen_image_defaults.toml",
            "musubi_tuner": "musubi_tuner_defaults.toml",
            "image_captioning": "image_captioning_defaults.toml"
        }
        
        if tab_name not in default_files:
            log.warning(f"No default config file defined for tab: {tab_name}")
            return GUIConfig()  # Return empty config
            
        default_file = default_files[tab_name]
        default_path = os.path.join(os.path.dirname(__file__), "..", default_file)
        
        try:
            if os.path.exists(default_path):
                log.info(f"Loading {tab_name} defaults from {default_file}")
                config = GUIConfig()
                with open(default_path, "r", encoding="utf-8") as f:
                    default_data = toml.load(f)
                    config.config.update(default_data)
                log.info(f"{tab_name} defaults loaded successfully")
                return config
            else:
                log.warning(f"Default config file not found: {default_path}")
                return GUIConfig()  # Return empty config
        except Exception as e:
            log.error(f"Error loading {tab_name} defaults: {e}")
            return GUIConfig()  # Return empty config
    
    def set_user_config(self, config_file_path: str):
        """Set user-loaded configuration (overrides all defaults)"""
        try:
            self.base_config = GUIConfig(config_file_path)
            self.user_loaded_config = True
            # Clear tab-specific configs so they use the user config
            self.configs = {"qwen_image": None, "musubi_tuner": None, "image_captioning": None}
            log.info(f"User configuration loaded from {config_file_path}")
        except Exception as e:
            log.error(f"Error loading user config: {e}")
    
    def reset_to_defaults(self):
        """Reset to use default configurations for each tab"""
        self.user_loaded_config = False
        self.configs = {"qwen_image": None, "musubi_tuner": None, "image_captioning": None}
        self.base_config = GUIConfig()  # Empty config
        log.info("Reset to using default configurations for each tab")
    
    def is_using_defaults(self) -> bool:
        """Check if currently using default configurations"""
        return not self.user_loaded_config