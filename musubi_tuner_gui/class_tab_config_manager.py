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
            "wan": None,
            "musubi_tuner": None,
            "image_captioning": None,
            "fp8_converter": None,
            "image_preprocessing": None
        }
        
        # Check if config file exists and has content
        import os
        if os.path.exists(config_file_path):
            # Initialize base config
            self.base_config = GUIConfig(config_file_path)
            # Check if user loaded a config file
            self.user_loaded_config = self.base_config.is_config_loaded()
        else:
            # No config file exists, use empty config
            self.base_config = GUIConfig()
            self.user_loaded_config = False
        
    def get_config_for_tab(self, tab_name: str) -> GUIConfig:
        """Get configuration for specific tab, loading defaults if needed"""
        if tab_name not in self.configs:
            log.error(f"Unknown tab: {tab_name}")
            return self.base_config
            
        # Special handling for image_captioning: check if config has this section
        if tab_name == "image_captioning":
            # Check if the base config has image_captioning section
            if self.user_loaded_config and "image_captioning" in self.base_config.config:
                # User has image_captioning config, use it
                return self.base_config
            else:
                # No image_captioning config, load defaults
                if self.configs[tab_name] is None:
                    self.configs[tab_name] = self._load_tab_defaults(tab_name)
                return self.configs[tab_name]
        
        # For other tabs, check if user loaded a custom config or if it's just the default startup config
        if self.user_loaded_config and not self.config_file_path.endswith(("qwen_image_defaults.toml", "wan_defaults.toml", "musubi_tuner_defaults.toml")):
            # User loaded a truly custom config, use it for all tabs
            return self.base_config
            
        # If no user config or using default configs, load tab-specific defaults
        if self.configs[tab_name] is None:
            self.configs[tab_name] = self._load_tab_defaults(tab_name)
            
        return self.configs[tab_name]
    
    def _load_tab_defaults(self, tab_name: str) -> GUIConfig:
        """Load default configuration for specific tab"""
        default_files = {
            "qwen_image": "qwen_image_defaults.toml",
            "wan": "wan_defaults.toml",
            "musubi_tuner": "musubi_tuner_defaults.toml",
            "image_captioning": "image_captioning_defaults.toml",
            "fp8_converter": "fp8_converter_defaults.toml",
            "image_preprocessing": None  # No defaults file needed for this tab
        }
        
        if tab_name not in default_files:
            log.warning(f"No default config file defined for tab: {tab_name}")
            return GUIConfig()  # Return empty config
            
        default_file = default_files[tab_name]
        
        # If no default file is specified (None), return empty config
        if default_file is None:
            return GUIConfig()  # Return empty config
            
        default_path = os.path.join(os.path.dirname(__file__), "..", default_file)
        
        try:
            if os.path.exists(default_path):
                log.info(f"Loading {tab_name} defaults from {default_file}")
                # Create a GUIConfig with the default file path
                config = GUIConfig(default_path)
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
            self.configs = {"qwen_image": None, "wan": None, "musubi_tuner": None, "image_captioning": None, "fp8_converter": None, "image_preprocessing": None}
            log.info(f"User configuration loaded from {config_file_path}")
        except Exception as e:
            log.error(f"Error loading user config: {e}")
    
    def reset_to_defaults(self):
        """Reset to use default configurations for each tab"""
        self.user_loaded_config = False
        self.configs = {"qwen_image": None, "wan": None, "musubi_tuner": None, "image_captioning": None, "fp8_converter": None, "image_preprocessing": None}
        self.base_config = GUIConfig()  # Empty config
        log.info("Reset to using default configurations for each tab")
    
    def is_using_defaults(self) -> bool:
        """Check if currently using default configurations"""
        return not self.user_loaded_config