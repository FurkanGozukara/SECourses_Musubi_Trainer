import toml
from .common_gui import scriptdir
from .custom_logging import setup_logging

# Set up logging
log = setup_logging()


class GUIConfig:
    """
    A class to handle the configuration for the Kohya SS GUI.
    """

    def __init__(self, config_file_path: str = "./config.toml"):
        """
        Initialize the KohyaSSGUIConfig class.
        """
        self.config = self.load_config(config_file_path=config_file_path)

    def load_config(self, config_file_path: str = "./config.toml") -> dict:
        """
        Loads the Kohya SS GUI configuration from a TOML file.

        Returns:
        dict: The configuration data loaded from the TOML file.
        """
        try:
            # Attempt to load the TOML configuration file from the specified directory.
            config = toml.load(f"{config_file_path}")
            log.debug(f"Loaded configuration from {config_file_path}")
        except FileNotFoundError:
            # If the config file is not found, initialize `config` as an empty dictionary to handle missing configurations gracefully.
            config = {}
            log.debug(
                f"No configuration file found at {config_file_path}. Initializing empty configuration."
            )

        return config

    def save_config(self, config: dict, config_file_path: str = "./config.toml"):
        """
        Saves the Kohya SS GUI configuration to a TOML file.

        Parameters:
        - config (dict): The configuration data to save.
        """
        # Write the configuration data to the TOML file
        with open(f"{config_file_path}", "w", encoding="utf-8") as f:
            toml.dump(config, f)

    def get(self, key: str, default=None):
        """
        Retrieves the value of a specified key from the configuration data.

        Parameters:
        - key (str): The key to retrieve the value for.
        - default: The default value to return if the key is not found.

        Returns:
        The value associated with the key, or the default value if the key is not found.
        """
        # Split the key into a list of keys if it contains a dot (.)
        keys = key.split(".")
        # Initialize `data` with the entire configuration data
        data = self.config

        # Iterate over the keys to access nested values
        for k in keys:
            log.debug(k)
            # If the key is not found in the current data, return the default value
            if k not in data:
                log.debug(
                    f"Key '{key}' not found in configuration. Returning default value."
                )
                return default

            # Update `data` to the value associated with the current key
            data = data.get(k)

        # Apply minimum constraints and optional parameter handling
        if data is not None:
            data = self._apply_constraints(key, data, default)

        # Return the final value
        log.debug(f"Returned {data}")
        return data

    def _apply_constraints(self, key: str, value, default):
        """
        Apply minimum constraints and handle optional parameters.
        
        Parameters:
        - key (str): The parameter key
        - value: The loaded value  
        - default: The default value for fallback
        
        Returns:
        The constrained value
        """
        # Define minimum value constraints to prevent validation errors
        minimum_constraints = {
            # Accelerate Launch components
            "num_processes": 1,
            "num_machines": 1,
            "num_cpu_threads_per_process": 1,
            "main_process_port": 0,
            # Components with minimum=0
            "vae_chunk_size": 0,
            "vae_spatial_tile_sample_min_size": 0,
            "blocks_to_swap": 0,
            "min_timestep": 0,
            "max_data_loader_n_workers": 0,
            "seed": 0,
            "max_grad_norm": 0.0,
            "lr_warmup_steps": 0,
            # Components with minimum=0.1
            "guidance_scale": 0.1,
            "logit_std": 0.1,
            "mode_scale": 0.1,
            "sigmoid_scale": 0.1,
            "lr_scheduler_power": 0.1,
            "network_alpha": 0.1,
            # Components with minimum=1
            "max_timestep": 1,
            "max_train_epochs": 1,
            "gradient_accumulation_steps": 1,
            "lr_scheduler_num_cycles": 1,
            "network_dim": 1,
            "caching_latent_batch_size": 1,
            "caching_teo_batch_size": 1,
            # Components changed to minimum=0 (can be disabled)
            "sample_every_n_steps": 0,
            "sample_every_n_epochs": 0,
            "ddp_timeout": 0,
            "save_every_n_steps": 0,
            "save_last_n_epochs": 0,
            # Components with minimum=100
            "max_train_steps": 100,
            # Components with minimum=1e-7
            "learning_rate": 1e-7,
            # Components with minimum=-10.0
            "logit_mean": -10.0
        }
        
        # Parameters that should be None when their value is 0 (optional parameters)
        # Removed parameters that now accept 0 as a valid disabled state
        optional_parameters = {
            "max_timestep", "min_timestep"
        }
        
        # Convert 0 to None for optional parameters to avoid minimum constraint violations
        if key in optional_parameters and value == 0:
            log.debug(f"Converting optional parameter '{key}' value 0 to None")
            return None
        elif key in minimum_constraints:
            min_val = minimum_constraints[key]
            if value < min_val:
                log.warning(f"Parameter '{key}' value {value} is below minimum {min_val}, adjusting to minimum")
                return min_val
        
        return value

    def is_config_loaded(self) -> bool:
        """
        Checks if the configuration was loaded from a file.

        Returns:
        bool: True if the configuration was loaded from a file, False otherwise.
        """
        is_loaded = self.config != {}
        log.debug(f"Configuration was loaded from file: {is_loaded}")
        return is_loaded
