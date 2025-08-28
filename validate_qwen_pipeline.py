"""
Comprehensive validation script for Qwen Image LoRA training pipeline
Cross-checks our implementation against Musubi-tuner requirements
"""
import os
import sys
import toml
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class QwenPipelineValidator:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.info = []
        
    def validate_all(self, config_file: Optional[str] = None):
        """Run all validation checks"""
        print("=" * 80)
        print("QWEN IMAGE LORA PIPELINE VALIDATION")
        print("=" * 80)
        
        # If no config file specified, use a recently generated one
        if not config_file:
            test_configs = [
                "E:/SECourses_Improved_Trainer_v1/test_case/test1_20250828-124620.toml",
                "E:/SECourses_Improved_Trainer_v1/test_case/test1_20250828-124400.toml"
            ]
            for cfg in test_configs:
                if os.path.exists(cfg):
                    config_file = cfg
                    break
        
        if config_file and os.path.exists(config_file):
            print(f"\n[CONFIG] Using config file: {config_file}")
            config = self.load_config(config_file)
        else:
            print("\n⚠️ No config file found, checking defaults...")
            config = self.load_config("qwen_image_defaults.toml")
        
        # Run validation checks
        self.validate_required_parameters(config)
        self.validate_model_paths(config)
        self.validate_data_types(config)
        self.validate_network_configuration(config)
        self.validate_training_parameters(config)
        self.validate_optimizer_settings(config)
        self.validate_dataset_config(config)
        self.validate_caching_settings(config)
        self.validate_accelerate_settings(config)
        
        # Print results
        self.print_results()
        
    def load_config(self, config_file: str) -> dict:
        """Load TOML configuration file"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return toml.load(f)
        except Exception as e:
            self.errors.append(f"Failed to load config: {e}")
            return {}
    
    def validate_required_parameters(self, config: dict):
        """Check all required parameters for Qwen Image training"""
        print("\n🔍 Validating Required Parameters...")
        
        # Absolutely required parameters
        required = {
            'dit': 'Path to DiT (Qwen Image) model checkpoint',
            'vae': 'Path to VAE model checkpoint',
            'text_encoder': 'Path to Qwen2.5-VL text encoder',
            'dataset_config': 'Path to dataset configuration TOML',
            'output_dir': 'Output directory for trained models',
            'output_name': 'Name for the output model',
        }
        
        for param, description in required.items():
            if param not in config or not config[param]:
                self.errors.append(f"❌ Missing required parameter: {param} ({description})")
            else:
                self.info.append(f"✅ {param}: {config[param]}")
    
    def validate_model_paths(self, config: dict):
        """Validate model file paths and formats"""
        print("\n🔍 Validating Model Paths...")
        
        model_paths = {
            'dit': '.safetensors or directory',
            'vae': '.safetensors or directory',
            'text_encoder': '.safetensors or directory',
        }
        
        for param, expected_format in model_paths.items():
            if param in config and config[param]:
                path = config[param]
                if not os.path.exists(path):
                    self.warnings.append(f"⚠️ {param} path does not exist: {path}")
                else:
                    if os.path.isfile(path):
                        if not path.endswith('.safetensors'):
                            self.warnings.append(f"⚠️ {param} should be {expected_format}, got: {path}")
                        else:
                            self.info.append(f"✅ {param} file exists: {path}")
                    else:
                        self.info.append(f"✅ {param} directory exists: {path}")
    
    def validate_data_types(self, config: dict):
        """Validate data type settings for models"""
        print("\n🔍 Validating Data Types...")
        
        # Qwen Image specific requirements
        dtype_requirements = {
            'dit_dtype': ('bfloat16', 'MUST be bfloat16 for Qwen Image'),
            'vae_dtype': ('bfloat16', 'Recommended bfloat16 for Qwen Image VAE'),
            'text_encoder_dtype': (['float16', 'bfloat16'], 'float16 or bfloat16'),
        }
        
        for param, (expected, description) in dtype_requirements.items():
            if param in config:
                value = config[param]
                if isinstance(expected, list):
                    if value not in expected:
                        self.warnings.append(f"⚠️ {param} should be one of {expected} ({description}), got: {value}")
                    else:
                        self.info.append(f"✅ {param}: {value}")
                else:
                    if value != expected:
                        if param == 'dit_dtype':
                            self.errors.append(f"❌ {param} {description}, got: {value}")
                        else:
                            self.warnings.append(f"⚠️ {param} {description}, got: {value}")
                    else:
                        self.info.append(f"✅ {param}: {value}")
    
    def validate_network_configuration(self, config: dict):
        """Validate LoRA network configuration"""
        print("\n🔍 Validating Network Configuration...")
        
        # Check network module
        if config.get('network_module') != 'networks.lora_qwen_image':
            self.errors.append(f"❌ network_module must be 'networks.lora_qwen_image', got: {config.get('network_module')}")
        else:
            self.info.append(f"✅ network_module: {config.get('network_module')}")
        
        # Check network dimensions
        network_dim = config.get('network_dim', 0)
        network_alpha = config.get('network_alpha', 0)
        
        if network_dim < 1:
            self.errors.append(f"❌ network_dim must be >= 1, got: {network_dim}")
        else:
            self.info.append(f"✅ network_dim: {network_dim}")
        
        if network_alpha < 0.1:
            self.warnings.append(f"⚠️ network_alpha is very low ({network_alpha}), recommended: {network_dim}")
        else:
            self.info.append(f"✅ network_alpha: {network_alpha}")
        
        # Check network_args format
        network_args = config.get('network_args', [])
        if isinstance(network_args, str):
            self.errors.append(f"❌ network_args should be a list, got string: '{network_args}'")
        elif isinstance(network_args, list):
            self.info.append(f"✅ network_args is correctly formatted as list: {network_args}")
            # Validate each argument format
            for arg in network_args:
                if '=' not in arg:
                    self.warnings.append(f"⚠️ Invalid network_arg format (missing '='): {arg}")
    
    def validate_training_parameters(self, config: dict):
        """Validate training parameters"""
        print("\n🔍 Validating Training Parameters...")
        
        # Check timestep sampling
        if config.get('timestep_sampling') != 'qwen_shift':
            self.warnings.append(f"⚠️ timestep_sampling should be 'qwen_shift' for Qwen Image, got: {config.get('timestep_sampling')}")
        else:
            self.info.append(f"✅ timestep_sampling: qwen_shift")
        
        # Check discrete flow shift
        discrete_flow_shift = config.get('discrete_flow_shift', 0)
        if discrete_flow_shift < 2.0 or discrete_flow_shift > 2.5:
            self.warnings.append(f"⚠️ discrete_flow_shift recommended 2.2 for Qwen, got: {discrete_flow_shift}")
        else:
            self.info.append(f"✅ discrete_flow_shift: {discrete_flow_shift}")
        
        # Check gradient checkpointing (recommended for memory efficiency)
        if not config.get('gradient_checkpointing', False):
            self.warnings.append("⚠️ gradient_checkpointing is disabled, consider enabling to save VRAM")
        else:
            self.info.append("✅ gradient_checkpointing: enabled")
        
        # Check training steps/epochs
        max_train_steps = config.get('max_train_steps', 0)
        max_train_epochs = config.get('max_train_epochs', 0)
        
        if max_train_steps > 0:
            self.info.append(f"✅ max_train_steps: {max_train_steps}")
        elif max_train_epochs > 0:
            self.info.append(f"✅ max_train_epochs: {max_train_epochs}")
        else:
            self.errors.append("❌ Either max_train_steps or max_train_epochs must be set")
    
    def validate_optimizer_settings(self, config: dict):
        """Validate optimizer configuration"""
        print("\n🔍 Validating Optimizer Settings...")
        
        # Check optimizer type
        optimizer = config.get('optimizer_type', '')
        valid_optimizers = ['adamw', 'adamw8bit', 'adam', 'sgd', 'lion']
        if optimizer.lower() not in valid_optimizers:
            self.warnings.append(f"⚠️ Unusual optimizer: {optimizer}, common choices: {valid_optimizers}")
        else:
            self.info.append(f"✅ optimizer_type: {optimizer}")
        
        # Check learning rate
        lr = config.get('learning_rate', 0)
        if lr <= 0:
            self.errors.append(f"❌ learning_rate must be > 0, got: {lr}")
        elif lr > 1e-3:
            self.warnings.append(f"⚠️ learning_rate seems high ({lr}), typical range: 1e-6 to 1e-4")
        else:
            self.info.append(f"✅ learning_rate: {lr}")
        
        # Check optimizer args format
        optimizer_args = config.get('optimizer_args', [])
        if isinstance(optimizer_args, str):
            self.errors.append(f"❌ optimizer_args should be a list, got string: '{optimizer_args}'")
        elif isinstance(optimizer_args, list):
            self.info.append(f"✅ optimizer_args is correctly formatted as list")
    
    def validate_dataset_config(self, config: dict):
        """Validate dataset configuration"""
        print("\n🔍 Validating Dataset Configuration...")
        
        dataset_config = config.get('dataset_config', '')
        if dataset_config:
            if os.path.exists(dataset_config):
                self.info.append(f"✅ dataset_config exists: {dataset_config}")
                # Try to load and validate dataset config
                try:
                    with open(dataset_config, 'r') as f:
                        ds_config = toml.load(f)
                    
                    # Check for required dataset fields
                    if 'general' in ds_config:
                        general = ds_config['general']
                        if 'resolution' in general:
                            self.info.append(f"  ✅ Dataset resolution: {general['resolution']}")
                        else:
                            self.warnings.append("  ⚠️ No resolution specified in dataset config")
                    
                    if 'datasets' in ds_config:
                        num_datasets = len(ds_config['datasets'])
                        self.info.append(f"  ✅ Number of datasets: {num_datasets}")
                        
                        for i, ds in enumerate(ds_config['datasets']):
                            if 'image_directory' in ds:
                                img_dir = ds['image_directory']
                                if os.path.exists(img_dir):
                                    self.info.append(f"    ✅ Dataset {i}: {img_dir}")
                                else:
                                    self.warnings.append(f"    ⚠️ Dataset {i} directory not found: {img_dir}")
                except Exception as e:
                    self.warnings.append(f"⚠️ Could not parse dataset config: {e}")
            else:
                self.warnings.append(f"⚠️ dataset_config file not found: {dataset_config}")
    
    def validate_caching_settings(self, config: dict):
        """Validate caching configuration"""
        print("\n🔍 Validating Caching Settings...")
        
        # These settings are used during the caching phase
        cache_settings = {
            'caching_latent_skip_existing': 'Should be true to avoid re-caching',
            'caching_teo_skip_existing': 'Should be true to avoid re-caching',
            'caching_latent_batch_size': 'Batch size for latent caching',
            'caching_teo_batch_size': 'Batch size for text encoder caching',
        }
        
        for setting, description in cache_settings.items():
            if setting in config:
                value = config[setting]
                if 'skip_existing' in setting:
                    if value:
                        self.info.append(f"✅ {setting}: {value} (will skip existing cache)")
                    else:
                        self.warnings.append(f"⚠️ {setting}: {value} (will re-cache every time)")
                else:
                    self.info.append(f"✅ {setting}: {value}")
    
    def validate_accelerate_settings(self, config: dict):
        """Validate Accelerate launch settings"""
        print("\n🔍 Validating Accelerate Settings...")
        
        # Check mixed precision
        mixed_precision = config.get('mixed_precision', '')
        if mixed_precision != 'bf16':
            self.warnings.append(f"⚠️ mixed_precision should be 'bf16' for Qwen Image, got: {mixed_precision}")
        else:
            self.info.append(f"✅ mixed_precision: bf16")
        
        # Check dynamo backend
        dynamo_backend = config.get('dynamo_backend', 'no')
        if dynamo_backend not in ['no', 'inductor', 'eager']:
            self.warnings.append(f"⚠️ Unusual dynamo_backend: {dynamo_backend}")
        else:
            self.info.append(f"✅ dynamo_backend: {dynamo_backend}")
    
    def print_results(self):
        """Print validation results"""
        print("\n" + "=" * 80)
        print("VALIDATION RESULTS")
        print("=" * 80)
        
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"  {error}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  {warning}")
        
        if self.info:
            print(f"\n✅ VALIDATED ({len(self.info)} checks passed)")
            # Only show first 10 info messages to avoid clutter
            for i, info in enumerate(self.info[:10]):
                print(f"  {info}")
            if len(self.info) > 10:
                print(f"  ... and {len(self.info) - 10} more checks passed")
        
        print("\n" + "=" * 80)
        if self.errors:
            print("❌ PIPELINE VALIDATION FAILED - Fix errors before training")
        elif self.warnings:
            print("⚠️  PIPELINE VALIDATION PASSED WITH WARNINGS - Review warnings")
        else:
            print("✅ PIPELINE VALIDATION PASSED - Ready for training!")
        print("=" * 80)
        
        return len(self.errors) == 0

def check_musubi_compatibility():
    """Additional checks for Musubi-tuner specific requirements"""
    print("\n" + "=" * 80)
    print("MUSUBI-TUNER COMPATIBILITY CHECK")
    print("=" * 80)
    
    # Check if networks.lora_qwen_image module exists
    network_module_path = "musubi-tuner/src/networks/lora_qwen_image.py"
    if os.path.exists(network_module_path):
        print(f"✅ LoRA network module found: {network_module_path}")
    else:
        print(f"❌ LoRA network module not found: {network_module_path}")
    
    # Check for required Python packages
    required_packages = {
        'torch': 'PyTorch',
        'accelerate': 'Hugging Face Accelerate',
        'safetensors': 'SafeTensors',
        'toml': 'TOML parser',
        'einops': 'Einops for tensor operations',
        'transformers': 'Hugging Face Transformers',
    }
    
    print("\n📦 Required Packages:")
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"  ✅ {name} ({package})")
        except ImportError:
            print(f"  ❌ {name} ({package}) - Not installed")
    
    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  PyTorch version: {torch.__version__}")
        else:
            print("\n❌ CUDA not available - GPU training will not work")
    except Exception as e:
        print(f"\n❌ Could not check CUDA: {e}")

def main():
    """Main validation function"""
    import argparse
    parser = argparse.ArgumentParser(description="Validate Qwen Image LoRA training pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to config TOML file to validate")
    parser.add_argument("--check-musubi", action="store_true", help="Also check Musubi-tuner compatibility")
    
    args = parser.parse_args()
    
    # Run validation
    validator = QwenPipelineValidator()
    is_valid = validator.validate_all(args.config)
    
    # Check Musubi compatibility if requested
    if args.check_musubi:
        check_musubi_compatibility()
    
    # Return exit code based on validation
    sys.exit(0 if is_valid else 1)

if __name__ == "__main__":
    main()