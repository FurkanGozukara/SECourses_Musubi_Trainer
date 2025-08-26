#!/usr/bin/env python3
"""
Test script to verify TOML generation accuracy against musubi tuner parameters.
This script simulates the GUI parameter generation and validates output.
"""

import sys
import os
import toml

# Add the GUI module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'musubi_tuner_gui'))

from common_gui import SaveConfigFileToRun

def test_toml_generation():
    """Test TOML generation with various parameter types and edge cases"""
    
    # Simulate GUI parameters including edge cases (0, False, empty strings, None)
    test_parameters = [
        # Required model paths
        ("dit", "/path/to/dit.safetensors"),
        ("vae", "/path/to/vae.safetensors"),
        ("text_encoder", "/path/to/text_encoder.safetensors"),
        ("dataset_config", "/path/to/dataset.toml"),
        ("output_dir", "/path/to/output"),
        ("output_name", "test_lora"),
        
        # Hardcoded Qwen Image values
        ("dit_dtype", "bfloat16"),
        
        # Test 0 values (should be preserved)
        ("lr_warmup_steps", 0),
        ("lr_decay_steps", 0),
        ("blocks_to_swap", 0),
        ("save_every_n_steps", 0),
        ("save_last_n_epochs", 0),
        ("min_timestep", 0),
        ("max_timestep", 0),
        ("seed", 0),  # Valid - means random seed
        
        # Test positive numeric values  
        ("learning_rate", 1e-4),
        ("max_train_steps", 1600),
        ("max_train_epochs", 16),
        ("network_dim", 32),
        ("network_alpha", 1.0),
        ("max_data_loader_n_workers", 2),
        ("gradient_accumulation_steps", 1),
        ("discrete_flow_shift", 3.0),
        ("guidance_scale", 1.0),
        ("caching_latent_batch_size", 4),
        ("caching_teo_batch_size", 16),
        
        # Test False boolean values (should be preserved)
        ("fp8_base", False),
        ("fp8_scaled", False),
        ("flash_attn", False),
        ("sage_attn", False),
        ("xformers", False),
        ("flash3", False),
        ("split_attn", False),
        ("save_state", False),
        ("no_metadata", False),
        ("vae_tiling", False),
        ("img_in_txt_in_offloading", False),
        ("preserve_distribution_shape", False),
        
        # Test True boolean values
        ("fp8_vl", True),
        ("sdpa", True),
        ("gradient_checkpointing", True),
        ("persistent_data_loader_workers", True),
        ("caching_latent_skip_existing", True),
        ("caching_latent_keep_cache", True),
        ("caching_teo_skip_existing", True),
        ("caching_teo_keep_cache", True),
        
        # Test empty strings (should be preserved for optional fields)
        ("network_weights", ""),
        ("base_weights", ""),
        ("resume", ""),
        ("logging_dir", ""),
        ("log_with", ""),
        ("log_prefix", ""),
        ("wandb_api_key", ""),
        ("network_args", ""),
        ("additional_parameters", ""),
        ("sample_prompts", ""),
        ("metadata_author", ""),
        ("metadata_description", ""),
        ("huggingface_repo_id", ""),
        
        # Test string values
        ("timestep_sampling", "shift"),
        ("weighting_scheme", "none"),
        ("optimizer_type", "adamw8bit"),
        ("lr_scheduler", "constant"),
        ("network_module", "networks.lora_qwen_image"),
        ("text_encoder_dtype", "float16"),
        ("vae_dtype", "bfloat16"),
        ("mixed_precision", "bf16"),
        ("dynamo_backend", "no"),
        ("caching_latent_device", "cuda"),
        ("caching_teo_device", "cuda"),
        
        # Test float values
        ("network_dropout", 0.0),  # Should be preserved
        ("scale_weight_norms", 0.0),  # Should be preserved  
        ("max_grad_norm", 1.0),
        ("logit_mean", 0.0),  # Should be preserved
        ("logit_std", 1.0),
        ("mode_scale", 1.29),
        ("sigmoid_scale", 1.0),
        ("base_weights_multiplier", 1.0),
        ("lr_scheduler_num_cycles", 1),
        ("lr_scheduler_power", 1.0),
        ("lr_scheduler_timescale", 0),  # Should be preserved
        ("lr_scheduler_min_lr_ratio", 0.0),  # Should be preserved
        
        # Test None values (should be excluded)
        ("optional_none_field", None),
        ("another_none_field", None),
        
        # GUI-specific exclusions (should be excluded)
        ("file_path", "/some/path"),
        ("save_as", "/some/save/path"), 
        ("headless", True),
        ("print_only", False),
    ]
    
    # Test output file
    output_file = "test_generated_config.toml"
    
    print("🧪 Testing TOML Generation...")
    print("=" * 50)
    
    # Generate TOML
    try:
        SaveConfigFileToRun(test_parameters, output_file)
        print(f"✅ TOML file generated successfully: {output_file}")
        
        # Read back and verify
        with open(output_file, 'r', encoding='utf-8') as f:
            generated_toml = toml.load(f)
            
        print(f"\n📊 Generated {len(generated_toml)} parameters")
        print("\n📝 TOML Content Preview:")
        print("-" * 30)
        
        # Show key parameters to verify
        critical_tests = {
            "0 values preserved": [
                ("lr_warmup_steps", 0),
                ("blocks_to_swap", 0), 
                ("min_timestep", 0),
                ("network_dropout", 0.0),
                ("logit_mean", 0.0),
            ],
            "False values preserved": [
                ("fp8_base", False),
                ("split_attn", False),
                ("vae_tiling", False),
            ],
            "Empty strings preserved": [
                ("network_weights", ""),
                ("resume", ""),
                ("logging_dir", ""),
            ],
            "Required values present": [
                ("dit", "/path/to/dit.safetensors"),
                ("vae", "/path/to/vae.safetensors"),
                ("text_encoder", "/path/to/text_encoder.safetensors"),
                ("dit_dtype", "bfloat16"),
            ],
            "None values excluded": [
                ("optional_none_field", "SHOULD_BE_MISSING"),
                ("another_none_field", "SHOULD_BE_MISSING"),
            ],
            "GUI exclusions": [
                ("file_path", "SHOULD_BE_MISSING"),
                ("save_as", "SHOULD_BE_MISSING"),
                ("headless", "SHOULD_BE_MISSING"),
                ("print_only", "SHOULD_BE_MISSING"),
            ]
        }
        
        # Verify critical tests
        all_passed = True
        
        for test_category, tests in critical_tests.items():
            print(f"\n🔍 Testing: {test_category}")
            category_passed = True
            
            for param_name, expected_value in tests:
                if expected_value == "SHOULD_BE_MISSING":
                    if param_name in generated_toml:
                        print(f"  ❌ {param_name}: Found in TOML (should be excluded)")
                        category_passed = False
                        all_passed = False
                    else:
                        print(f"  ✅ {param_name}: Correctly excluded")
                else:
                    if param_name in generated_toml:
                        actual_value = generated_toml[param_name]
                        if actual_value == expected_value:
                            print(f"  ✅ {param_name}: {actual_value} (correct)")
                        else:
                            print(f"  ❌ {param_name}: Expected {expected_value}, got {actual_value}")
                            category_passed = False
                            all_passed = False
                    else:
                        print(f"  ❌ {param_name}: Missing from TOML")
                        category_passed = False
                        all_passed = False
                        
            if category_passed:
                print(f"  ✅ {test_category}: ALL PASSED")
            else:
                print(f"  ❌ {test_category}: SOME FAILED")
        
        print("\n" + "=" * 50)
        if all_passed:
            print("🎉 ALL TESTS PASSED - TOML Generation 100% Accurate!")
        else:
            print("⚠️  SOME TESTS FAILED - Issues found in TOML generation")
            
        # Show sample of generated TOML
        print(f"\n📄 Sample Generated TOML (first 20 entries):")
        print("-" * 40)
        for i, (key, value) in enumerate(sorted(generated_toml.items())):
            if i >= 20:
                print(f"... and {len(generated_toml) - 20} more parameters")
                break
            print(f"{key} = {repr(value)}")
            
        return all_passed
        
    except Exception as e:
        print(f"❌ Error generating TOML: {e}")
        return False
    
    finally:
        # Clean up test file
        if os.path.exists(output_file):
            os.remove(output_file)

if __name__ == "__main__":
    success = test_toml_generation()
    sys.exit(0 if success else 1)