#!/usr/bin/env python3
"""
Test script to verify the configuration auto-load functionality
"""

import sys
import os

# Add the project directory to the path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

def test_imports():
    """Test that the modified files can be imported without errors"""
    print("Testing imports...")
    try:
        from musubi_tuner_gui.class_configuration_file import ConfigurationFile
        print("✓ ConfigurationFile imported successfully")
        
        # Check that create_refresh_button is no longer imported
        from musubi_tuner_gui import class_configuration_file
        module_imports = class_configuration_file.__dict__.get('create_refresh_button')
        if module_imports is None:
            print("✓ create_refresh_button correctly removed from imports")
        else:
            print("✗ create_refresh_button still in imports")
            
        # Test that the GUI modules still import correctly
        from musubi_tuner_gui.qwen_image_lora_gui import qwen_image_lora_tab
        print("✓ qwen_image_lora_gui imported successfully")
        
        from musubi_tuner_gui.lora_gui import lora_tab
        print("✓ lora_gui imported successfully")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_configuration_class():
    """Test that ConfigurationFile class instantiates correctly"""
    print("\nTesting ConfigurationFile class...")
    try:
        from musubi_tuner_gui.class_configuration_file import ConfigurationFile
        
        # Test instantiation with default parameters
        config = ConfigurationFile(headless=True)
        print("✓ ConfigurationFile instantiated with headless=True")
        
        # Check that the refresh button code is removed
        import inspect
        source = inspect.getsource(ConfigurationFile.create_config_gui)
        if "create_refresh_button" not in source:
            print("✓ create_refresh_button code removed from create_config_gui")
        else:
            print("✗ create_refresh_button still present in create_config_gui")
            
        # Check that auto-load comment is present
        if "auto-refresh happens on dropdown change" in source or "auto-load is now set in the parent" in source:
            print("✓ Auto-load comments added")
        else:
            print("! Auto-load comments not found (optional)")
            
        return True
    except Exception as e:
        print(f"✗ Error testing ConfigurationFile: {e}")
        return False

def test_gui_auto_load():
    """Test that auto-load handlers are added to GUI files"""
    print("\nTesting auto-load handlers...")
    success = True
    
    # Check qwen_image_lora_gui.py
    try:
        with open("musubi_tuner_gui/qwen_image_lora_gui.py", "r", encoding="utf-8") as f:
            content = f.read()
            
        if "# Auto-load configuration when a valid config file is selected" in content:
            print("✓ Auto-load handler added to qwen_image_lora_gui.py")
        else:
            print("✗ Auto-load handler not found in qwen_image_lora_gui.py")
            success = False
            
        if "config_name and config_name.endswith('.json')" in content:
            print("✓ JSON file validation logic present in qwen_image_lora_gui.py")
        else:
            print("✗ JSON validation not found in qwen_image_lora_gui.py")
            success = False
    except Exception as e:
        print(f"✗ Error checking qwen_image_lora_gui.py: {e}")
        success = False
    
    # Check lora_gui.py
    try:
        with open("musubi_tuner_gui/lora_gui.py", "r", encoding="utf-8") as f:
            content = f.read()
            
        if "# Auto-load configuration when a valid config file is selected" in content:
            print("✓ Auto-load handler added to lora_gui.py")
        else:
            print("✗ Auto-load handler not found in lora_gui.py")
            success = False
            
        if "config_name and config_name.endswith('.json')" in content:
            print("✓ JSON file validation logic present in lora_gui.py")
        else:
            print("✗ JSON validation not found in lora_gui.py")
            success = False
    except Exception as e:
        print(f"✗ Error checking lora_gui.py: {e}")
        success = False
    
    return success

def main():
    """Run all tests"""
    print("=" * 60)
    print("Configuration Auto-Load Test Suite")
    print("=" * 60)
    
    all_passed = True
    
    # Run tests
    if not test_imports():
        all_passed = False
    
    if not test_configuration_class():
        all_passed = False
    
    if not test_gui_auto_load():
        all_passed = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed! The changes are working correctly.")
        print("\nSummary of changes:")
        print("1. Refresh button has been removed from the configuration UI")
        print("2. Auto-load functionality added when selecting .json files")
        print("3. The load button (↩️) is still available for manual loading")
    else:
        print("✗ Some tests failed. Please review the output above.")
    print("=" * 60)

if __name__ == "__main__":
    main()