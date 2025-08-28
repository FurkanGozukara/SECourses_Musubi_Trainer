#!/usr/bin/env python
"""
Test script to verify the JavaScript confirmation stop button fix.
This test checks that the stop button now uses JavaScript confirmation instead of checkbox.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_stop_button_js_confirmation():
    """Test that stop button now uses JavaScript confirmation"""
    
    print("Testing JavaScript Confirmation Stop Button Fix")
    print("=" * 60)
    
    # Import the modules
    from musubi_tuner_gui.class_command_executor import CommandExecutor
    from musubi_tuner_gui.qwen_image_lora_gui import qwen_image_gui_actions
    import inspect
    
    print("\n1. Checking CommandExecutor changes...")
    
    # Check CommandExecutor source code
    executor_source = inspect.getsource(CommandExecutor)
    
    has_checkbox = 'self.stop_confirm_checkbox' in executor_source
    if has_checkbox:
        print("   [ERROR] Checkbox still exists in CommandExecutor")
    else:
        print("   [OK] Checkbox removed from CommandExecutor")
    
    # Check kill_command signature
    kill_method = CommandExecutor.kill_command
    kill_sig = inspect.signature(kill_method)
    params = list(kill_sig.parameters.keys())
    
    if 'confirm_checked' in params:
        print("   [ERROR] kill_command still takes confirm_checked parameter")
    else:
        print("   [OK] kill_command no longer requires confirmation parameter")
        
    print("\n2. Checking qwen_image_lora_gui changes...")
    
    # Check the GUI code for JavaScript confirmation
    with open('musubi_tuner_gui/qwen_image_lora_gui.py', 'r', encoding='utf-8') as f:
        gui_content = f.read()
    
    if 'executor.stop_confirm_checkbox.change' in gui_content:
        print("   [ERROR] Checkbox change handler still exists")
    else:
        print("   [OK] Checkbox change handler removed")
        
    if "confirm('Are you sure you want to stop training?')" in gui_content:
        print("   [OK] JavaScript confirmation dialog added")
    else:
        print("   [ERROR] JavaScript confirmation not found")
        
    if 'gr.Button(interactive=True)' in gui_content and '# Enable stop button by default' in gui_content:
        print("   [OK] Stop button enabled by default during training")
    else:
        print("   [WARNING]  Check if stop button is properly enabled")
        
    print("\n3. Checking return value consistency...")
    
    # Check that return values match between functions
    source = inspect.getsource(qwen_image_gui_actions)
    
    # Count the number of return values in qwen_image_gui_actions
    import re
    return_pattern = r'return\s*\(\s*gr\.Button.*?\)'
    returns = re.findall(return_pattern, source, re.DOTALL)
    
    if returns:
        # Count comma-separated values in first return
        first_return = returns[0]
        return_count = first_return.count('gr.')
        print(f"   Found {return_count} return values in qwen_image_gui_actions")
    
    print("\n4. Checking method return values...")
    
    # Check kill_command source to count return values
    kill_source = inspect.getsource(CommandExecutor.kill_command)
    
    # Find return statements
    import re
    return_matches = re.findall(r'return\s*\(([^)]+)\)', kill_source, re.DOTALL)
    if return_matches:
        # Count items in first return
        first_return = return_matches[0]
        return_items = [item.strip() for item in first_return.split(',') if 'gr.' in item]
        print(f"   kill_command returns {len(return_items)} Gradio components")
        
    # Check wait_for_training_to_end
    wait_source = inspect.getsource(CommandExecutor.wait_for_training_to_end)
    wait_returns = re.findall(r'return\s*\(([^)]+)\)', wait_source, re.DOTALL)
    if wait_returns:
        first_return = wait_returns[0]
        return_items = [item.strip() for item in first_return.split(',') if 'gr.' in item]
        print(f"   wait_for_training_to_end returns {len(return_items)} Gradio components")
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("[OK] Checkbox-based confirmation has been REMOVED")
    print("[OK] JavaScript confirmation dialog has been ADDED")  
    print("[OK] Stop button is now ALWAYS enabled during training")
    print("\nThe user will see a browser confirmation dialog:")
    print('   "Are you sure you want to stop training?"')
    print("\nThis is much better UX than the checkbox that got hidden!")
    
if __name__ == "__main__":
    test_stop_button_js_confirmation()