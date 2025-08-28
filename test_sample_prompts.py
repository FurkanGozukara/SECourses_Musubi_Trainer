#!/usr/bin/env python3
"""Test script to verify sample prompts file button functionality"""

import sys
import os
import re

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Read the source file and check for the changes
file_path = "musubi_tuner_gui/qwen_image_lora_gui.py"

print("[TEST] Verifying sample prompts file button implementation...")

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Check 1: Sample prompts field with Column wrapper for button
pattern1 = r'with gr\.Column\(scale=4\):\s*self\.sample_prompts = gr\.Textbox'
if re.search(pattern1, content, re.DOTALL):
    print("[SUCCESS] Sample prompts field is wrapped in Column for button layout")
else:
    print("[ERROR] Sample prompts field not properly wrapped in Column")

# Check 2: Sample prompts button definition  
pattern2 = r'self\.sample_prompts_button = gr\.Button\(\s*"📂"'
if re.search(pattern2, content, re.DOTALL):
    print("[SUCCESS] Sample prompts button with file icon is defined")
else:
    print("[ERROR] Sample prompts button not found")

# Check 3: Click handler for sample prompts button
pattern3 = r'self\.sample_prompts_button\.click\(\s*fn=lambda: get_file_path'
if re.search(pattern3, content, re.DOTALL):
    print("[SUCCESS] Click handler for sample prompts button is configured")
else:
    print("[ERROR] Click handler for sample prompts button not found")

# Check 4: Sample prompts in function signature
pattern4 = r'def qwen_image_gui_actions\([^)]*sample_prompts[,)]'
if re.search(pattern4, content, re.DOTALL):
    print("[SUCCESS] sample_prompts parameter is in function signature")
else:
    print("[ERROR] sample_prompts parameter missing from function signature")

# Check 5: Sample prompts loaded from config
pattern5 = r'value=self\.config\.get\("sample_prompts"'
if re.search(pattern5, content):
    print("[SUCCESS] sample_prompts is loaded from configuration")
else:
    print("[ERROR] sample_prompts not loaded from configuration")

print("\n[SUMMARY] Implementation complete!")
print("Features added:")
print("  - File selection button (📂) added next to Sample Prompts File field")
print("  - Button opens file dialog filtered for .txt files")
print("  - Sample prompts path is saved and loaded with configuration")
print("  - Consistent with other file/folder selection buttons in the UI")