#!/usr/bin/env python
"""
Deep parameter validation script to identify all potential parameter errors
between GUI configuration and musubi tuner expectations.
"""

import os
import re
import sys
import toml
import ast
from typing import Dict, List, Set, Tuple

def find_division_operations(file_path: str) -> List[Tuple[int, str, str]]:
    """Find all division and modulo operations with args parameters."""
    results = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines, 1):
            # Look for modulo operations with args.
            if '% args.' in line or '/ args.' in line:
                # Extract the parameter name
                param_match = re.search(r'args\.(\w+)', line)
                if param_match:
                    param_name = param_match.group(1)
                    results.append((i, param_name, line.strip()))
            
            # Also look for divisions with variables that might come from args
            if ' % ' in line or ' / ' in line:
                if 'args.' in line:
                    results.append((i, "potential", line.strip()))
    
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    return results

def find_none_checks(file_path: str) -> List[Tuple[int, str, str]]:
    """Find all None checks for args parameters."""
    results = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines, 1):
            # Look for None checks
            if 'args.' in line and ('is not None' in line or 'is None' in line or '!= None' in line or '== None' in line):
                param_match = re.search(r'args\.(\w+)', line)
                if param_match:
                    param_name = param_match.group(1)
                    results.append((i, param_name, line.strip()))
    
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    return results

def analyze_config_defaults() -> Dict:
    """Analyze default configuration values."""
    config_files = [
        'E:/SECourses_Improved_Trainer_v1/SECourses_Improved_Trainer/config.toml',
        'E:/SECourses_Improved_Trainer_v1/SECourses_Improved_Trainer/qwen_image_defaults.toml'
    ]
    
    all_configs = {}
    
    for config_file in config_files:
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = toml.load(f)
                    all_configs.update(config)
            except Exception as e:
                print(f"Error loading {config_file}: {e}")
    
    return all_configs

def find_risky_parameters(configs: Dict) -> List[Tuple[str, any, str]]:
    """Find parameters that could cause issues."""
    risky = []
    
    for key, value in configs.items():
        # Check for 0 values that might be used in division
        if value == 0:
            risky.append((key, value, "Zero value - potential division by zero"))
        
        # Check for empty strings that might be expected as None
        elif value == "":
            risky.append((key, value, "Empty string - might need None"))
        
        # Check for string representations of lists
        elif value == "[]":
            risky.append((key, value, "String '[]' - should be actual list"))
    
    return risky

def scan_musubi_tuner():
    """Scan musubi tuner source for potential issues."""
    musubi_dir = 'E:/SECourses_Improved_Trainer_v1/SECourses_Improved_Trainer/musubi-tuner/src/musubi_tuner'
    
    division_issues = {}
    none_checks = {}
    
    for root, dirs, files in os.walk(musubi_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, musubi_dir)
                
                # Find division operations
                divisions = find_division_operations(file_path)
                if divisions:
                    division_issues[rel_path] = divisions
                
                # Find None checks
                checks = find_none_checks(file_path)
                if checks:
                    none_checks[rel_path] = checks
    
    return division_issues, none_checks

def main():
    print("=" * 80)
    print("DEEP PARAMETER VALIDATION CHECK")
    print("=" * 80)
    
    # Load configurations
    print("\n1. LOADING CONFIGURATIONS...")
    configs = analyze_config_defaults()
    print(f"   Loaded {len(configs)} configuration parameters")
    
    # Find risky parameters
    print("\n2. ANALYZING RISKY PARAMETERS...")
    risky_params = find_risky_parameters(configs)
    
    if risky_params:
        print(f"   Found {len(risky_params)} potentially risky parameters:")
        
        # Group by risk type
        zero_params = [p for p in risky_params if p[2] == "Zero value - potential division by zero"]
        empty_params = [p for p in risky_params if p[2] == "Empty string - might need None"]
        list_params = [p for p in risky_params if p[2] == "String '[]' - should be actual list"]
        
        if zero_params:
            print("\n   Parameters with value 0 (division by zero risk):")
            for param, value, risk in zero_params[:10]:  # Show first 10
                print(f"      - {param} = {value}")
            if len(zero_params) > 10:
                print(f"      ... and {len(zero_params) - 10} more")
        
        if empty_params:
            print("\n   Parameters with empty string (might need None):")
            for param, value, risk in empty_params[:10]:
                print(f"      - {param} = '{value}'")
            if len(empty_params) > 10:
                print(f"      ... and {len(empty_params) - 10} more")
        
        if list_params:
            print("\n   Parameters with string '[]' (should be list):")
            for param, value, risk in list_params:
                print(f"      - {param} = '{value}'")
    
    # Scan musubi tuner code
    print("\n3. SCANNING MUSUBI TUNER CODE...")
    division_issues, none_checks = scan_musubi_tuner()
    
    # Analyze division operations
    print("\n4. DIVISION/MODULO OPERATIONS WITH args PARAMETERS:")
    
    critical_params = set()
    for file, issues in division_issues.items():
        for line_no, param, code in issues:
            if param != "potential" and '% args.' in code:
                critical_params.add(param)
                print(f"\n   File: {file}:{line_no}")
                print(f"   Parameter: args.{param}")
                print(f"   Code: {code}")
                
                # Check if this parameter is 0 in config
                if param in configs and configs[param] == 0:
                    print(f"   [WARNING] {param} is 0 in config - WILL CAUSE DIVISION BY ZERO!")
    
    # Analyze None checks
    print("\n5. PARAMETERS CHECKED FOR None:")
    
    none_expected = set()
    for file, checks in none_checks.items():
        for line_no, param, code in checks:
            if 'is not None' in code:
                none_expected.add(param)
    
    print(f"\n   Found {len(none_expected)} parameters that are checked for None")
    
    # Cross-reference with config
    print("\n6. CROSS-REFERENCE ANALYSIS:")
    
    problematic = []
    for param in none_expected:
        if param in configs:
            if configs[param] == 0:
                problematic.append((param, 0, "Should be None when disabled, not 0"))
            elif configs[param] == "":
                problematic.append((param, "", "Should be None when disabled, not empty string"))
    
    if problematic:
        print("\n   FOUND PROBLEMATIC PARAMETERS:")
        for param, value, issue in problematic:
            print(f"      - {param} = {repr(value)}: {issue}")
    
    # Final recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS:")
    print("=" * 80)
    
    params_needing_fix = set()
    
    # Parameters that use modulo/division and can be 0
    for param in critical_params:
        if param in configs and configs[param] == 0:
            params_needing_fix.add(param)
    
    # Parameters expected to be None but are 0 or ""
    for param, value, _ in problematic:
        params_needing_fix.add(param)
    
    if params_needing_fix:
        print("\nParameters that need 0 -> None conversion in SaveConfigFile:")
        for param in sorted(params_needing_fix):
            print(f"   - {param}")
    
    # Check current implementation
    print("\n" + "=" * 80)
    print("CURRENT FIX STATUS:")
    print("=" * 80)
    
    with open('E:/SECourses_Improved_Trainer_v1/SECourses_Improved_Trainer/musubi_tuner_gui/common_gui.py', 'r') as f:
        content = f.read()
    
    # Find the current conversion logic
    match = re.search(r'if name in \[(.*?)\]:\s*\n\s*if value == 0:\s*\n\s*value = None', content, re.DOTALL)
    if match:
        current_params = re.findall(r'"(\w+)"', match.group(1))
        print(f"\nCurrently handling: {current_params}")
        
        missing = params_needing_fix - set(current_params)
        if missing:
            print(f"\nMISSING PARAMETERS that also need handling:")
            for param in sorted(missing):
                print(f"   - {param}")
        else:
            print("\n[OK] All critical parameters are already being handled!")
    
    return params_needing_fix

if __name__ == "__main__":
    critical_params = main()
    print("\n" + "=" * 80)
    print(f"Total critical parameters found: {len(critical_params)}")