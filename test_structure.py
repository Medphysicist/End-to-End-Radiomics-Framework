#!/usr/bin/env python3
"""
Minimal test script to verify the code structure and syntax.
"""

import os
import sys

# Add the current directory to path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_code_structure():
    """Test that our code structure is valid and functions are properly defined."""
    
    print("🔍 Testing code structure...")
    
    # Test that files exist
    files_to_check = ['ui.py', 'processing.py', 'utils.py', 'app.py']
    
    for file in files_to_check:
        if os.path.exists(file):
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")
            return False
    
    # Test that we can parse the Python files
    for file in files_to_check:
        try:
            with open(file, 'r') as f:
                content = f.read()
                compile(content, file, 'exec')
            print(f"✅ {file} syntax valid")
        except SyntaxError as e:
            print(f"❌ {file} syntax error: {e}")
            return False
        except Exception as e:
            print(f"⚠️  {file} other error: {e}")
    
    return True

def test_function_definitions():
    """Test that required functions are defined."""
    
    print("\n🔍 Testing function definitions...")
    
    # Check ui.py functions
    ui_functions = [
        'build_tab1_data_upload',
        'build_single_patient_workflow',
        'build_multiple_patients_workflow',
        'display_single_patient_analysis',
        'display_directory_analysis',
        'continue_single_patient_processing',
        'continue_multiple_patients_processing'
    ]
    
    try:
        with open('ui.py', 'r') as f:
            ui_content = f.read()
        
        for func in ui_functions:
            if f'def {func}(' in ui_content:
                print(f"✅ ui.py::{func} defined")
            else:
                print(f"❌ ui.py::{func} missing")
                return False
    except Exception as e:
        print(f"❌ Error checking ui.py: {e}")
        return False
    
    # Check processing.py functions
    processing_functions = [
        'analyze_directory_structure',
        'get_single_patient_info',
        'check_imaging_rtstruct_compatibility'
    ]
    
    try:
        with open('processing.py', 'r') as f:
            processing_content = f.read()
        
        for func in processing_functions:
            if f'def {func}(' in processing_content:
                print(f"✅ processing.py::{func} defined")
            else:
                print(f"❌ processing.py::{func} missing")
                return False
    except Exception as e:
        print(f"❌ Error checking processing.py: {e}")
        return False
    
    # Check utils.py functions
    utils_functions = [
        'format_patient_summary',
        'calculate_processing_readiness'
    ]
    
    try:
        with open('utils.py', 'r') as f:
            utils_content = f.read()
        
        for func in utils_functions:
            if f'def {func}(' in utils_content:
                print(f"✅ utils.py::{func} defined")
            else:
                print(f"❌ utils.py::{func} missing")
                return False
    except Exception as e:
        print(f"❌ Error checking utils.py: {e}")
        return False
    
    return True

def test_session_state_keys():
    """Test that all required session state keys are defined."""
    
    print("\n🔍 Testing session state initialization...")
    
    required_keys = [
        'workflow_type',
        'directory_analysis',
        'single_patient_info',
        'selected_pair',
        'selected_roi',
        'selected_modality',
        'eligible_patients'
    ]
    
    try:
        with open('utils.py', 'r') as f:
            utils_content = f.read()
        
        for key in required_keys:
            if f"'{key}'" in utils_content:
                print(f"✅ Session state key '{key}' defined")
            else:
                print(f"❌ Session state key '{key}' missing")
                return False
    except Exception as e:
        print(f"❌ Error checking session state: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🧪 Running minimal code structure tests...\n")
    
    all_passed = True
    
    if not test_code_structure():
        all_passed = False
    
    if not test_function_definitions():
        all_passed = False
    
    if not test_session_state_keys():
        all_passed = False
    
    if all_passed:
        print("\n✅ All code structure tests passed!")
        print("🎉 The redesigned Tab 1 implementation appears to be correctly structured.")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)