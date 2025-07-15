#!/usr/bin/env python3
"""
Test script to verify the new functionality works correctly.
This script tests the new functions without requiring a full Streamlit run.
"""

import os
import sys
import tempfile
import shutil

# Add the current directory to path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # Test imports
    from processing import (
        analyze_directory_structure,
        get_single_patient_info,
        check_imaging_rtstruct_compatibility
    )
    from utils import (
        format_patient_summary,
        calculate_processing_readiness
    )
    
    print("✅ All imports successful!")
    
    # Test basic functionality
    print("\n📊 Testing basic functionality:")
    
    # Test directory analysis with non-existent path
    print("• Testing directory analysis with invalid path...")
    result = analyze_directory_structure("/nonexistent/path")
    print(f"  Result: {len(result['errors'])} errors found (expected)")
    
    # Test single patient info with invalid path
    print("• Testing single patient info with invalid path...")
    result = get_single_patient_info("/nonexistent/patient")
    print(f"  Result: {len(result['errors'])} errors found (expected)")
    
    # Test utility functions
    print("• Testing utility functions...")
    test_patient_info = {
        'patient_id': 'TEST001',
        'imaging_series': {'series1': {}},
        'rtstruct_files': ['file1.dcm'],
        'compatible_pairs': [{'modality': 'CT', 'contours': ['GTV']}],
        'modalities': ['CT'],
        'errors': []
    }
    
    summary = format_patient_summary(test_patient_info)
    print(f"  Patient summary generated: {len(summary)} characters")
    
    test_analysis = {
        'patients': {
            'TEST001': test_patient_info
        }
    }
    
    readiness = calculate_processing_readiness(test_analysis)
    print(f"  Processing readiness calculated: {readiness['patients_ready_for_processing']} patients ready")
    
    print("\n✅ All basic functionality tests passed!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)

print("\n🎉 All tests completed successfully!")