# utils.py
"""
This module contains various utility and helper functions for the Radiomics Pipeline application.
These functions support the main application logic by handling tasks like session state
initialization, system monitoring, and data validation.
"""

import streamlit as st
import pandas as pd
import psutil
import os
import shutil
import atexit

# --- Session State Management ---

def initialize_session_state():
    """
    Initializes all the necessary keys in Streamlit's session state.
    This function is called once at the beginning of the app run to ensure
    all keys exist, preventing KeyErrors.
    """
    defaults = {
        'uploaded_data_path': None,      # Path to organized DICOM data
        'temp_output_dir': None,         # Path to temporary NIfTI files
        'preprocessing_done': False,     # Flag to indicate if pre-processing is complete
        'dataset_df': pd.DataFrame(),    # DataFrame with paths to NIfTI files
        'all_contours': [],              # List of all unique contours found
        'patient_contour_data': {},      # Dict mapping patients to their contours
        'patient_status': {},            # Dict tracking processing status per patient
        'features_df': None,             # DataFrame with extracted radiomic features
        'clinical_df': None,             # DataFrame with uploaded clinical data
        'merged_df': None,               # DataFrame with merged features and clinical data
        'cleanup_registered': False      # Flag to ensure cleanup runs only once
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


# --- System and File Helpers ---

def check_system_resources():
    """
    Check system resources and return information about available RAM and CPU cores.
    
    Returns:
        dict: Dictionary containing system resource information
    """
    try:
        # Get memory information
        memory = psutil.virtual_memory()
        available_ram_gb = memory.available / (1024**3)  # Convert bytes to GB
        
        # Get CPU information
        cpu_count = os.cpu_count()
        
        return {
            'available_ram_gb': available_ram_gb,
            'cpu_count': cpu_count,
            'total_ram_gb': memory.total / (1024**3),
            'used_ram_gb': memory.used / (1024**3),
            'ram_percent': memory.percent
        }
    except Exception as e:
        # Return fallback values if there's an error
        return {
            'available_ram_gb': 0.0,
            'cpu_count': 1,
            'total_ram_gb': 0.0,
            'used_ram_gb': 0.0,
            'ram_percent': 0.0,
            'error': str(e)
        }


def validate_uploaded_files(uploaded_files, max_file_size_mb=500, max_total_size_mb=2000):
    """
    Validates the size of uploaded files against predefined limits.

    Args:
        uploaded_files (list): A list of Streamlit UploadedFile objects.
        max_file_size_mb (int): The maximum size for a single file in MB.
        max_total_size_mb (int): The maximum total size for all files in MB.

    Returns:
        list: A list of error messages. An empty list means validation passed.
    """
    total_size_mb = sum(len(file.getvalue()) for file in uploaded_files) / (1024 * 1024)
    issues = []
    
    if total_size_mb > max_total_size_mb:
        issues.append(f"Total upload size ({total_size_mb:.1f} MB) exceeds the limit of {max_total_size_mb} MB.")

    for file in uploaded_files:
        size_mb = len(file.getvalue()) / (1024 * 1024)
        if size_mb > max_file_size_mb:
            issues.append(f"File '{file.name}' ({size_mb:.1f} MB) exceeds the single file limit of {max_file_size_mb} MB.")
    
    return issues


def cleanup_temp_dirs():
    """
    Removes temporary directories created during the session.
    This function is designed to be registered with `atexit` to run automatically
    when the Streamlit script process ends.
    """
    print("Running cleanup...")
    # Clean up the main organized DICOM directory if it's in a temp location
    if st.session_state.get('uploaded_data_path') and 'radiomics_upload_' in st.session_state.uploaded_data_path:
        try:
            shutil.rmtree(st.session_state.uploaded_data_path)
            print(f"Cleaned up {st.session_state.uploaded_data_path}")
        except Exception as e:
            print(f"Error cleaning up uploaded_data_path: {e}")
            
    # Clean up the NIfTI output directory
    if st.session_state.get('temp_output_dir') and os.path.exists(st.session_state.temp_output_dir):
        try:
            shutil.rmtree(st.session_state.temp_output_dir)
            print(f"Cleaned up {st.session_state.temp_output_dir}")
        except Exception as e:
            print(f"Error cleaning up temp_output_dir: {e}")


def register_cleanup():
    """
    Registers the cleanup function to run on script exit.
    Uses a session_state flag to ensure it's only registered once per session.
    """
    if not st.session_state.get('cleanup_registered'):
        atexit.register(cleanup_temp_dirs)
        st.session_state['cleanup_registered'] = True
        print("Cleanup function registered.")


# --- Data-Specific Helpers ---

def categorize_contours(contour_list):
    """
    Categorizes a list of contour names into 'Targets', 'Organs at Risk', and 'Other'.

    Args:
        contour_list (list): A list of contour name strings.

    Returns:
        tuple: A tuple containing three lists: (targets, oars, other).
    """
    # Keywords are converted to lowercase for case-insensitive matching
    target_keywords = [
        'gtv', 'ctv', 'ptv', 'tumor', 'lesion', 'target', 'boost'
    ]
    oar_keywords = [
        'brain', 'stem', 'spinal', 'cord', 'heart', 'lung', 'liver', 'kidney',
        'bladder', 'rectum', 'bowel', 'esophagus', 'parotid', 'eye', 'lens',
        'optic', 'chiasm', 'cochlea', 'mandible', 'skin'
    ]

    targets, oars, other = [], [], []

    for contour in contour_list:
        contour_lower = contour.lower()
        is_target = any(keyword in contour_lower for keyword in target_keywords)
        is_oar = any(keyword in contour_lower for keyword in oar_keywords)

        if is_target:
            targets.append(contour)
        elif is_oar:
            oars.append(contour)
        else:
            other.append(contour)

    return sorted(targets), sorted(oars), sorted(other)
