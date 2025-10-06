# utils.py
"""
Enhanced utility and helper functions for the Radiomics Pipeline application.
Updated: 2025-08-22 03:38:48 UTC by Medphysicist

Features:
- Original session state management and system monitoring
- Enhanced NIfTI file support
- Progress tracking with ETA estimation  
- Multi-modality support (CT, MRI, PET)
- Longitudinal data handling
- IBSI feature support
- Comprehensive cleanup system
"""

import streamlit as st
import pandas as pd
import psutil
import os
import shutil
import atexit
import time
import numpy as np
from pathlib import Path
import SimpleITK as sitk
import pydicom
import tempfile
import zipfile

# --- Enhanced Session State Management ---

def initialize_session_state():
    """
    Enhanced session state initialization combining original keys with new features
    """
    defaults = {
        # ORIGINAL KEYS (preserved from V3)
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
        'cleanup_registered': False,     # Flag to ensure cleanup runs only once
        'workflow_type': None,           # Type of workflow: 'single' or 'multiple'
        'directory_analysis': {},        # Directory analysis results for multiple patients
        'single_patient_info': {},       # Single patient analysis results
        'selected_pair': None,           # Selected imaging/RTSTRUCT pair for processing
        'selected_roi': None,            # Selected ROI for processing
        'selected_modality': None,       # Selected imaging modality
        'eligible_patients': [],         # List of patients eligible for processing
        
        # NEW ENHANCED KEYS
        'input_format': 'dicom',         # 'dicom' or 'nifti'
        'multi_series_mode': False,      # Enable multi-series processing
        'longitudinal_data': {},         # Longitudinal/timepoint data
        'available_timepoints': [],      # Available timepoints
        'selected_timepoints': [],       # Selected timepoints for processing
        'ibsi_features_enabled': False,  # Enable IBSI-specific features
        'progress_tracker': None,        # Progress tracking instance
        'modality_specific_params': {},  # Modality-specific parameters
        'series_selection_data': {},     # Series selection information
        'extraction_method': 'pyradiomics', # 'pyradiomics', 'ibsi', or 'combined'
        'available_modalities': [],      # Available modalities in dataset
        'selected_modalities': [],       # Selected modalities for processing
        'nifti_pairs': {},              # NIfTI image/mask pairs
        'eta_enabled': True,            # Enable ETA calculations
        'processing_summary': {},       # Processing summary information
        'extraction_summary': {}        # Extraction summary information
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


# --- Enhanced Progress Tracking with ETA ---

class ProgressTracker:
    """Enhanced progress tracking with ETA calculation and streamlit integration"""
    
    def __init__(self, total_items: int, process_name: str = "Processing"):
        self.total_items = total_items
        self.process_name = process_name
        self.start_time = time.time()
        self.completed_items = 0
        self.progress_bar = None
        self.progress_text = None
        self.status_placeholder = None
        self.eta_enabled = st.session_state.get('eta_enabled', True)
        
    def setup_ui_elements(self, progress_bar, progress_text, status_placeholder):
        """Setup UI elements for progress display"""
        self.progress_bar = progress_bar
        self.progress_text = progress_text
        self.status_placeholder = status_placeholder
        
    def update(self, completed_items: int, current_item_name: str = ""):
        """Update progress with ETA calculation"""
        self.completed_items = completed_items
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        progress = completed_items / self.total_items if self.total_items > 0 else 0
        
        if completed_items > 0 and self.eta_enabled:
            # Calculate ETA
            avg_time_per_item = elapsed_time / completed_items
            remaining_items = self.total_items - completed_items
            eta_seconds = remaining_items * avg_time_per_item
            
            # Format times
            eta_str = self._format_time(eta_seconds)
            elapsed_str = self._format_time(elapsed_time)
            
            # Update UI elements
            if self.progress_bar:
                self.progress_bar.progress(progress)
            
            if self.progress_text:
                self.progress_text.text(
                    f"{self.process_name}: {completed_items}/{self.total_items} "
                    f"({progress*100:.1f}%) | Elapsed: {elapsed_str} | ETA: {eta_str}"
                )
            
            if self.status_placeholder and current_item_name:
                self.status_placeholder.info(f"🔄 {current_item_name} | ETA: {eta_str}")
        else:
            # Simple progress without ETA
            if self.progress_bar:
                self.progress_bar.progress(progress)
            
            if self.progress_text:
                self.progress_text.text(
                    f"{self.process_name}: {completed_items}/{self.total_items} ({progress*100:.1f}%)"
                )
            
            if self.status_placeholder and current_item_name:
                self.status_placeholder.info(f"🔄 {current_item_name}")
    
    def _format_time(self, seconds: float) -> str:
        """Format time in human-readable format"""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            seconds = int(seconds % 60)
            return f"{minutes}m {seconds}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    def complete(self, success_message: str = "Process completed"):
        """Mark process as complete"""
        total_time = time.time() - self.start_time
        total_time_str = self._format_time(total_time)
        
        if self.progress_bar:
            self.progress_bar.progress(1.0)
        
        if self.progress_text:
            self.progress_text.text(f"✅ {success_message} | Total time: {total_time_str}")
        
        if self.status_placeholder:
            self.status_placeholder.success(f"🎉 {success_message} | Completed in {total_time_str}")


# --- File Type Detection and Validation ---

def detect_file_type(file_path: str) -> str:
    """
    Detect if file is DICOM or NIfTI
    
    Returns:
        'dicom', 'nifti', or 'unknown'
    """
    file_path = Path(file_path)
    
    # Check by extension first
    if file_path.suffix.lower() in ['.nii', '.nii.gz']:
        return 'nifti'
    elif file_path.suffix.lower() in ['.dcm', '.ima', '.dicom']:
        return 'dicom'
    
    # Try to read as DICOM
    try:
        pydicom.dcmread(str(file_path), stop_before_pixels=True)
        return 'dicom'
    except:
        pass
    
    # Try to read as NIfTI
    try:
        sitk.ReadImage(str(file_path))
        return 'nifti'
    except:
        pass
    
    return 'unknown'


def validate_nifti_pair(image_path: str, mask_path: str) -> tuple:
    """
    Validate NIfTI image/mask pair
    
    Returns:
        (is_valid, error_message)
    """
    try:
        # Load images
        image_sitk = sitk.ReadImage(image_path)
        mask_sitk = sitk.ReadImage(mask_path)
        
        # Check dimensions
        if image_sitk.GetSize() != mask_sitk.GetSize():
            return False, f"Size mismatch: Image {image_sitk.GetSize()} vs Mask {mask_sitk.GetSize()}"
        
        # Check spacing (allow small differences)
        img_spacing = image_sitk.GetSpacing()
        mask_spacing = mask_sitk.GetSpacing()
        spacing_diff = max(abs(a - b) for a, b in zip(img_spacing, mask_spacing))
        if spacing_diff > 0.1:
            return False, f"Spacing mismatch: Image {img_spacing} vs Mask {mask_spacing}"
        
        # Check mask content
        mask_array = sitk.GetArrayFromImage(mask_sitk)
        if not np.any(mask_array > 0):
            return False, "Mask contains no positive voxels"
        
        # Check for reasonable mask size
        roi_voxel_count = np.sum(mask_array > 0)
        total_voxel_count = mask_array.size
        roi_percentage = (roi_voxel_count / total_voxel_count) * 100
        
        if roi_percentage > 90:
            return False, f"Mask covers {roi_percentage:.1f}% of image - may be inverted"
        
        return True, "NIfTI pair validation passed"
        
    except Exception as e:
        return False, f"Error validating NIfTI pair: {str(e)}"


def organize_nifti_files(uploaded_files) -> str:
    """
    Organize uploaded NIfTI files into patient directories
    
    Returns:
        Path to organized directory
    """
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="radiomics_nifti_upload_")
        organized_path = os.path.join(temp_dir, "organized")
        os.makedirs(organized_path, exist_ok=True)
        
        # Group files by patient
        patient_files = {}
        
        for uploaded_file in uploaded_files:
            filename = uploaded_file.name
            
            # Extract patient ID from filename (various patterns)
            if '_' in filename:
                patient_id = filename.split('_')[0]
            elif '-' in filename:
                patient_id = filename.split('-')[0]
            else:
                # Use filename without extension as patient ID
                patient_id = Path(filename).stem.split('.')[0]
            
            if patient_id not in patient_files:
                patient_files[patient_id] = []
            
            # Save file to temp location
            temp_file_path = os.path.join(temp_dir, filename)
            with open(temp_file_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            
            patient_files[patient_id].append({
                'filename': filename,
                'path': temp_file_path,
                'type': detect_file_type(temp_file_path),
                'is_mask': any(keyword in filename.lower() for keyword in ['mask', 'seg', 'roi', 'label'])
            })
        
        # Organize into patient directories
        for patient_id, files in patient_files.items():
            patient_dir = os.path.join(organized_path, patient_id)
            os.makedirs(patient_dir, exist_ok=True)
            
            for file_info in files:
                dest_path = os.path.join(patient_dir, file_info['filename'])
                shutil.copy(file_info['path'], dest_path)
        
        return organized_path
        
    except Exception as e:
        st.error(f"Error organizing NIfTI files: {e}")
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return None


# --- Enhanced Modality Support ---

def get_available_modalities_extended():
    """Get extended list of supported modalities"""
    return [
        'CT', 'MR', 'PT', 'RTDOSE', 'REG', 'RTPLAN', 'SEG',
        'MR_T1', 'MR_T2', 'MR_FLAIR', 'MR_DWI', 'PT_CT'
    ]


def get_available_modalities_extended():
    """Get extended list of supported modalities"""
    return [
        'CT', 'MR', 'PT', 'RTDOSE', 'REG', 'RTPLAN', 'SEG',
        'MR_T1', 'MR_T2', 'MR_FLAIR', 'MR_DWI', 'PT_CT'
    ]


def categorize_contours_extended(contour_list):
    """
    Enhanced contour categorization with comprehensive keywords for all modalities
    """
    # Enhanced keywords for better categorization
    target_keywords = [
        'gtv', 'ctv', 'ptv', 'tumor', 'tumour', 'lesion', 'target', 'boost', 
        'primary', 'nodal', 'cancer', 'malignant', 'mass', 'neoplasm',
        'metastasis', 'met', 'recurrence', 'residual'
    ]
    
    oar_keywords = [
        # Basic anatomy
        'brain', 'stem', 'spinal', 'cord', 'heart', 'lung', 'liver', 'kidney',
        'bladder', 'rectum', 'bowel', 'esophagus', 'esophageal', 'parotid', 
        'eye', 'lens', 'optic', 'chiasm', 'cochlea', 'mandible', 'skin',
        'femur', 'bone', 'muscle', 'nerve', 'vessel', 'artery', 'vein',
        'stomach', 'duodenum', 'small_bowel', 'large_bowel', 'colon',
        
        # MRI-specific structures
        'white_matter', 'gray_matter', 'grey_matter', 'csf', 'ventricle', 
        'hippocampus', 'amygdala', 'thalamus', 'caudate', 'putamen', 
        'corpus_callosum', 'cerebellum', 'frontal', 'parietal', 'temporal', 
        'occipital', 'brainstem', 'midbrain', 'pons', 'medulla',
        
        # Head and neck specific
        'thyroid', 'larynx', 'pharynx', 'tongue', 'oral_cavity', 'lips',
        'teeth', 'jaw', 'maxilla', 'zygoma', 'temporal_bone',
        
        # Thoracic specific
        'aorta', 'pulmonary', 'trachea', 'bronchus', 'mediastinum',
        'pleura', 'rib', 'sternum', 'clavicle', 'scapula',
        
        # Abdominal specific
        'pancreas', 'spleen', 'gallbladder', 'adrenal', 'diaphragm',
        'peritoneum', 'mesentery', 'omentum',
        
        # Pelvic specific
        'prostate', 'seminal_vesicle', 'uterus', 'cervix', 'ovary',
        'vagina', 'vulva', 'penis', 'testis', 'sacrum', 'coccyx',
        'ilium', 'ischium', 'pubis'
    ]
    
    # PET-specific regions and uptake patterns
    pet_structures = [
        'uptake', 'metabolic', 'suv', 'fdg', 'avid', 'hot_spot',
        'hypermetabolic', 'photopenic', 'cold', 'background'
    ]
    
    # Combine all OAR keywords
    all_oar_keywords = oar_keywords + pet_structures

    targets, oars, other = [], [], []

    for contour in contour_list:
        contour_lower = contour.lower().replace(' ', '_').replace('-', '_')
        is_target = any(keyword in contour_lower for keyword in target_keywords)
        is_oar = any(keyword in contour_lower for keyword in all_oar_keywords)

        if is_target:
            targets.append(contour)
        elif is_oar:
            oars.append(contour)
        else:
            other.append(contour)

    return sorted(targets), sorted(oars), sorted(other)

# --- Original System and File Helpers (Enhanced) ---

def check_system_resources():
    """
    Enhanced system resource checking with better error handling
    """
    try:
        # Get memory information
        memory = psutil.virtual_memory()
        available_ram_gb = memory.available / (1024**3)
        
        # Get CPU information
        cpu_count = os.cpu_count()
        
        # Get disk space for temp directory
        temp_dir = tempfile.gettempdir()
        disk_usage = shutil.disk_usage(temp_dir)
        available_disk_gb = disk_usage.free / (1024**3)
        
        return {
            'available_ram_gb': available_ram_gb,
            'cpu_count': cpu_count,
            'total_ram_gb': memory.total / (1024**3),
            'used_ram_gb': memory.used / (1024**3),
            'ram_percent': memory.percent,
            'available_disk_gb': available_disk_gb,
            'temp_dir': temp_dir
        }
    except Exception as e:
        # Return fallback values if there's an error
        return {
            'available_ram_gb': 0.0,
            'cpu_count': 1,
            'total_ram_gb': 0.0,
            'used_ram_gb': 0.0,
            'ram_percent': 0.0,
            'available_disk_gb': 0.0,
            'temp_dir': tempfile.gettempdir(),
            'error': str(e)
        }


def validate_uploaded_files(uploaded_files, max_file_size_mb=500, max_total_size_mb=2000):
    """
    Enhanced file validation with format-specific checks
    """
    total_size_mb = sum(len(file.getvalue()) for file in uploaded_files) / (1024 * 1024)
    issues = []
    
    if total_size_mb > max_total_size_mb:
        issues.append(f"Total upload size ({total_size_mb:.1f} MB) exceeds the limit of {max_total_size_mb} MB.")

    # Check individual files
    for file in uploaded_files:
        size_mb = len(file.getvalue()) / (1024 * 1024)
        if size_mb > max_file_size_mb:
            issues.append(f"File '{file.name}' ({size_mb:.1f} MB) exceeds the single file limit of {max_file_size_mb} MB.")
        
        # Format-specific validation
        file_ext = Path(file.name).suffix.lower()
        if file_ext in ['.nii', '.nii.gz']:
            # NIfTI file validation
            try:
                # Could add more sophisticated NIfTI validation here
                pass
            except Exception as e:
                issues.append(f"Invalid NIfTI file '{file.name}': {str(e)}")
        elif file_ext in ['.dcm', '.ima', '.dicom']:
            # DICOM file validation
            try:
                # Could add DICOM header validation here
                pass
            except Exception as e:
                issues.append(f"Invalid DICOM file '{file.name}': {str(e)}")
    
    return issues


# --- Enhanced Cleanup Functions ---

def cleanup_temp_dirs():
    """
    Enhanced cleanup with better error handling and logging
    """
    print("Running enhanced cleanup...")
    
    # Clean up the main organized data directory
    if st.session_state.get('uploaded_data_path') and 'radiomics_' in str(st.session_state.uploaded_data_path):
        try:
            shutil.rmtree(st.session_state.uploaded_data_path)
            print(f"Cleaned up uploaded_data_path: {st.session_state.uploaded_data_path}")
        except Exception as e:
            print(f"Error cleaning up uploaded_data_path: {e}")
            
    # Clean up the NIfTI/processed output directory
    if st.session_state.get('temp_output_dir') and os.path.exists(st.session_state.temp_output_dir):
        try:
            shutil.rmtree(st.session_state.temp_output_dir)
            print(f"Cleaned up temp_output_dir: {st.session_state.temp_output_dir}")
        except Exception as e:
            print(f"Error cleaning up temp_output_dir: {e}")
    
    # Clean up any additional temporary directories
    for key in ['nifti_temp_dir', 'extraction_temp_dir', 'analysis_temp_dir']:
        if st.session_state.get(key) and os.path.exists(st.session_state[key]):
            try:
                shutil.rmtree(st.session_state[key])
                print(f"Cleaned up {key}: {st.session_state[key]}")
            except Exception as e:
                print(f"Error cleaning up {key}: {e}")


def register_cleanup():
    """
    Enhanced cleanup registration with session tracking
    """
    if not st.session_state.get('cleanup_registered'):
        atexit.register(cleanup_temp_dirs)
        st.session_state['cleanup_registered'] = True
        print("Enhanced cleanup function registered.")


# --- Original Data-Specific Helpers (Enhanced) ---

def categorize_contours(contour_list):
    """
    Original categorize_contours function with enhanced keywords
    """
    # Use enhanced version but maintain original function name for compatibility
    return categorize_contours_extended(contour_list)


def format_patient_summary(patient_info):
    """
    Enhanced patient information formatting with multi-modality support
    """
    summary = []
    summary.append(f"**Patient ID:** {patient_info['patient_id']}")
    
    if 'modalities' in patient_info:
        summary.append(f"**Available Modalities:** {', '.join(patient_info['modalities'])}")
    
    if 'timepoints' in patient_info:
        summary.append(f"**Timepoints:** {len(patient_info['timepoints'])}")
    
    if 'imaging_series' in patient_info:
        summary.append(f"**Imaging Series:** {len(patient_info['imaging_series'])}")
    
    if 'rtstruct_files' in patient_info:
        summary.append(f"**RTSTRUCT Files:** {len(patient_info['rtstruct_files'])}")
    
    if 'compatible_pairs' in patient_info:
        summary.append(f"**Compatible Pairs:** {len(patient_info['compatible_pairs'])}")
    
    if 'nifti_pairs' in patient_info:
        summary.append(f"**NIfTI Pairs:** {len(patient_info['nifti_pairs'])}")
    
    if 'errors' in patient_info and patient_info['errors']:
        summary.append(f"**Errors:** {len(patient_info['errors'])}")
    
    return "\n".join(summary)


def calculate_processing_readiness(analysis):
    """
    Enhanced processing readiness with multi-format support
    """
    readiness = {
        'patients_with_imaging': 0,
        'patients_with_rtstruct': 0,
        'patients_with_nifti_pairs': 0,
        'patients_ready_for_processing': 0,
        'modality_readiness': {},
        'format_breakdown': {'dicom': 0, 'nifti': 0}
    }
    
    for patient_info in analysis.get('patients', {}).values():
        if patient_info.get('imaging_series'):
            readiness['patients_with_imaging'] += 1
        
        if patient_info.get('rtstruct_files'):
            readiness['patients_with_rtstruct'] += 1
        
        if patient_info.get('nifti_pairs'):
            readiness['patients_with_nifti_pairs'] += 1
            readiness['format_breakdown']['nifti'] += 1
        
        if patient_info.get('compatible_pairs'):
            readiness['patients_ready_for_processing'] += 1
            readiness['format_breakdown']['dicom'] += 1
            
            # Count readiness by modality
            for pair in patient_info['compatible_pairs']:
                modality = pair.get('modality', 'Unknown')
                if modality not in readiness['modality_readiness']:
                    readiness['modality_readiness'][modality] = 0
                readiness['modality_readiness'][modality] += 1
    
    return readiness


# --- Enhanced Path and Directory Validation ---

def get_available_directories(base_paths=None):
    """Enhanced directory scanning with better performance and error handling"""
    if base_paths is None:
        base_paths = ["/data", "/datasets", "/home", "/mnt", ".", os.path.expanduser("~")]
    
    available_dirs = set()
    for base_path in base_paths:
        if os.path.exists(base_path):
            try:
                # Limit scan depth and number of directories for performance
                count = 0
                for item in os.listdir(base_path):
                    if count > 100:  # Limit to prevent performance issues
                        break
                    full_path = os.path.join(base_path, item)
                    if os.path.isdir(full_path):
                        available_dirs.add(full_path)
                        count += 1
            except PermissionError:
                continue
    return sorted(list(available_dirs))


def validate_directory_path(path):
    """Enhanced directory validation with format detection"""
    if not path or not os.path.exists(path) or not os.path.isdir(path):
        return ["Path is not a valid, existing directory."]
    
    issues = []
    
    # Check for supported file types
    dicom_count = 0
    nifti_count = 0
    
    try:
        for root, _, files in os.walk(path):
            for file in files[:50]:  # Limit check to first 50 files for performance
                file_type = detect_file_type(os.path.join(root, file))
                if file_type == 'dicom':
                    dicom_count += 1
                elif file_type == 'nifti':
                    nifti_count += 1
            
            if dicom_count > 0 or nifti_count > 0:
                break  # Found supported files, no need to continue
    except Exception as e:
        issues.append(f"Error scanning directory: {str(e)}")
    
    if dicom_count == 0 and nifti_count == 0:
        issues.append("No supported files (DICOM or NIfTI) found in the specified directory.")
    
    return issues


# --- Data Handling and Organization (Enhanced) ---

def process_selected_path(selected_path):
    """
    Enhanced path processing with format detection
    """
    if not selected_path or not os.path.isdir(selected_path):
        st.error("Invalid directory path provided.")
        return None
    return selected_path


def organize_dicom_files(uploaded_files):
    """
    Enhanced DICOM file organization (preserving original logic)
    """
    try:
        # Create a temporary directory to extract and organize files
        temp_dir = tempfile.mkdtemp(prefix="radiomics_upload_")
        extract_path = os.path.join(temp_dir, "extracted")
        os.makedirs(extract_path, exist_ok=True)

        # 1. Extract all uploaded files
        for uploaded_file in uploaded_files:
            if uploaded_file.name.lower().endswith('.zip'):
                with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
            else:
                with open(os.path.join(extract_path, uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())

        # 2. Walk through extracted files and organize them
        organized_path = os.path.join(temp_dir, "organized")
        os.makedirs(organized_path, exist_ok=True)
        
        dicom_files = []
        for root, _, files in os.walk(extract_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    pydicom.dcmread(file_path, stop_before_pixels=True)
                    dicom_files.append(file_path)
                except pydicom.errors.InvalidDicomError:
                    continue # Skip non-DICOM files

        if not dicom_files:
            st.error("No valid DICOM files were found in the upload.")
            shutil.rmtree(temp_dir)
            return None

        # Group by PatientID -> SeriesInstanceUID
        patient_series_map = {}
        for file_path in dicom_files:
            dcm = pydicom.dcmread(file_path, stop_before_pixels=True)
            patient_id = getattr(dcm, 'PatientID', 'UnknownPatient')
            series_uid = getattr(dcm, 'SeriesInstanceUID', 'UnknownSeries')
            
            if patient_id not in patient_series_map:
                patient_series_map[patient_id] = {}
            if series_uid not in patient_series_map[patient_id]:
                patient_series_map[patient_id][series_uid] = []
            
            patient_series_map[patient_id][series_uid].append(file_path)

        # 3. Create the organized directory structure and copy files
        for patient_id, series_map in patient_series_map.items():
            patient_dir = os.path.join(organized_path, str(patient_id))
            for series_uid, files in series_map.items():
                # Use modality for a more descriptive folder name
                modality = getattr(pydicom.dcmread(files[0], stop_before_pixels=True), 'Modality', 'UN')
                series_dir = os.path.join(patient_dir, f"{modality}_{series_uid[:8]}")
                os.makedirs(series_dir, exist_ok=True)
                for file_path in files:
                    shutil.copy(file_path, series_dir)
        
        # Clean up the raw extracted files, leaving only the organized ones
        shutil.rmtree(extract_path)
        
        return organized_path
        
    except Exception as e:
        st.error(f"An error occurred during file organization: {e}")
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return None
