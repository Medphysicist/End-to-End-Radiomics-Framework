# utils.py
"""
Utility functions for the Enhanced Radiomics Framework.
Updated: 2025-11-03 22:32:37 UTC by Medphysicist

Features:
- Session state management with multi-ROI tracking
- Progress tracking and ETA calculations
- File type detection and validation
- NIfTI file handling and organization
- ROI categorization and management (NEW - Multi-ROI support)
- Cleanup and resource management
- System resource monitoring

Multi-ROI Enhancements:
- ROI categorization (Targets, OARs, Other structures)
- Extended ROI naming patterns for better classification
- Multi-ROI session tracking
- ROI availability analysis across patients/series
"""

import os
import shutil
import tempfile
import time
import psutil
import streamlit as st
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import re


# =============================================================================
# SESSION STATE MANAGEMENT (Multi-ROI Enhanced)
# =============================================================================

def initialize_session_state():
    """
    Initialize Streamlit session state with all required variables.
    Enhanced with multi-ROI tracking capabilities.
    """
    # Original session state variables
    if 'uploaded_data_path' not in st.session_state:
        st.session_state.uploaded_data_path = None
    
    if 'all_contours' not in st.session_state:
        st.session_state.all_contours = []
    
    if 'patient_contour_data' not in st.session_state:
        st.session_state.patient_contour_data = {}
    
    if 'patient_status' not in st.session_state:
        st.session_state.patient_status = {}
    
    if 'dataset_df' not in st.session_state:
        st.session_state.dataset_df = None
    
    if 'preprocessing_done' not in st.session_state:
        st.session_state.preprocessing_done = False
    
    if 'features_df' not in st.session_state:
        st.session_state.features_df = None
    
    if 'extraction_done' not in st.session_state:
        st.session_state.extraction_done = False
    
    if 'pyradiomics_params' not in st.session_state:
        st.session_state.pyradiomics_params = None
    
    if 'selected_modality' not in st.session_state:
        st.session_state.selected_modality = 'CT'
    
    if 'temp_output_dir' not in st.session_state:
        st.session_state.temp_output_dir = None
    
    if 'output_directory' not in st.session_state:
        st.session_state.output_directory = 'output'
    
    # Enhanced multi-modality support
    if 'selected_modalities' not in st.session_state:
        st.session_state.selected_modalities = ['CT']
    
    if 'available_modalities' not in st.session_state:
        st.session_state.available_modalities = set()
    
    if 'multi_series_mode' not in st.session_state:
        st.session_state.multi_series_mode = False
    
    if 'longitudinal_data' not in st.session_state:
        st.session_state.longitudinal_data = {}
    
    if 'input_format' not in st.session_state:
        st.session_state.input_format = 'dicom'
    
    # ✅ NEW: Multi-ROI tracking
    if 'selected_rois' not in st.session_state:
        st.session_state.selected_rois = []
    
    if 'roi_processing_mode' not in st.session_state:
        st.session_state.roi_processing_mode = 'single'  # 'single', 'multiple', 'all'
    
    if 'roi_categories' not in st.session_state:
        st.session_state.roi_categories = {
            'targets': [],
            'oars': [],
            'other': []
        }
    
    if 'roi_availability' not in st.session_state:
        st.session_state.roi_availability = {}  # patient_id -> [available_rois]
    
    if 'multi_roi_session_id' not in st.session_state:
        st.session_state.multi_roi_session_id = None
    
    # UI progress tracking elements
    if 'ui_progress_bar' not in st.session_state:
        st.session_state.ui_progress_bar = None
    
    if 'ui_progress_text' not in st.session_state:
        st.session_state.ui_progress_text = None
    
    if 'ui_status_placeholder' not in st.session_state:
        st.session_state.ui_status_placeholder = None
    
    # Analysis results
    if 'outcome_df' not in st.session_state:
        st.session_state.outcome_df = None
    
    if 'merged_df' not in st.session_state:
        st.session_state.merged_df = None
    
    if 'univariate_results' not in st.session_state:
        st.session_state.univariate_results = None
    
    if 'lasso_features' not in st.session_state:
        st.session_state.lasso_features = None
    
    if 'correlation_heatmap' not in st.session_state:
        st.session_state.correlation_heatmap = None


def register_cleanup():
    """Register cleanup handlers for temporary files."""
    import atexit
    
    def cleanup_temp_files():
        """Clean up temporary directories on exit."""
        if 'temp_output_dir' in st.session_state:
            temp_dir = st.session_state.temp_output_dir
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except Exception:
                    pass
    
    atexit.register(cleanup_temp_files)


# =============================================================================
# PROGRESS TRACKING (Enhanced with ETA)
# =============================================================================

class ProgressTracker:
    """
    Enhanced progress tracker with ETA calculation and detailed logging.
    Supports multi-ROI processing with per-ROI progress tracking.
    """
    
    def __init__(self, total_items: int, description: str = "Processing"):
        """
        Initialize progress tracker.
        
        Args:
            total_items: Total number of items to process
            description: Description of the task being tracked
        """
        self.total_items = total_items
        self.description = description
        self.current_item = 0
        self.start_time = time.time()
        self.item_times = []
        
        print(f"\n{'='*60}")
        print(f"Starting: {description}")
        print(f"Total items: {total_items}")
        print(f"{'='*60}\n")
    
    def update(self, current: int, message: str = ""):
        """
        Update progress and display status.
        
        Args:
            current: Current item index (0-based)
            message: Optional status message
        """
        self.current_item = current + 1
        
        # Calculate progress percentage
        progress_pct = (self.current_item / self.total_items) * 100
        
        # Calculate ETA
        elapsed_time = time.time() - self.start_time
        if self.current_item > 0:
            avg_time_per_item = elapsed_time / self.current_item
            remaining_items = self.total_items - self.current_item
            eta_seconds = avg_time_per_item * remaining_items
            eta_str = self._format_time(eta_seconds)
        else:
            eta_str = "calculating..."
        
        # Display progress
        status_line = f"[{self.current_item}/{self.total_items}] {progress_pct:.1f}% - ETA: {eta_str}"
        if message:
            status_line += f" - {message}"
        
        print(status_line)
        
        # Update UI progress if available
        if st.session_state.get('ui_progress_bar'):
            st.session_state['ui_progress_bar'].progress(progress_pct / 100)
        
        if st.session_state.get('ui_progress_text'):
            st.session_state['ui_progress_text'].text(status_line)
    
    def complete(self, message: str = ""):
        """
        Mark task as complete and display summary.
        
        Args:
            message: Optional completion message
        """
        elapsed_time = time.time() - self.start_time
        elapsed_str = self._format_time(elapsed_time)
        
        print(f"\n{'='*60}")
        print(f"✅ Complete: {self.description}")
        print(f"Total time: {elapsed_str}")
        print(f"Items processed: {self.current_item}/{self.total_items}")
        if message:
            print(f"Status: {message}")
        print(f"{'='*60}\n")
        
        # Update UI
        if st.session_state.get('ui_progress_bar'):
            st.session_state['ui_progress_bar'].progress(1.0)
        
        if st.session_state.get('ui_progress_text'):
            completion_msg = f"✅ {self.description} complete! ({elapsed_str})"
            if message:
                completion_msg += f" - {message}"
            st.session_state['ui_progress_text'].text(completion_msg)
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds into human-readable time string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"


# =============================================================================
# FILE TYPE DETECTION AND VALIDATION
# =============================================================================

def detect_file_type(file_path: str) -> str:
    """
    Detect file type from extension.
    
    Args:
        file_path: Path to file
    
    Returns:
        File type: 'nifti', 'dicom', or 'unknown'
    """
    file_path = str(file_path).lower()
    
    if file_path.endswith(('.nii', '.nii.gz')):
        return 'nifti'
    elif file_path.endswith(('.dcm', '.ima', '.dicom')):
        return 'dicom'
    elif '.' not in os.path.basename(file_path):
        # DICOM files sometimes have no extension
        try:
            import pydicom
            pydicom.dcmread(file_path, stop_before_pixels=True)
            return 'dicom'
        except Exception:
            pass
    
    return 'unknown'


def validate_uploaded_files(uploaded_files) -> List[str]:
    """
    Validate uploaded files for common issues.
    
    Args:
        uploaded_files: List of uploaded file objects from Streamlit
    
    Returns:
        List of validation issues (empty if all valid)
    """
    issues = []
    
    if not uploaded_files:
        issues.append("No files uploaded")
        return issues
    
    # Check file sizes
    total_size = sum(file.size for file in uploaded_files)
    max_size = 5 * 1024 * 1024 * 1024  # 5 GB
    
    if total_size > max_size:
        issues.append(f"Total file size ({total_size / (1024**3):.2f} GB) exceeds maximum (5 GB)")
    
    # Check file types
    valid_extensions = {'.dcm', '.ima', '.dicom', '.nii', '.nii.gz', '.zip'}
    invalid_files = []
    
    for file in uploaded_files:
        file_ext = Path(file.name).suffix.lower()
        if file_ext not in valid_extensions and not file.name.lower().endswith('.nii.gz'):
            invalid_files.append(file.name)
    
    if invalid_files:
        issues.append(f"Invalid file types: {', '.join(invalid_files[:5])}")
        if len(invalid_files) > 5:
            issues.append(f"... and {len(invalid_files) - 5} more")
    
    return issues


def validate_nifti_pair(image_path: str, mask_path: str) -> Tuple[bool, str]:
    """
    Validate NIfTI image/mask pair for compatibility.
    
    Args:
        image_path: Path to image file
        mask_path: Path to mask file
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        if not os.path.exists(image_path):
            return False, f"Image file not found: {image_path}"
        
        if not os.path.exists(mask_path):
            return False, f"Mask file not found: {mask_path}"
        
        # Read images
        image = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(mask_path)
        
        # Check dimensions
        if image.GetSize() != mask.GetSize():
            return False, f"Dimension mismatch: image {image.GetSize()} vs mask {mask.GetSize()}"
        
        # Check spacing (allow small differences)
        image_spacing = np.array(image.GetSpacing())
        mask_spacing = np.array(mask.GetSpacing())
        spacing_diff = np.abs(image_spacing - mask_spacing)
        
        if np.any(spacing_diff > 0.1):
            return False, f"Spacing mismatch: image {image_spacing} vs mask {mask_spacing}"
        
        # Check mask values
        mask_array = sitk.GetArrayFromImage(mask)
        unique_values = np.unique(mask_array)
        
        if len(unique_values) == 1 and unique_values[0] == 0:
            return False, "Mask contains only zeros (empty mask)"
        
        # Check for reasonable mask values (should be binary or multi-label integers)
        if not np.all((mask_array >= 0) & (mask_array == mask_array.astype(int))):
            return False, "Mask contains invalid values (should be non-negative integers)"
        
        return True, "Valid pair"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"


# =============================================================================
# NIFTI FILE ORGANIZATION
# =============================================================================

def organize_nifti_files(uploaded_files):
    """
    Organize uploaded NIfTI files into a structured directory.
    
    Args:
        uploaded_files: List of uploaded file objects
    
    Returns:
        Path to organized directory, or None if failed
    """
    try:
        temp_dir = tempfile.mkdtemp(prefix="radiomics_nifti_upload_")
        
        # Extract files
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            
            # Handle ZIP files
            if uploaded_file.name.lower().endswith('.zip'):
                import zipfile
                with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
            else:
                # Regular file
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
        
        # Organize by patient ID (inferred from filename patterns)
        organized_dir = os.path.join(temp_dir, "organized")
        os.makedirs(organized_dir, exist_ok=True)
        
        # Find all NIfTI files
        nifti_files = []
        for root, _, files in os.walk(temp_dir):
            for file in files:
                if detect_file_type(file) == 'nifti':
                    nifti_files.append(os.path.join(root, file))
        
        # Group files by patient ID (extracted from filename)
        patient_files = {}
        for file_path in nifti_files:
            patient_id = extract_patient_id_from_filename(Path(file_path).name)
            if patient_id not in patient_files:
                patient_files[patient_id] = []
            patient_files[patient_id].append(file_path)
        
        # Copy files to organized structure
        for patient_id, files in patient_files.items():
            patient_dir = os.path.join(organized_dir, patient_id)
            os.makedirs(patient_dir, exist_ok=True)
            
            for file_path in files:
                shutil.copy(file_path, patient_dir)
        
        return organized_dir
        
    except Exception as e:
        print(f"Error organizing NIfTI files: {e}")
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return None


def extract_patient_id_from_filename(filename: str) -> str:
    """
    Extract patient ID from filename using common patterns.
    
    Args:
        filename: Name of file
    
    Returns:
        Extracted patient ID, or "Unknown" if not found
    """
    # Remove extension
    name = Path(filename).stem
    if name.endswith('.nii'):
        name = name[:-4]
    
    # Common patterns
    patterns = [
        r'(Patient[-_]?\d+)',
        r'(P\d+)',
        r'(SUB[-_]?\d+)',
        r'(\d{3,})',  # 3+ digit number
    ]
    
    for pattern in patterns:
        match = re.search(pattern, name, re.IGNORECASE)
        if match:
            return match.group(1)
    
    # Fallback: use first part before underscore/dash
    parts = re.split(r'[-_]', name)
    if parts:
        return parts[0]
    
    return "Unknown"


# =============================================================================
# ROI CATEGORIZATION AND MANAGEMENT (NEW - Multi-ROI Support)
# =============================================================================

def categorize_contours(contour_names: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """
    Categorize contours into Targets, OARs, and Other structures.
    Original version with basic patterns.
    
    Args:
        contour_names: List of ROI names
    
    Returns:
        Tuple of (targets, oars, other)
    """
    targets = []
    oars = []
    other = []
    
    target_keywords = ['gtv', 'ctv', 'ptv', 'tumor', 'tumour', 'target', 'lesion', 'mass']
    oar_keywords = [
        'brain', 'brainstem', 'spinal', 'cord', 'heart', 'lung', 'liver', 'kidney',
        'bladder', 'rectum', 'bowel', 'stomach', 'parotid', 'submandibular',
        'eye', 'lens', 'optic', 'chiasm', 'nerve', 'cochlea', 'mandible',
        'esophagus', 'trachea', 'larynx', 'thyroid'
    ]
    
    for contour in contour_names:
        contour_lower = contour.lower()
        
        if any(keyword in contour_lower for keyword in target_keywords):
            targets.append(contour)
        elif any(keyword in contour_lower for keyword in oar_keywords):
            oars.append(contour)
        else:
            other.append(contour)
    
    return targets, oars, other


def categorize_contours_extended(contour_names: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """
    Enhanced contour categorization with more comprehensive patterns.
    NEW: Extended for better multi-ROI classification.
    
    Args:
        contour_names: List of ROI names
    
    Returns:
        Tuple of (targets, oars, other)
    """
    targets = []
    oars = []
    other = []
    
    # Extended target keywords
    target_keywords = [
        'gtv', 'ctv', 'ptv', 'itv',  # Standard target volumes
        'tumor', 'tumour', 'target', 'lesion', 'mass', 'nodal',  # General tumor
        'primary', 'boost', 'cavity', 'bed',  # Treatment areas
        'lymph.*node', 'ln', 'metastasis', 'met'  # Nodal/metastatic disease
    ]
    
    # Extended OAR keywords with anatomical regions
    oar_keywords = {
        'cns': ['brain', 'brainstem', 'spinal.*cord', 'spinal.*canal', 'cerebellum', 'pons', 'medulla'],
        'head_neck': ['parotid', 'submandibular', 'sublingual', 'pharynx', 'larynx', 'glottis', 
                     'oral.*cavity', 'tongue', 'lips', 'mandible', 'tmj', 'masseter',
                     'buccal', 'soft.*palate', 'hard.*palate'],
        'vision': ['eye', 'lens', 'retina', 'optic.*nerve', 'optic.*chiasm', 'cornea', 'lacrimal'],
        'hearing': ['cochlea', 'ear', 'vestibular', 'acoustic'],
        'thorax': ['lung', 'heart', 'great.*vessel', 'aorta', 'vena.*cava', 'pulmonary',
                  'esophagus', 'trachea', 'bronch', 'rib', 'clavicle', 'sternum'],
        'abdomen': ['liver', 'stomach', 'bowel', 'duodenum', 'jejunum', 'ileum', 
                   'colon', 'sigmoid', 'cecum', 'spleen', 'pancreas', 'gallbladder'],
        'pelvis': ['bladder', 'rectum', 'anal.*canal', 'prostate', 'seminal.*vesicle',
                  'uterus', 'cervix', 'vagina', 'ovary', 'testis', 'penis'],
        'urinary': ['kidney', 'ureter', 'urethra'],
        'bone': ['femur', 'femoral.*head', 'acetabulum', 'pelvis', 'sacrum', 'coccyx',
                'vertebra', 'vertebrae', 'humerus', 'scapula'],
        'vascular': ['carotid', 'jugular', 'subclavian', 'iliac', 'femoral.*artery'],
        'glandular': ['thyroid', 'pituitary', 'adrenal', 'thymus'],
        'skin': ['skin', 'dermis', 'epidermis'],
        'muscle': ['muscle', 'constrictor', 'pterygoid', 'temporal']
    }
    
    # Flatten OAR keywords
    all_oar_keywords = []
    for region_keywords in oar_keywords.values():
        all_oar_keywords.extend(region_keywords)
    
    for contour in contour_names:
        contour_lower = contour.lower().strip()
        
        # Check for targets (use regex for flexible matching)
        is_target = False
        for keyword in target_keywords:
            if re.search(keyword, contour_lower):
                is_target = True
                break
        
        if is_target:
            targets.append(contour)
            continue
        
        # Check for OARs
        is_oar = False
        for keyword in all_oar_keywords:
            if re.search(keyword, contour_lower):
                is_oar = True
                break
        
        if is_oar:
            oars.append(contour)
        else:
            other.append(contour)
    
    return sorted(targets), sorted(oars), sorted(other)


def get_available_modalities_extended(patient_data: Dict) -> Set[str]:
    """
    Extract all available modalities from patient data.
    
    Args:
        patient_data: Dictionary containing patient series data
    
    Returns:
        Set of available modality strings
    """
    modalities = set()
    
    for patient_id, data in patient_data.items():
        if isinstance(data, dict):
            if 'available_modalities' in data:
                modalities.update(data['available_modalities'])
            
            if 'series_data' in data:
                modalities.update(data['series_data'].keys())
    
    return modalities


def analyze_roi_availability(patient_contour_data: Dict, longitudinal_data: Dict = None) -> Dict:
    """
    Analyze ROI availability across patients and series.
    NEW: For multi-ROI processing - helps identify which ROIs are available where.
    
    Args:
        patient_contour_data: Dictionary of patient_id -> list of contours
        longitudinal_data: Optional dictionary with per-series contour information
    
    Returns:
        Dictionary with availability statistics
    """
    analysis = {
        'total_patients': len(patient_contour_data),
        'total_unique_rois': set(),
        'roi_patient_count': {},  # roi_name -> count of patients having it
        'roi_series_count': {},   # roi_name -> count of series having it (if longitudinal)
        'patients_per_roi': {},   # roi_name -> list of patient_ids
        'series_per_roi': {}      # roi_name -> list of (patient_id, series_info) tuples
    }
    
    # Analyze patient-level availability
    for patient_id, contours in patient_contour_data.items():
        for contour in contours:
            analysis['total_unique_rois'].add(contour)
            
            if contour not in analysis['roi_patient_count']:
                analysis['roi_patient_count'][contour] = 0
                analysis['patients_per_roi'][contour] = []
            
            analysis['roi_patient_count'][contour] += 1
            analysis['patients_per_roi'][contour].append(patient_id)
    
    # Analyze series-level availability (if longitudinal data provided)
    if longitudinal_data:
        for patient_id, patient_long_data in longitudinal_data.items():
            compatible_pairs = patient_long_data.get('compatible_pairs', [])
            
            for pair in compatible_pairs:
                contours = pair.get('contours', [])
                series_info = {
                    'series_uid': pair.get('series_uid', ''),
                    'modality': pair.get('modality', ''),
                    'timepoint': pair.get('timepoint', '')
                }
                
                for contour in contours:
                    if contour not in analysis['roi_series_count']:
                        analysis['roi_series_count'][contour] = 0
                        analysis['series_per_roi'][contour] = []
                    
                    analysis['roi_series_count'][contour] += 1
                    analysis['series_per_roi'][contour].append((patient_id, series_info))
    
    # Convert set to sorted list for easier use
    analysis['total_unique_rois'] = sorted(list(analysis['total_unique_rois']))
    
    return analysis


def find_common_rois_across_patients(patient_contour_data: Dict, min_availability: float = 0.8) -> List[str]:
    """
    Find ROIs that are available in most patients.
    Useful for multi-ROI processing - helps select ROIs that will work for most patients.
    
    Args:
        patient_contour_data: Dictionary of patient_id -> list of contours
        min_availability: Minimum fraction of patients that must have the ROI (0.0 to 1.0)
    
    Returns:
        List of commonly available ROI names
    """
    if not patient_contour_data:
        return []
    
    roi_counts = {}
    total_patients = len(patient_contour_data)
    
    for contours in patient_contour_data.values():
        for contour in contours:
            roi_counts[contour] = roi_counts.get(contour, 0) + 1
    
    min_patient_count = int(total_patients * min_availability)
    common_rois = [roi for roi, count in roi_counts.items() if count >= min_patient_count]
    
    return sorted(common_rois)


def generate_roi_summary_report(patient_contour_data: Dict, longitudinal_data: Dict = None) -> str:
    """
    Generate a text summary report of ROI availability.
    Useful for displaying in UI.
    
    Args:
        patient_contour_data: Dictionary of patient_id -> list of contours
        longitudinal_data: Optional longitudinal data
    
    Returns:
        Formatted text summary
    """
    analysis = analyze_roi_availability(patient_contour_data, longitudinal_data)
    
    report_lines = [
        "="*60,
        "ROI AVAILABILITY SUMMARY",
        "="*60,
        f"Total Patients: {analysis['total_patients']}",
        f"Total Unique ROIs: {len(analysis['total_unique_rois'])}",
        "",
        "Top 10 Most Available ROIs:",
        "-"*60
    ]
    
    # Sort ROIs by availability
    roi_availability = [(roi, count) for roi, count in analysis['roi_patient_count'].items()]
    roi_availability.sort(key=lambda x: x[1], reverse=True)
    
    for i, (roi, count) in enumerate(roi_availability[:10], 1):
        availability_pct = (count / analysis['total_patients']) * 100
        report_lines.append(f"{i:2d}. {roi:30s} - {count:3d} patients ({availability_pct:5.1f}%)")
    
    if longitudinal_data:
        report_lines.extend([
            "",
            "Series-Level Availability:",
            "-"*60
        ])
        
        total_series = sum(len(ld.get('compatible_pairs', [])) for ld in longitudinal_data.values())
        report_lines.append(f"Total Series: {total_series}")
        
        # Top ROIs by series count
        roi_series_sorted = [(roi, count) for roi, count in analysis['roi_series_count'].items()]
        roi_series_sorted.sort(key=lambda x: x[1], reverse=True)
        
        for i, (roi, count) in enumerate(roi_series_sorted[:5], 1):
            series_pct = (count / total_series) * 100 if total_series > 0 else 0
            report_lines.append(f"{i:2d}. {roi:30s} - {count:3d} series ({series_pct:5.1f}%)")
    
    report_lines.append("="*60)
    
    return "\n".join(report_lines)


# =============================================================================
# SYSTEM RESOURCE MONITORING
# =============================================================================

def check_system_resources() -> Dict:
    """
    Check available system resources.
    
    Returns:
        Dictionary with system resource information
    """
    try:
        memory = psutil.virtual_memory()
        cpu_count = psutil.cpu_count()
        
        return {
            'available_ram_gb': memory.available / (1024**3),
            'total_ram_gb': memory.total / (1024**3),
            'ram_percent_used': memory.percent,
            'cpu_count': cpu_count,
            'cpu_percent': psutil.cpu_percent(interval=0.1)
        }
    except Exception as e:
        return {
            'available_ram_gb': 0,
            'total_ram_gb': 0,
            'ram_percent_used': 0,
            'cpu_count': 1,
            'cpu_percent': 0,
            'error': str(e)
        }


def estimate_memory_requirement(num_patients: int, roi_count: int = 1, 
                               modality_count: int = 1) -> float:
    """
    Estimate memory requirements for processing.
    Useful for multi-ROI processing planning.
    
    Args:
        num_patients: Number of patients to process
        roi_count: Number of ROIs per patient
        modality_count: Number of modalities per patient
    
    Returns:
        Estimated memory requirement in GB
    """
    # Rough estimates based on typical medical imaging data
    avg_image_size_mb = 50  # Average CT/MR image size
    avg_mask_size_mb = 10   # Average mask size
    processing_overhead_factor = 3  # Processing requires multiple copies
    
    base_requirement = (avg_image_size_mb + avg_mask_size_mb) * num_patients
    multi_roi_factor = roi_count * modality_count
    
    total_mb = base_requirement * multi_roi_factor * processing_overhead_factor
    total_gb = total_mb / 1024
    
    return total_gb


# =============================================================================
# MULTI-ROI SESSION MANAGEMENT
# =============================================================================

def create_multi_roi_session_id() -> str:
    """
    Create unique session ID for multi-ROI processing.
    Helps track which results belong to which multi-ROI processing run.
    
    Returns:
        Unique session ID string
    """
    import uuid
    timestamp = int(time.time())
    unique_id = str(uuid.uuid4())[:8]
    return f"multi_roi_{timestamp}_{unique_id}"


def save_multi_roi_session_info(session_id: str, roi_list: List[str], 
                                patient_count: int, series_count: int = 0):
    """
    Save information about current multi-ROI processing session.
    
    Args:
        session_id: Unique session ID
        roi_list: List of ROIs being processed
        patient_count: Number of patients
        series_count: Number of series (for multi-series mode)
    """
    session_info = {
        'session_id': session_id,
        'timestamp': time.time(),
        'roi_list': roi_list,
        'roi_count': len(roi_list),
        'patient_count': patient_count,
        'series_count': series_count,
        'total_combinations': patient_count * len(roi_list) * max(1, series_count)
    }
    
    st.session_state['multi_roi_session_info'] = session_info
    
    return session_info


# =============================================================================
# HELPER FUNCTIONS FOR MULTI-ROI RESULT AGGREGATION
# =============================================================================

def aggregate_multi_roi_results(results_list: List[Tuple]) -> Tuple:
    """
    Aggregate results from multiple ROI processing calls.
    Used by UI to combine results when processing multiple ROIs.
    
    Args:
        results_list: List of (dataframe, summary_dict) tuples
    
    Returns:
        Combined (dataframe, summary_dict)
    """
    import pandas as pd
    
    all_dfs = []
    combined_summary = {
        'total_patients': 0,
        'successful_patients': 0,
        'failed_patients': {},
        'recovery_statistics': {},
        'roi_count': len(results_list)
    }
    
    for df, summary in results_list:
        if isinstance(df, pd.DataFrame) and not df.empty:
            all_dfs.append(df)
        
        if isinstance(summary, dict):
            combined_summary['total_patients'] = max(
                combined_summary['total_patients'],
                summary.get('total_patients', 0)
            )
            combined_summary['successful_patients'] += summary.get('successful_patients', 0)
            
            # Merge failed patients
            for pid, info in summary.get('failed_patients', {}).items():
                if pid not in combined_summary['failed_patients']:
                    combined_summary['failed_patients'][pid] = info
            
            # Merge recovery statistics
            for key, val in summary.get('recovery_statistics', {}).items():
                combined_summary['recovery_statistics'][key] = \
                    combined_summary['recovery_statistics'].get(key, 0) + val
    
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
    else:
        combined_df = pd.DataFrame()
    
    return combined_df, combined_summary


# =============================================================================
# END OF FILE
# =============================================================================
