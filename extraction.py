# extraction.py
"""
This module contains functions for radiomics feature extraction using PyRadiomics.
"""

import os
import tempfile
import yaml
import pandas as pd
import numpy as np
import streamlit as st
from concurrent.futures import ProcessPoolExecutor, as_completed
import radiomics
from radiomics import featureextractor
import SimpleITK as sitk
import traceback
from pathlib import Path

def validate_mask_for_extraction(mask_path, patient_id):
    """
    Validates mask file before PyRadiomics extraction.
    
    Returns:
        tuple: (is_valid, error_message, mask_info)
    """
    try:
        # Load mask
        mask_sitk = sitk.ReadImage(mask_path)
        mask_array = sitk.GetArrayFromImage(mask_sitk)
        
        # Basic validation
        if mask_array.size == 0:
            return False, "Mask file is empty", {}
        
        # Check for any non-zero values
        unique_values = np.unique(mask_array)
        nonzero_count = np.sum(mask_array > 0)
        
        mask_info = {
            'unique_values': unique_values.tolist(),
            'nonzero_voxels': int(nonzero_count),
            'total_voxels': int(mask_array.size),
            'mask_shape': mask_array.shape,
            'mask_dtype': str(mask_array.dtype),
            'mask_min': float(np.min(mask_array)),
            'mask_max': float(np.max(mask_array)),
            'spacing': mask_sitk.GetSpacing(),
            'origin': mask_sitk.GetOrigin(),
            'size': mask_sitk.GetSize()
        }
        
        if nonzero_count == 0:
            return False, "Mask contains no positive values (all zeros)", mask_info
        
        # Check if mask values are appropriate for PyRadiomics
        if np.max(mask_array) < 0.5:
            return False, f"Mask values too small (max: {np.max(mask_array)}), may indicate data type issue", mask_info
        
        return True, "Mask validation passed", mask_info
        
    except Exception as e:
        return False, f"Mask validation failed: {str(e)}", {}


def preprocess_mask_for_radiomics(mask_path, output_path=None):
    """
    Preprocesses mask to ensure compatibility with PyRadiomics.
    
    Returns:
        str: Path to processed mask file
    """
    mask_sitk = sitk.ReadImage(mask_path)
    mask_array = sitk.GetArrayFromImage(mask_sitk)
    
    # Convert to binary mask (0 and 1)
    processed_array = (mask_array > 0).astype(np.uint8)
    
    # Create new SimpleITK image
    processed_mask = sitk.GetImageFromArray(processed_array)
    processed_mask.CopyInformation(mask_sitk)
    
    # Save processed mask
    if output_path is None:
        output_path = mask_path.replace('.nii.gz', '_processed.nii.gz')
    
    sitk.WriteImage(processed_mask, output_path)
    return output_path


def generate_pyradiomics_params(feature_classes=None, normalize_image=True, 
                               resample_pixel_spacing=False, pixel_spacing=None,
                               bin_width=25, interpolator='sitkBSpline', 
                               pad_distance=5, geometryTolerance=0.0001):
    """
    Generates PyRadiomics parameter configuration based on user inputs.
    """
    if feature_classes is None:
        feature_classes = {
            'firstorder': True,
            'shape': True,
            'glcm': True,
            'glrlm': True,
            'glszm': True,
            'ngtdm': True,
            'gldm': True
        }
    
    params = {
        'setting': {
            'binWidth': bin_width,
            'interpolator': interpolator,
            'padDistance': pad_distance,
            'geometryTolerance': geometryTolerance
        },
        'imageType': {
            'Original': {}
        },
        'featureClass': {}
    }
    
    # Add normalization if requested
    if normalize_image:
        params['setting']['normalize'] = True
        params['setting']['normalizeScale'] = 1
    
    # Add resampling if requested
    if resample_pixel_spacing and pixel_spacing:
        params['setting']['resampledPixelSpacing'] = [pixel_spacing, pixel_spacing, pixel_spacing]
    
    # Add enabled feature classes
    for feature_class, enabled in feature_classes.items():
        if enabled:
            params['featureClass'][feature_class] = []
    
    return params


def extract_features_single_patient(args):
    """
    Extract features for a single patient - designed for parallel processing.
    Args is a tuple of (patient_data, params_dict, patient_index, total_patients)
    """
    try:
        patient_data, params_dict, patient_index, total_patients = args
        
        patient_id = patient_data['patient_id']
        image_path = patient_data['image_path']
        mask_path = patient_data['mask_path']
        
        # Verify files exist
        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            error_msg = f"Missing files for {patient_id}: Image={os.path.exists(image_path)}, Mask={os.path.exists(mask_path)}"
            return {'patient_id': patient_id, 'error': error_msg}
        
        # Validate mask before extraction
        is_valid, validation_msg, mask_info = validate_mask_for_extraction(mask_path, patient_id)
        if not is_valid:
            error_msg = f"Mask validation failed for {patient_id}: {validation_msg}"
            detailed_error = f"{error_msg}\nMask info: {mask_info}"
            return {'patient_id': patient_id, 'error': detailed_error}
        
        # Preprocess mask if needed
        processed_mask_path = mask_path
        if mask_info['mask_max'] != 1.0 or mask_info['mask_dtype'] != 'uint8':
            try:
                processed_mask_path = preprocess_mask_for_radiomics(mask_path)
            except Exception as preprocess_error:
                error_msg = f"Mask preprocessing failed for {patient_id}: {str(preprocess_error)}"
                return {'patient_id': patient_id, 'error': error_msg}
        
        # Initialize extractor
        extractor = featureextractor.RadiomicsFeatureExtractor()
        
        # Configure extractor with parameters
        for key, value in params_dict.get('setting', {}).items():
            extractor.settings[key] = value
        
        # Enable feature classes
        extractor.disableAllFeatures()
        for feature_class in params_dict.get('featureClass', {}):
            extractor.enableFeatureClassByName(feature_class)
        
        # Extract features
        try:
            features = extractor.execute(image_path, processed_mask_path)
            
            # Convert to flat dictionary
            feature_dict = {'PatientID': patient_id}
            for key, value in features.items():
                if not key.startswith('diagnostics_'):
                    # Convert numpy types to Python native types
                    if isinstance(value, (np.integer, np.floating)):
                        value = value.item()
                    elif isinstance(value, np.ndarray):
                        value = value.tolist()
                    feature_dict[key] = value
            
            return feature_dict
            
        except Exception as extraction_error:
            if "No labels found" in str(extraction_error):
                detailed_error = f"PyRadiomics 'No labels found' error for {patient_id}. Mask info: {mask_info}"
                return {'patient_id': patient_id, 'error': detailed_error}
            else:
                error_msg = f"Feature extraction failed for {patient_id}: {str(extraction_error)}"
                return {'patient_id': patient_id, 'error': error_msg}
            
    except Exception as e:
        patient_id = args[0].get('patient_id', 'Unknown') if len(args) > 0 else 'Unknown'
        error_msg = f"Critical error processing {patient_id}: {str(e)}"
        return {'patient_id': patient_id, 'error': error_msg}


def extract_features_single_patient_sequential(patient_data, params_dict, patient_index, total_patients):
    """
    Extract features for a single patient with UI progress updates and enhanced debugging - for sequential processing.
    """
    try:
        patient_id = patient_data['patient_id']
        image_path = patient_data['image_path']
        mask_path = patient_data['mask_path']
        
        # Update progress if UI elements are available
        progress_bar = st.session_state.get('extraction_progress_bar')
        progress_text = st.session_state.get('extraction_progress_text')
        status_placeholder = st.session_state.get('extraction_status_placeholder')
        
        current_progress = (patient_index + 1) / total_patients
        
        if progress_bar:
            progress_bar.progress(current_progress)
        if progress_text:
            progress_text.text(f"Extracting features {patient_index + 1}/{total_patients}: {patient_id} ({current_progress*100:.1f}%)")
        if status_placeholder:
            status_placeholder.info(f"🔄 Extracting features for {patient_id}...")
        
        # Verify files exist
        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            error_msg = f"Missing files for {patient_id}: Image={os.path.exists(image_path)}, Mask={os.path.exists(mask_path)}"
            if status_placeholder:
                status_placeholder.error(f"❌ {error_msg}")
            return {'patient_id': patient_id, 'error': error_msg}
        
        # Comprehensive mask validation
        is_valid, validation_msg, mask_info = validate_mask_for_extraction(mask_path, patient_id)
        
        if not is_valid:
            error_msg = f"Mask validation failed for {patient_id}: {validation_msg}"
            detailed_error = f"{error_msg}\nMask info: {mask_info}"
            
            if status_placeholder:
                status_placeholder.error(f"❌ {error_msg}")
                # Show detailed debugging info
                with st.expander(f"🔍 Debug info for {patient_id}"):
                    st.json(mask_info)
            
            return {'patient_id': patient_id, 'error': detailed_error}
        
        # Log mask info for successful validation
        if status_placeholder:
            status_placeholder.info(f"✅ Mask validated for {patient_id}: {mask_info['nonzero_voxels']} segmented voxels")
        
        # Try to preprocess mask if it has unusual values
        processed_mask_path = mask_path
        if mask_info['mask_max'] != 1.0 or mask_info['mask_dtype'] != 'uint8':
            if status_placeholder:
                status_placeholder.info(f"🔧 Preprocessing mask for {patient_id} (converting to binary uint8)")
            try:
                processed_mask_path = preprocess_mask_for_radiomics(mask_path)
            except Exception as preprocess_error:
                error_msg = f"Mask preprocessing failed for {patient_id}: {str(preprocess_error)}"
                if status_placeholder:
                    status_placeholder.error(f"❌ {error_msg}")
                return {'patient_id': patient_id, 'error': error_msg}
        
        # Initialize extractor
        extractor = featureextractor.RadiomicsFeatureExtractor()
        
        # Configure extractor with parameters
        for key, value in params_dict.get('setting', {}).items():
            extractor.settings[key] = value
        
        # Enable feature classes
        extractor.disableAllFeatures()
        for feature_class in params_dict.get('featureClass', {}):
            extractor.enableFeatureClassByName(feature_class)
        
        # Extract features with the processed mask
        try:
            features = extractor.execute(image_path, processed_mask_path)
            
            # Convert to flat dictionary
            feature_dict = {'PatientID': patient_id}
            for key, value in features.items():
                if not key.startswith('diagnostics_'):
                    # Convert numpy types to Python native types
                    if isinstance(value, (np.integer, np.floating)):
                        value = value.item()
                    elif isinstance(value, np.ndarray):
                        value = value.tolist()
                    feature_dict[key] = value
            
            if status_placeholder:
                status_placeholder.success(f"✅ Successfully extracted features for {patient_id}")
            
            return feature_dict
            
        except Exception as extraction_error:
            # Enhanced error reporting for PyRadiomics errors
            if "No labels found" in str(extraction_error):
                detailed_error = f"PyRadiomics 'No labels found' error for {patient_id}. Mask info: {mask_info}"
                if status_placeholder:
                    status_placeholder.error(f"❌ No labels found for {patient_id}")
                    with st.expander(f"🔍 Debug info for {patient_id}"):
                        st.json(mask_info)
                        st.write("**Suggestions:**")
                        st.write("- Check if mask and image have matching dimensions")
                        st.write("- Verify mask contains positive integer values")
                        st.write("- Ensure proper spatial alignment")
                return {'patient_id': patient_id, 'error': detailed_error}
            else:
                error_msg = f"Feature extraction failed for {patient_id}: {str(extraction_error)}"
                if status_placeholder:
                    status_placeholder.error(f"❌ {error_msg}")
                return {'patient_id': patient_id, 'error': error_msg}
            
    except Exception as e:
        error_msg = f"Critical error processing {patient_data.get('patient_id', 'Unknown')}: {str(e)}"
        if status_placeholder:
            status_placeholder.error(f"❌ {error_msg}")
        return {'patient_id': patient_data.get('patient_id', 'Unknown'), 'error': error_msg}


def run_extraction(dataset_df, params, n_jobs=1):
    """
    Runs feature extraction on the dataset with progress tracking.
    """
    if dataset_df.empty:
        st.error("No data provided for feature extraction.")
        return pd.DataFrame()
    
    # Get UI elements from session state
    progress_bar = st.session_state.get('extraction_progress_bar')
    progress_text = st.session_state.get('extraction_progress_text')
    status_placeholder = st.session_state.get('extraction_status_placeholder')
    
    total_patients = len(dataset_df)
    successful_extractions = []
    failed_extractions = []
    
    try:
        if n_jobs == 1:
            # Sequential processing with detailed progress updates
            if status_placeholder:
                status_placeholder.info(f"🔄 Starting sequential feature extraction for {total_patients} patients...")
            
            for i, (_, patient_data) in enumerate(dataset_df.iterrows()):
                result = extract_features_single_patient_sequential(
                    patient_data.to_dict(), 
                    params, 
                    i, 
                    total_patients
                )
                
                if 'error' in result:
                    failed_extractions.append(result)
                else:
                    successful_extractions.append(result)
        else:
            # Parallel processing with basic progress tracking
            if status_placeholder:
                status_placeholder.info(f"🔄 Starting parallel extraction with {n_jobs} workers for {total_patients} patients...")
            
            # Prepare arguments for parallel processing
            args_list = []
            for i, (_, patient_data) in enumerate(dataset_df.iterrows()):
                args_list.append((patient_data.to_dict(), params, i, total_patients))
            
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                # Submit all jobs
                future_to_patient = {}
                for args in args_list:
                    future = executor.submit(extract_features_single_patient, args)
                    future_to_patient[future] = args[0]['patient_id']  # patient_id is in patient_data
                
                # Collect results as they complete
                completed_count = 0
                for future in as_completed(future_to_patient):
                    completed_count += 1
                    current_progress = completed_count / total_patients
                    
                    if progress_bar:
                        progress_bar.progress(current_progress)
                    if progress_text:
                        progress_text.text(f"Completed {completed_count}/{total_patients} patients ({current_progress*100:.1f}%)")
                    if status_placeholder:
                        status_placeholder.info(f"🔄 Processed {completed_count}/{total_patients} patients...")
                    
                    try:
                        result = future.result()
                        if 'error' in result:
                            failed_extractions.append(result)
                        else:
                            successful_extractions.append(result)
                    except Exception as e:
                        patient_id = future_to_patient[future]
                        failed_extractions.append({
                            'patient_id': patient_id, 
                            'error': f"Parallel processing error: {str(e)}"
                        })
        
        # Final progress update
        if progress_bar:
            progress_bar.progress(1.0)
        if progress_text:
            progress_text.text(f"Extraction complete! {len(successful_extractions)}/{total_patients} patients processed successfully")
        
        # Create results summary
        if successful_extractions:
            features_df = pd.DataFrame(successful_extractions)
            
            if status_placeholder:
                if failed_extractions:
                    status_placeholder.warning(f"⚠️ Extraction complete with some failures: {len(successful_extractions)} successful, {len(failed_extractions)} failed")
                else:
                    status_placeholder.success(f"🎉 Feature extraction complete! All {len(successful_extractions)} patients processed successfully")
            
            # Store extraction summary in session state
            st.session_state['extraction_summary'] = {
                'total_patients': total_patients,
                'successful_extractions': len(successful_extractions),
                'failed_extractions': failed_extractions,
                'total_features': len(features_df.columns) - 1  # Exclude PatientID
            }
            
            return features_df
        else:
            if status_placeholder:
                status_placeholder.error(f"❌ Feature extraction failed for all patients")
            
            # Store extraction summary even for complete failure
            st.session_state['extraction_summary'] = {
                'total_patients': total_patients,
                'successful_extractions': 0,
                'failed_extractions': failed_extractions,
                'total_features': 0
            }
            
            return pd.DataFrame()
            
    except Exception as e:
        if status_placeholder:
            status_placeholder.error(f"❌ Critical error during feature extraction: {str(e)}")
        st.error(f"Feature extraction failed with error: {str(e)}")
        with st.expander("🔍 Error Details"):
            st.code(traceback.format_exc())
        
        # Store extraction summary for critical failure
        st.session_state['extraction_summary'] = {
            'total_patients': total_patients,
            'successful_extractions': len(successful_extractions),
            'failed_extractions': failed_extractions + [{'patient_id': 'System', 'error': str(e)}],
            'total_features': 0
        }
        
        return pd.DataFrame()


def validate_extraction_parameters(params):
    """
    Validates PyRadiomics parameters before extraction.
    """
    validation_errors = []
    
    if not isinstance(params, dict):
        validation_errors.append("Parameters must be a dictionary")
        return validation_errors
    
    # Check required sections
    if 'setting' not in params:
        validation_errors.append("Missing 'setting' section in parameters")
    
    if 'featureClass' not in params:
        validation_errors.append("Missing 'featureClass' section in parameters")
    
    # Check if at least one feature class is enabled
    if 'featureClass' in params:
        if not params['featureClass']:
            validation_errors.append("No feature classes enabled")
    
    # Validate specific settings
    if 'setting' in params:
        settings = params['setting']
        
        # Validate bin width
        if 'binWidth' in settings:
            try:
                bin_width = float(settings['binWidth'])
                if bin_width <= 0:
                    validation_errors.append("Bin width must be positive")
            except (ValueError, TypeError):
                validation_errors.append("Bin width must be a number")
        
        # Validate interpolator
        valid_interpolators = ['sitkBSpline', 'sitkLinear', 'sitkNearestNeighbor']
        if 'interpolator' in settings:
            if settings['interpolator'] not in valid_interpolators:
                validation_errors.append(f"Interpolator must be one of: {valid_interpolators}")
    
    return validation_errors


def get_feature_extraction_info():
    """
    Returns information about the current PyRadiomics installation and available features.
    """
    try:
        info = {
            'radiomics_version': radiomics.__version__,
            'available_feature_classes': [
                'firstorder', 'shape', 'glcm', 'glrlm', 
                'glszm', 'ngtdm', 'gldm'
            ],
            'available_interpolators': [
                'sitkBSpline', 'sitkLinear', 'sitkNearestNeighbor'
            ],
            'default_settings': {
                'binWidth': 25,
                'interpolator': 'sitkBSpline',
                'padDistance': 5,
                'geometryTolerance': 0.0001
            }
        }
        return info
    except Exception as e:
        return {'error': str(e)}
