# processing.py
"""
This module contains all functions related to data loading, validation,
organization, and pre-processing. It prepares the raw DICOM data
for the feature extraction step.
"""

import os
import shutil
import tempfile
import zipfile
import pydicom
import SimpleITK as sitk
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from rt_utils import RTStructBuilder

# --- Path and Directory Validation ---

def get_available_directories(base_paths=None):
    """Scans common system locations for potential data directories."""
    if base_paths is None:
        base_paths = ["/data", "/datasets", "/home", "/mnt", ".", os.path.expanduser("~")]
    
    available_dirs = set()
    for base_path in base_paths:
        if os.path.exists(base_path):
            try:
                # Limit scan to the first level of subdirectories for performance
                for item in os.listdir(base_path):
                    full_path = os.path.join(base_path, item)
                    if os.path.isdir(full_path):
                        available_dirs.add(full_path)
            except PermissionError:
                continue
    return sorted(list(available_dirs))


def validate_directory_path(path):
    """Validates if a given path is a directory and contains DICOM files."""
    if not path or not os.path.exists(path) or not os.path.isdir(path):
        return ["Path is not a valid, existing directory."]
    
    # Quick check for DICOM files
    for _, _, files in os.walk(path):
        for file in files:
            if file.lower().endswith('.dcm'):
                return [] # Found at least one, good enough for a basic check
    
    return ["No DICOM (.dcm) files found in the specified directory."]


# --- Data Handling and Organization ---

def process_selected_path(selected_path):
    """
    Simply validates the path and returns it if valid.
    The main logic will then use this path directly.
    """
    if not selected_path or not os.path.isdir(selected_path):
        st.error("Invalid directory path provided.")
        return None
    return selected_path


def organize_dicom_files(uploaded_files):
    """
    Extracts, organizes, and structures uploaded DICOM files by PatientID and SeriesUID.
    This function handles both individual files and ZIP archives.
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


# --- Contour Scanning and Pre-processing to NIfTI ---

def scan_uploaded_data_for_contours(data_path, selected_modality='CT'):
    """
    Scans for contours using sequential processing.
    It explicitly finds RTSTRUCTs and Image Series, then finds a compatible pair.
    Now includes modality filtering.
    """
    all_contours = set()
    patient_contour_data = {}
    patient_status = {}
    available_modalities = set()

    if not data_path or not os.path.isdir(data_path):
        return [], {}, {}, list(available_modalities)

    patient_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]

    with st.status("Scanning patient data...", expanded=True) as status:
        total_patients = len(patient_dirs)
        
        # Process each patient sequentially
        for i, patient_id in enumerate(patient_dirs):
            status.update(label=f"Scanning patient {i+1}/{total_patients}: {patient_id}")
            patient_path = os.path.join(data_path, patient_id)
            
            patient_status[patient_id] = {'status': 'error', 'issues': [], 'contours': []}
            rtstruct_files = []
            series_dirs = {}  # Using a dict to store series_uid -> (series_path, modality)

            try:
                st.write(f"  - Identifying all DICOM series for `{patient_id}`...")
                
                # Sequential scan of all files
                for dirpath, _, filenames in os.walk(patient_path):
                    for f in filenames:
                        # A simple check for DICOM-like files
                        if not f.lower().endswith(('.dcm', '.ima')) and '.' in f:
                            continue
                        
                        full_path = os.path.join(dirpath, f)
                        try:
                            dcm = pydicom.dcmread(full_path, stop_before_pixels=True)
                            modality = getattr(dcm, 'Modality', 'Unknown')
                            
                            if modality == 'RTSTRUCT':
                                rtstruct_files.append(full_path)
                            elif modality in ['CT', 'MR', 'PT']:
                                available_modalities.add(modality)
                                series_uid = getattr(dcm, 'SeriesInstanceUID', 'UnknownSeries')
                                if series_uid not in series_dirs:
                                    series_dirs[series_uid] = (dirpath, modality)
                        except Exception:
                            continue

                # Filter series by selected modality
                filtered_series_dirs = {uid: path for uid, (path, modality) in series_dirs.items() 
                                      if modality == selected_modality}

                if not rtstruct_files:
                    st.warning(f"  - No RTSTRUCT files found for `{patient_id}`.")
                    patient_status[patient_id]['issues'].append("No RTSTRUCT files found")
                    continue
                if not filtered_series_dirs:
                    st.warning(f"  - No {selected_modality} image series found for `{patient_id}`.")
                    patient_status[patient_id]['issues'].append(f"No {selected_modality} image series found")
                    continue

                st.write(f"  - Found {len(rtstruct_files)} RTSTRUCT(s) and {len(filtered_series_dirs)} {selected_modality} image series.")
                st.write(f"  - Now finding a compatible RTSTRUCT/Image pair...")

                # Sequential search for compatible pair
                compatible_found = False
                for rt_path in rtstruct_files:
                    for series_path in filtered_series_dirs.values():
                        try:
                            # Use the more explicit create_from method
                            rtstruct = RTStructBuilder.create_from(
                                dicom_series_path=series_path,
                                rt_struct_path=rt_path
                            )
                            contours = rtstruct.get_roi_names()
                            st.success(f"  - ✅ Success! Found compatible {selected_modality} pair for `{patient_id}`.")
                            
                            all_contours.update(contours)
                            patient_contour_data[patient_id] = contours
                            patient_status[patient_id]['status'] = 'success'
                            patient_status[patient_id]['contours'] = contours
                            compatible_found = True
                            break  # Exit inner loop
                        except Exception as e:
                            # This pair was not compatible, try the next one
                            continue
                    if compatible_found:
                        break # Exit outer loop

                if not compatible_found:
                    st.error(f"  - ❌ No compatible RTSTRUCT/{selected_modality} pair found for `{patient_id}`.")
                    patient_status[patient_id]['issues'].append(f"No compatible RTSTRUCT/{selected_modality} pair found")

            except Exception as e:
                st.error(f"  - A critical error occurred while processing `{patient_id}`: {e}")
                patient_status[patient_id]['issues'].append(f"Critical error: {e}")
        
        status.update(label="Scan complete!", state="complete")

    return sorted(list(all_contours)), patient_contour_data, patient_status, sorted(list(available_modalities))


def preprocess_uploaded_data(data_path, target_contour_name, selected_modality='CT'):
    """
    Processes each patient folder sequentially:
    1. Finds the image series and the RTSTRUCT file for the specified modality.
    2. Finds the specified target contour.
    3. Converts the image series to a NIfTI file.
    4. Converts the RTSTRUCT contour to a NIfTI mask with proper spatial alignment.
    5. Returns a pandas DataFrame with paths to the NIfTI files.
    """
    # Create a single output directory for all NIfTI files for this session
    output_dir = tempfile.mkdtemp(prefix="radiomics_nifti_")
    st.session_state['temp_output_dir'] = output_dir # Store for later cleanup
    
    dataset_records = []
    patient_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    
    progress_bar = st.progress(0)
    
    # Process each patient sequentially
    for i, patient_id in enumerate(patient_dirs):
        progress_bar.progress((i + 1) / len(patient_dirs), text=f"Processing patient: {patient_id}")
        
        patient_path = os.path.join(data_path, patient_id)
        
        try:
            # Find the main image series (filtered by modality) and RTSTRUCT
            rtstruct_files = []
            series_candidates = []
            
            for root, _, files in os.walk(patient_path):
                for f in files:
                    if not f.lower().endswith(('.dcm', '.ima')) and '.' in f:
                        continue
                    
                    full_path = os.path.join(root, f)
                    try:
                        dcm = pydicom.dcmread(full_path, stop_before_pixels=True)
                        modality = getattr(dcm, 'Modality', 'Unknown')
                        
                        if modality == 'RTSTRUCT':
                            rtstruct_files.append(full_path)
                        elif modality == selected_modality:
                            series_uid = getattr(dcm, 'SeriesInstanceUID', 'UnknownSeries')
                            # Store series info with slice count for better selection
                            series_candidates.append({
                                'path': root,
                                'series_uid': series_uid,
                                'file_count': len([f for f in os.listdir(root) if f.lower().endswith(('.dcm', '.ima'))])
                            })
                    except Exception:
                        continue
            
            # Select the series with the most files (likely the main imaging series)
            if not series_candidates:
                st.warning(f"Skipping {patient_id}: No {selected_modality} image series found.")
                continue
            
            # Remove duplicates and select the series with most files
            unique_series = {}
            for candidate in series_candidates:
                if candidate['series_uid'] not in unique_series or candidate['file_count'] > unique_series[candidate['series_uid']]['file_count']:
                    unique_series[candidate['series_uid']] = candidate
            
            best_series = max(unique_series.values(), key=lambda x: x['file_count'])
            series_path = best_series['path']
            
            if not rtstruct_files:
                st.warning(f"Skipping {patient_id}: No RTSTRUCT files found.")
                continue

            # Find compatible RTSTRUCT/Image pair
            compatible_pair = None
            for rt_file in rtstruct_files:
                try:
                    # Test compatibility
                    rtstruct = RTStructBuilder.create_from(
                        dicom_series_path=series_path, 
                        rt_struct_path=rt_file
                    )
                    compatible_pair = (series_path, rt_file, rtstruct)
                    break
                except Exception as e:
                    st.write(f"  - Testing RTSTRUCT compatibility failed: {e}")
                    continue
            
            if not compatible_pair:
                st.warning(f"Skipping {patient_id}: No compatible RTSTRUCT/{selected_modality} pair found.")
                continue
            
            series_path, rt_file, rtstruct = compatible_pair

            # Find the exact contour name (case-insensitive search)
            actual_roi_name = None
            available_rois = rtstruct.get_roi_names()
            for roi in available_rois:
                if target_contour_name.lower() in roi.lower():
                    actual_roi_name = roi
                    break
            
            if not actual_roi_name:
                st.warning(f"Skipping {patient_id}: Target contour '{target_contour_name}' not found. Available: {available_rois}")
                continue

            # Load image series with SimpleITK
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(series_path)
            if not dicom_names:
                st.warning(f"Skipping {patient_id}: No DICOM files found in series path.")
                continue
            
            reader.SetFileNames(dicom_names)
            image_sitk = reader.Execute()
            
            # Get the mask with proper spatial alignment
            mask_3d = rtstruct.get_roi_mask_by_name(actual_roi_name)
            
            # CRITICAL FIX: Ensure proper orientation and spacing
            # The mask from rt_utils is in the same coordinate system as the DICOM series
            # but we need to make sure it matches the SimpleITK image exactly
            
            # Convert mask to SimpleITK image with matching properties
            mask_sitk = sitk.GetImageFromArray(mask_3d.astype(np.uint8))
            
            # Copy all spatial information from the original image
            mask_sitk.SetOrigin(image_sitk.GetOrigin())
            mask_sitk.SetSpacing(image_sitk.GetSpacing())
            mask_sitk.SetDirection(image_sitk.GetDirection())
            
            # Verify dimensions match
            if mask_sitk.GetSize() != image_sitk.GetSize():
                st.warning(f"Dimension mismatch for {patient_id}: Image {image_sitk.GetSize()} vs Mask {mask_sitk.GetSize()}")
                # Try to resample mask to match image
                try:
                    resampler = sitk.ResampleImageFilter()
                    resampler.SetReferenceImage(image_sitk)
                    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                    resampler.SetDefaultPixelValue(0)
                    mask_sitk = resampler.Execute(mask_sitk)
                    st.info(f"  - Resampled mask to match image dimensions for {patient_id}")
                except Exception as e:
                    st.error(f"Failed to resample mask for {patient_id}: {e}")
                    continue

            # Save both image and mask as NIfTI files
            patient_output_dir = os.path.join(output_dir, patient_id)
            os.makedirs(patient_output_dir, exist_ok=True)
            
            output_image_path = os.path.join(patient_output_dir, "image.nii.gz")
            output_mask_path = os.path.join(patient_output_dir, "mask.nii.gz")
            
            sitk.WriteImage(image_sitk, output_image_path)
            sitk.WriteImage(mask_sitk, output_mask_path)
            
            # Verify the saved files
            try:
                test_img = sitk.ReadImage(output_image_path)
                test_mask = sitk.ReadImage(output_mask_path)
                if test_img.GetSize() != test_mask.GetSize():
                    st.error(f"Size mismatch after saving for {patient_id}")
                    continue
                
                # Check if mask has any positive values
                mask_array = sitk.GetArrayFromImage(test_mask)
                if np.sum(mask_array) == 0:
                    st.warning(f"Warning: Mask for {patient_id} appears to be empty")
                else:
                    st.info(f"  - Mask for {patient_id} contains {np.sum(mask_array)} voxels")
                
            except Exception as e:
                st.error(f"Failed to verify saved files for {patient_id}: {e}")
                continue
            
            dataset_records.append({
                'patient_id': patient_id,
                'image_path': output_image_path,
                'mask_path': output_mask_path,
                'roi_name': actual_roi_name,
                'modality': selected_modality
            })
            st.success(f"Successfully processed {patient_id} with {selected_modality} and ROI '{actual_roi_name}'.")

        except Exception as e:
            st.error(f"Failed to process {patient_id}: {e}")
            import traceback
            st.code(traceback.format_exc())
            continue
            
    return pd.DataFrame(dataset_records)
