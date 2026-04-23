# processing.py
"""
Enhanced data loading, validation, organization, and pre-processing module.
Updated: 2025-11-03 17:18:58 UTC by Medphysicist

Features:
- Original robust DICOM processing with ultimate mask recovery (PRESERVED)
- Multi-ROI processing support (NEW - UI orchestrated, zero workflow changes)
- Enhanced NIfTI file support alongside DICOM
- Enhanced multi-modality support (CT, MRI, PET/CT)
- Longitudinal data handling
- Progress tracking with ETA
- Improved error handling and recovery
- All original mask generation methods preserved (5 robust methods)

Multi-ROI Processing Strategy:
- Functions accept `selected_roi` parameter but process ONE ROI at a time
- UI layer handles multi-ROI by calling these functions multiple times
- Each call processes one ROI completely through robust pipeline
- Results are concatenated by UI layer
- Zero changes to existing robust mask generation workflow
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
import cv2
from skimage import draw
import scipy.ndimage as ndimage
import time
from typing import Dict, List, Tuple, Optional

# =============================================================================
# UTILITY FUNCTIONS (Stubs for missing imports from utils.py)
# =============================================================================

def ProgressTracker(total, description):
    """Simple progress tracker - prints to console"""
    class Tracker:
        def __init__(self, total, desc):
            self.total = total
            self.desc = desc
            self.current = 0
            print(f"Starting: {desc} ({total} items)")
        
        def update(self, current, message=""):
            self.current = current + 1
            if message:
                print(f"  [{self.current}/{self.total}] {message}")
        
        def complete(self, message=""):
            print(f"✅ Complete: {self.desc} - {message}")
    
    return Tracker(total, description)


def detect_file_type(file_path):
    """Detect file type from extension"""
    file_path = str(file_path).lower()
    if file_path.endswith(('.nii', '.nii.gz')):
        return 'nifti'
    elif file_path.endswith(('.dcm', '.ima', '.dicom')):
        return 'dicom'
    return 'unknown'


def validate_nifti_pair(image_path, mask_path):
    """Validate NIfTI image/mask pair"""
    try:
        if not os.path.exists(image_path):
            return False, "Image file not found"
        if not os.path.exists(mask_path):
            return False, "Mask file not found"
        
        image = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(mask_path)
        
        if image.GetSize() != mask.GetSize():
            return False, f"Dimension mismatch: image {image.GetSize()} vs mask {mask.GetSize()}"
        
        return True, "Valid pair"
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def organize_nifti_files(uploaded_files):
    """Organize uploaded NIfTI files"""
    temp_dir = tempfile.mkdtemp(prefix="radiomics_nifti_upload_")
    
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    
    return temp_dir


# =============================================================================
# ROBUST MASK CONVERSION AND SAVING SYSTEM (ORIGINAL - PRESERVED)
# =============================================================================

def smart_mask_resampling_with_coordinate_preservation(mask_3d, mask_sitk, image_sitk, patient_id, status_placeholder=None):
    """Smart resampling that preserves mask voxels by trying multiple approaches."""
    
    if status_placeholder:
        status_placeholder.info(f"  - Smart resampling for {patient_id}...")
    
    original_voxels = np.sum(mask_3d > 0)
    
    # Method 1: Skip resampling if sizes are close enough
    mask_size = mask_sitk.GetSize()
    image_size = image_sitk.GetSize()
    
    size_diff = max(abs(a - b) for a, b in zip(mask_size, image_size))
    if size_diff <= 2:
        if status_placeholder:
            status_placeholder.info(f"  - Size difference is small ({size_diff}), skipping resampling")
        return mask_sitk, original_voxels
    
    # Method 2: Manual coordinate scaling (most reliable)
    try:
        if status_placeholder:
            status_placeholder.info(f"  - Trying manual coordinate scaling...")
        
        new_mask_array = np.zeros(sitk.GetArrayFromImage(image_sitk).shape, dtype=np.uint8)
        mask_array = sitk.GetArrayFromImage(mask_sitk)
        
        scale_z = new_mask_array.shape[0] / mask_array.shape[0]
        scale_y = new_mask_array.shape[1] / mask_array.shape[1]
        scale_x = new_mask_array.shape[2] / mask_array.shape[2]
        
        for z in range(mask_array.shape[0]):
            for y in range(mask_array.shape[1]):
                for x in range(mask_array.shape[2]):
                    if mask_array[z, y, x] > 0:
                        new_z = int(z * scale_z)
                        new_y = int(y * scale_y)
                        new_x = int(x * scale_x)
                        
                        if (0 <= new_z < new_mask_array.shape[0] and
                            0 <= new_y < new_mask_array.shape[1] and
                            0 <= new_x < new_mask_array.shape[2]):
                            new_mask_array[new_z, new_y, new_x] = 1
        
        new_mask_sitk = sitk.GetImageFromArray(new_mask_array)
        new_mask_sitk.SetOrigin(image_sitk.GetOrigin())
        new_mask_sitk.SetSpacing(image_sitk.GetSpacing())
        new_mask_sitk.SetDirection(image_sitk.GetDirection())
        
        final_voxels = np.sum(new_mask_array > 0)
        
        if final_voxels > 0:
            if status_placeholder:
                status_placeholder.success(f"  - ✅ Manual scaling worked: {original_voxels} -> {final_voxels} voxels")
            return new_mask_sitk, final_voxels
            
    except Exception as e:
        if status_placeholder:
            status_placeholder.warning(f"  - Manual scaling failed: {e}")
    
    # Method 3: Try different interpolation methods
    interpolation_methods = [
        ("linear", sitk.sitkLinear),
        ("bspline", sitk.sitkBSpline),
        ("gaussian", sitk.sitkGaussian)
    ]
    
    for method_name, interpolator in interpolation_methods:
        try:
            if status_placeholder:
                status_placeholder.info(f"  - Trying {method_name} interpolation...")
            
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(image_sitk)
            resampler.SetInterpolator(interpolator)
            resampler.SetDefaultPixelValue(0)
            
            resampled_mask = resampler.Execute(mask_sitk)
            resampled_mask = sitk.BinaryThreshold(resampled_mask, 0.5, 2.0, 1, 0)
            
            resampled_array = sitk.GetArrayFromImage(resampled_mask)
            final_voxels = np.sum(resampled_array > 0)
            
            if final_voxels > 0:
                if status_placeholder:
                    status_placeholder.success(f"  - ✅ {method_name} worked: {original_voxels} -> {final_voxels} voxels")
                return resampled_mask, final_voxels
                
        except Exception as e:
            if status_placeholder:
                status_placeholder.warning(f"  - {method_name} failed: {e}")
            continue
    
    return None, 0


def bypass_resampling_when_possible(mask_sitk, image_sitk, patient_id, status_placeholder=None):
    """Try to avoid resampling altogether."""
    mask_size = mask_sitk.GetSize()
    image_size = image_sitk.GetSize()
    
    if mask_size == image_size:
        if status_placeholder:
            status_placeholder.info(f"  - Sizes match exactly, no resampling needed for {patient_id}")
        return mask_sitk, True
    
    max_diff = max(abs(a - b) for a, b in zip(mask_size, image_size))
    if max_diff <= 5:
        if status_placeholder:
            status_placeholder.info(f"  - Small size difference ({max_diff}), copying spatial properties only")
        
        try:
            mask_sitk.SetOrigin(image_sitk.GetOrigin())
            mask_sitk.SetSpacing(image_sitk.GetSpacing())
            mask_sitk.SetDirection(image_sitk.GetDirection())
            return mask_sitk, True
        except:
            pass
    
    return mask_sitk, False


def robust_mask_to_sitk_conversion(mask_3d, image_sitk, patient_id, status_placeholder=None):
    """Robust conversion of mask array to SimpleITK image with multiple fallback methods."""
    
    # Convert to uint8
    mask_uint8 = mask_3d.astype(np.uint8)
    debug_info = {
        'input_nonzero_voxels': int(np.sum(mask_3d > 0)),
        'uint8_nonzero_voxels': int(np.sum(mask_uint8 > 0))
    }
    
    conversion_methods = [
        ("standard_conversion", lambda: sitk.GetImageFromArray(mask_uint8)),
        ("explicit_uint8_conversion", lambda: sitk.GetImageFromArray(mask_uint8.astype(np.uint8))),
        ("binary_conversion", lambda: sitk.GetImageFromArray((mask_uint8 > 0).astype(np.uint8))),
        ("clipped_conversion", lambda: sitk.GetImageFromArray(np.clip(mask_uint8, 0, 1).astype(np.uint8)))
    ]
    
    for method_name, conversion_func in conversion_methods:
        try:
            if status_placeholder:
                status_placeholder.info(f"  - Trying {method_name} for {patient_id}...")
            
            mask_sitk = conversion_func()
            mask_sitk.SetOrigin(image_sitk.GetOrigin())
            mask_sitk.SetSpacing(image_sitk.GetSpacing())
            mask_sitk.SetDirection(image_sitk.GetDirection())
            
            test_array = sitk.GetArrayFromImage(mask_sitk)
            final_voxels = np.sum(test_array > 0)
            
            debug_info[f'{method_name}_final_voxels'] = int(final_voxels)
            
            if final_voxels > 0:
                if status_placeholder:
                    status_placeholder.success(f"  - ✅ {method_name} worked for {patient_id} ({final_voxels} voxels)")
                return mask_sitk, debug_info
            else:
                if status_placeholder:
                    status_placeholder.warning(f"  - {method_name} resulted in empty mask for {patient_id}")
                
        except Exception as e:
            debug_info[f'{method_name}_error'] = str(e)
            if status_placeholder:
                status_placeholder.warning(f"  - {method_name} failed for {patient_id}: {str(e)}")
            continue
    
    return None, debug_info


def robust_mask_file_saving(mask_sitk, output_path, patient_id, status_placeholder=None):
    """Robust mask file saving with multiple methods and verification."""
    
    saving_methods = [
        ("standard_write", lambda: sitk.WriteImage(mask_sitk, output_path)),
        ("explicit_cast_write", lambda: sitk.WriteImage(sitk.Cast(mask_sitk, sitk.sitkUInt8), output_path)),
        ("compressed_write", lambda: sitk.WriteImage(mask_sitk, output_path, True))
    ]
    
    for method_name, saving_func in saving_methods:
        try:
            if status_placeholder:
                status_placeholder.info(f"  - Trying {method_name} for {patient_id}...")
            
            saving_func()
            
            if os.path.exists(output_path):
                test_mask = sitk.ReadImage(output_path)
                test_array = sitk.GetArrayFromImage(test_mask)
                saved_voxels = np.sum(test_array > 0)
                
                if saved_voxels > 0:
                    if status_placeholder:
                        status_placeholder.success(f"  - ✅ {method_name} successfully saved {patient_id} ({saved_voxels} voxels)")
                    return True, saved_voxels, output_path
                else:
                    if status_placeholder:
                        status_placeholder.error(f"  - ❌ {method_name} saved empty file for {patient_id}")
                    try:
                        os.remove(output_path)
                    except:
                        pass
            else:
                if status_placeholder:
                    status_placeholder.error(f"  - {method_name} didn't create file for {patient_id}")
                
        except Exception as e:
            if status_placeholder:
                status_placeholder.error(f"  - {method_name} failed for {patient_id}: {str(e)}")
            continue
    
    return False, 0, output_path


def alternative_mask_saving_formats(mask_3d, image_sitk, patient_output_dir, patient_id, status_placeholder=None):
    """Try saving in alternative formats if NIfTI fails."""
    alternative_formats = [
        ("mask.mha", "MetaImage format"),
        ("mask.nrrd", "NRRD format"),
        ("mask_uncompressed.nii", "Uncompressed NIfTI")
    ]
    
    for filename, description in alternative_formats:
        try:
            output_path = os.path.join(patient_output_dir, filename)
            
            if status_placeholder:
                status_placeholder.info(f"  - Trying {description} for {patient_id}...")
            
            mask_sitk, debug_info = robust_mask_to_sitk_conversion(mask_3d, image_sitk, patient_id, status_placeholder)
            
            if mask_sitk is not None:
                sitk.WriteImage(mask_sitk, output_path)
                
                if os.path.exists(output_path):
                    test_mask = sitk.ReadImage(output_path)
                    test_array = sitk.GetArrayFromImage(test_mask)
                    saved_voxels = np.sum(test_array > 0)
                    
                    if saved_voxels > 0:
                        if status_placeholder:
                            status_placeholder.success(f"  - ✅ {description} worked for {patient_id} ({saved_voxels} voxels)")
                        return output_path, saved_voxels
                    
        except Exception as e:
            if status_placeholder:
                status_placeholder.warning(f"  - {description} failed for {patient_id}: {str(e)}")
            continue
    
    return None, 0


# =============================================================================
# MULTI-LIBRARY ROBUST MASK GENERATION (ORIGINAL 5 METHODS - PRESERVED)
# =============================================================================

def create_mask_using_sitk_skimage(ds, roi_name, image_sitk, series_path):
    """Use SimpleITK + scikit-image for robust polygon processing."""
    try:
        image_array = sitk.GetArrayFromImage(image_sitk)
        mask_array = np.zeros(image_array.shape, dtype=np.uint8)
        
        roi_number = None
        for roi_struct in ds.StructureSetROISequence:
            if getattr(roi_struct, 'ROIName', '').lower() == roi_name.lower():
                roi_number = getattr(roi_struct, 'ROINumber', None)
                break
        
        if roi_number is None:
            return None
        
        processed_slices = 0
        total_points_processed = 0
        
        for roi_contour in ds.ROIContourSequence:
            if getattr(roi_contour, 'ReferencedROINumber', None) == roi_number:
                contour_sequence = getattr(roi_contour, 'ContourSequence', [])
                
                for contour in contour_sequence:
                    contour_data = getattr(contour, 'ContourData', [])
                    if len(contour_data) < 9:
                        continue
                    
                    points_world = np.array(contour_data).reshape(-1, 3)
                    points_image = []
                    
                    for world_point in points_world:
                        try:
                            image_point = image_sitk.TransformPhysicalPointToIndex(world_point.tolist())
                            
                            if (0 <= image_point[0] < image_sitk.GetSize()[0] and
                                0 <= image_point[1] < image_sitk.GetSize()[1] and
                                0 <= image_point[2] < image_sitk.GetSize()[2]):
                                points_image.append([image_point[2], image_point[1], image_point[0]])
                        except:
                            continue
                    
                    if len(points_image) < 3:
                        continue
                    
                    points_image = np.array(points_image)
                    total_points_processed += len(points_image)
                    
                    slice_groups = {}
                    for point in points_image:
                        z_idx = int(round(point[0]))
                        if 0 <= z_idx < mask_array.shape[0]:
                            if z_idx not in slice_groups:
                                slice_groups[z_idx] = []
                            slice_groups[z_idx].append([point[1], point[2]])
                    
                    for z_idx, slice_points in slice_groups.items():
                        if len(slice_points) >= 3:
                            points_array = np.array(slice_points)
                            
                            try:
                                rr, cc = draw.polygon(points_array[:, 0], points_array[:, 1], 
                                                    shape=mask_array[z_idx].shape)
                                
                                valid_indices = (rr >= 0) & (rr < mask_array.shape[1]) & \
                                              (cc >= 0) & (cc < mask_array.shape[2])
                                
                                if np.any(valid_indices):
                                    mask_array[z_idx, rr[valid_indices], cc[valid_indices]] = 1
                                    processed_slices += 1
                                
                            except Exception:
                                continue
                
                break
        
        final_voxel_count = np.sum(mask_array > 0)
        return mask_array if final_voxel_count > 0 else None
        
    except Exception as e:
        return None


def create_mask_using_morphology_enhancement(ds, roi_name, image_sitk, series_path):
    """Try direct DICOM approach with morphological enhancement."""
    try:
        mask_array = create_mask_from_dicom_contours_direct(ds, roi_name, image_sitk)
        
        if mask_array is None or np.sum(mask_array > 0) == 0:
            return None
            
        mask_closed = ndimage.binary_closing(mask_array, structure=np.ones((3,3,3)))
        
        for z in range(mask_closed.shape[0]):
            if np.any(mask_closed[z]):
                mask_closed[z] = ndimage.binary_fill_holes(mask_closed[z])
        
        mask_dilated = ndimage.binary_dilation(mask_closed, structure=np.ones((2,2,2)))
        
        final_mask = mask_dilated.astype(np.uint8)
        final_voxel_count = np.sum(final_mask > 0)
        
        return final_mask if final_voxel_count > 0 else None
        
    except Exception:
        return None


def create_mask_from_dicom_contours_direct(ds, roi_name, image_sitk):
    """Bypass rt-utils completely and create mask directly from DICOM contour data."""
    try:
        image_array = sitk.GetArrayFromImage(image_sitk)
        spacing = np.array(image_sitk.GetSpacing())
        origin = np.array(image_sitk.GetOrigin())
        direction = np.array(image_sitk.GetDirection()).reshape(3, 3)
        
        mask_array = np.zeros(image_array.shape, dtype=np.uint8)
        
        roi_number = None
        for roi_struct in ds.StructureSetROISequence:
            if getattr(roi_struct, 'ROIName', '').lower() == roi_name.lower():
                roi_number = getattr(roi_struct, 'ROINumber', None)
                break
        
        if roi_number is None:
            return None
        
        processed_slices = 0
        
        for roi_contour in ds.ROIContourSequence:
            if getattr(roi_contour, 'ReferencedROINumber', None) == roi_number:
                contour_sequence = getattr(roi_contour, 'ContourSequence', [])
                
                for contour in contour_sequence:
                    contour_data = getattr(contour, 'ContourData', [])
                    if len(contour_data) < 9:
                        continue
                    
                    points_world = np.array(contour_data).reshape(-1, 3)
                    
                    points_image = []
                    for world_point in points_world:
                        relative_pos = world_point - origin
                        
                        if not np.allclose(direction, np.eye(3)):
                            direction_inv = np.linalg.inv(direction)
                            rotated_pos = direction_inv @ relative_pos
                        else:
                            rotated_pos = relative_pos
                        
                        image_coord = rotated_pos / spacing
                        points_image.append([image_coord[2], image_coord[1], image_coord[0]])
                    
                    points_image = np.array(points_image)
                    
                    slice_groups = {}
                    for point in points_image:
                        z_idx = int(round(point[0]))
                        if 0 <= z_idx < mask_array.shape[0]:
                            if z_idx not in slice_groups:
                                slice_groups[z_idx] = []
                            slice_groups[z_idx].append([int(round(point[2])), int(round(point[1]))])
                    
                    for z_idx, slice_points in slice_groups.items():
                        if len(slice_points) >= 3:
                            points_2d = np.array(slice_points, dtype=np.int32)
                            
                            points_2d[:, 0] = np.clip(points_2d[:, 0], 0, mask_array.shape[2] - 1)
                            points_2d[:, 1] = np.clip(points_2d[:, 1], 0, mask_array.shape[1] - 1)
                            
                            cv2.fillPoly(mask_array[z_idx], [points_2d], 1)
                            processed_slices += 1
                
                break
        
        final_voxel_count = np.sum(mask_array > 0)
        return mask_array if final_voxel_count > 0 else None
        
    except Exception:
        return None


def create_mask_using_enhanced_coordinate_transform(ds, roi_name, image_sitk, series_path):
    """Enhanced coordinate transformation using multiple validation methods."""
    try:
        image_array = sitk.GetArrayFromImage(image_sitk)
        mask_array = np.zeros(image_array.shape, dtype=np.uint8)
        
        roi_number = None
        for roi_struct in ds.StructureSetROISequence:
            if getattr(roi_struct, 'ROIName', '').lower() == roi_name.lower():
                roi_number = getattr(roi_struct, 'ROINumber', None)
                break
        
        if roi_number is None:
            return None
        
        processed_slices = 0
        
        for roi_contour in ds.ROIContourSequence:
            if getattr(roi_contour, 'ReferencedROINumber', None) == roi_number:
                contour_sequence = getattr(roi_contour, 'ContourSequence', [])
                
                for contour in contour_sequence:
                    contour_data = getattr(contour, 'ContourData', [])
                    if len(contour_data) < 9:
                        continue
                    
                    points_world = np.array(contour_data).reshape(-1, 3)
                    
                    points_image_methods = []
                    
                    try:
                        points_sitk = []
                        for world_point in points_world:
                            image_point = image_sitk.TransformPhysicalPointToIndex(world_point.tolist())
                            if all(0 <= p < s for p, s in zip(image_point, image_sitk.GetSize())):
                                points_sitk.append([image_point[2], image_point[1], image_point[0]])
                        if len(points_sitk) >= 3:
                            points_image_methods.append(("sitk_transform", np.array(points_sitk)))
                    except:
                        pass
                    
                    try:
                        spacing = np.array(image_sitk.GetSpacing())
                        origin = np.array(image_sitk.GetOrigin())
                        
                        points_manual = []
                        for world_point in points_world:
                            image_coord = (world_point - origin) / spacing
                            image_point = [image_coord[2], image_coord[1], image_coord[0]]
                            
                            if (0 <= image_point[0] < image_array.shape[0] and
                                0 <= image_point[1] < image_array.shape[1] and
                                0 <= image_point[2] < image_array.shape[2]):
                                points_manual.append(image_point)
                        
                        if len(points_manual) >= 3:
                            points_image_methods.append(("manual_transform", np.array(points_manual)))
                    except:
                        pass
                    
                    if not points_image_methods:
                        continue
                    
                    best_method, points_image = max(points_image_methods, key=lambda x: len(x[1]))
                    
                    slice_groups = {}
                    for point in points_image:
                        z_idx = int(round(point[0]))
                        if 0 <= z_idx < mask_array.shape[0]:
                            if z_idx not in slice_groups:
                                slice_groups[z_idx] = []
                            slice_groups[z_idx].append([point[1], point[2]])
                    
                    for z_idx, slice_points in slice_groups.items():
                        if len(slice_points) >= 3:
                            points_array = np.array(slice_points)
                            
                            try:
                                rr, cc = draw.polygon(points_array[:, 0], points_array[:, 1], 
                                                    shape=mask_array[z_idx].shape)
                                
                                valid_indices = (rr >= 0) & (rr < mask_array.shape[1]) & \
                                              (cc >= 0) & (cc < mask_array.shape[2])
                                
                                if np.any(valid_indices):
                                    mask_array[z_idx, rr[valid_indices], cc[valid_indices]] = 1
                                    processed_slices += 1
                                    
                            except:
                                continue
                
                break
        
        final_voxel_count = np.sum(mask_array > 0)
        return mask_array if final_voxel_count > 0 else None
        
    except Exception:
        return None


def ultimate_mask_recovery_robust(rtstruct, roi_name, image_sitk, series_path, patient_id, status_placeholder=None):
    """
    Try multiple robust approaches in sequence for maximum recovery rate.
    This is the CORE function that makes mask generation ultra-robust.
    """
    approaches = [
        ("rt-utils", lambda: rtstruct.get_roi_mask_by_name(roi_name)),
        ("sitk-skimage", lambda: create_mask_using_sitk_skimage(rtstruct.ds, roi_name, image_sitk, series_path)),
        ("enhanced-coord-transform", lambda: create_mask_using_enhanced_coordinate_transform(rtstruct.ds, roi_name, image_sitk, series_path)),
        ("morphology-enhanced", lambda: create_mask_using_morphology_enhancement(rtstruct.ds, roi_name, image_sitk, series_path)),
        ("direct-dicom", lambda: create_mask_from_dicom_contours_direct(rtstruct.ds, roi_name, image_sitk))
    ]
    
    for i, (approach_name, approach_func) in enumerate(approaches):
        try:
            if status_placeholder:
                status_placeholder.info(f"  - Method {i+1}/5: Trying {approach_name} approach for {patient_id}...")
            
            mask_result = approach_func()
            
            if mask_result is not None and np.sum(mask_result > 0) > 0:
                voxel_count = np.sum(mask_result > 0)
                if status_placeholder:
                    status_placeholder.success(f"  - ✅ SUCCESS: {approach_name} worked for {patient_id} ({voxel_count:,} voxels)")
                return mask_result, roi_name, f"robust_{approach_name.replace('-', '_')}"
            else:
                if status_placeholder:
                    status_placeholder.warning(f"  - {approach_name} returned empty mask for {patient_id}")
            
        except Exception as e:
            if status_placeholder:
                status_placeholder.warning(f"  - {approach_name} failed for {patient_id}: {str(e)}")
            continue
    
    if status_placeholder:
        status_placeholder.error(f"  - ❌ All 5 methods failed for {patient_id}")
    
    return None, None, None


# =============================================================================
# RECOVERY HELPER FUNCTIONS
# =============================================================================

def find_similar_roi_names(available_rois, target_roi, min_similarity=0.6):
    """Find similar ROI names using simple string matching."""
    import difflib
    
    target_lower = target_roi.lower().strip()
    matches = []
    
    for roi in available_rois:
        roi_lower = roi.lower().strip()
        
        if target_lower in roi_lower or roi_lower in target_lower:
            matches.append((roi, 0.9, "substring"))
        
        similarity = difflib.SequenceMatcher(None, target_lower, roi_lower).ratio()
        if similarity >= min_similarity:
            matches.append((roi, similarity, "difflib"))
    
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches


def create_fallback_mask(image_sitk, patient_id, status_placeholder=None):
    """Create a small fallback mask in the center of the image."""
    try:
        image_array = sitk.GetArrayFromImage(image_sitk)
        mask_array = np.zeros_like(image_array, dtype=np.uint8)
        
        center_z = image_array.shape[0] // 2
        center_y = image_array.shape[1] // 2
        center_x = image_array.shape[2] // 2
        
        size = 5
        for z in range(max(0, center_z - size), min(image_array.shape[0], center_z + size)):
            for y in range(max(0, center_y - size), min(image_array.shape[1], center_y + size)):
                for x in range(max(0, center_x - size), min(image_array.shape[2], center_x + size)):
                    mask_array[z, y, x] = 1
        
        if status_placeholder:
            voxel_count = np.sum(mask_array)
            status_placeholder.warning(f"  - ⚠️ Created fallback mask for {patient_id} - contains {voxel_count} voxels")
        
        return mask_array
        
    except Exception:
        return None


def validate_rtstruct_contours(rtstruct, roi_name):
    """Validates RTSTRUCT contours before mask generation."""
    try:
        roi_names = rtstruct.get_roi_names()
        if roi_name not in roi_names:
            return False, f"ROI '{roi_name}' not found in available ROIs: {roi_names}"
        
        return True, "ROI validation passed"
        
    except Exception as e:
        return False, f"ROI validation failed: {str(e)}"


# =============================================================================
# ENHANCED MODALITY DETECTION
# =============================================================================

def enhanced_modality_detection(dcm_file_path):
    """Enhanced modality detection with better MRI/PET support and sub-classification"""
    try:
        dcm = pydicom.dcmread(dcm_file_path, stop_before_pixels=True)
        modality = getattr(dcm, 'Modality', 'Unknown')
        
        if modality == 'MR':
            sequence_name = getattr(dcm, 'SequenceName', '').upper()
            series_description = getattr(dcm, 'SeriesDescription', '').upper()
            protocol_name = getattr(dcm, 'ProtocolName', '').upper()
            
            all_descriptions = f"{sequence_name} {series_description} {protocol_name}"
            
            if any(keyword in all_descriptions for keyword in ['T1', 'T1W', 'T1_', 'MPRAGE', 'SPGR']):
                return 'MR_T1'
            elif any(keyword in all_descriptions for keyword in ['T2', 'T2W', 'T2_', 'TSE', 'FSE']):
                return 'MR_T2'
            elif any(keyword in all_descriptions for keyword in ['FLAIR', 'FLUID']):
                return 'MR_FLAIR'
            elif any(keyword in all_descriptions for keyword in ['DWI', 'DIFFUSION', 'ADC']):
                return 'MR_DWI'
            else:
                return 'MR'
        
        elif modality == 'PT':
            series_description = getattr(dcm, 'SeriesDescription', '').upper()
            
            if 'CT' in series_description:
                return 'PT_CT'
            elif 'FDG' in series_description:
                return 'PT_FDG'
            else:
                return 'PT'
        
        elif modality == 'CT':
            series_description = getattr(dcm, 'SeriesDescription', '').upper()
            
            if any(keyword in series_description for keyword in ['CONTRAST', 'POST', 'ENHANCED', 'C+']):
                return 'CT_CONTRAST'
            elif any(keyword in series_description for keyword in ['PLAIN', 'PRE', 'NON', 'NATIVE']):
                return 'CT_PLAIN'
            elif 'ANGIO' in series_description or 'CTA' in series_description:
                return 'CT_ANGIO'
            else:
                return 'CT'
        
        return modality
        
    except Exception:
        return 'Unknown'


def get_supported_modalities():
    """Get list of supported DICOM modalities"""
    return [
        'CT', 'CT_CONTRAST', 'CT_PLAIN', 'CT_ANGIO',
        'MR', 'MR_T1', 'MR_T2', 'MR_FLAIR', 'MR_DWI', 'MR_SWI', 'MR_TOF', 'MR_PERFUSION',
        'PT', 'PT_CT', 'PT_ATTN', 'PT_FDG'
    ]


# =============================================================================
# PATH AND DIRECTORY VALIDATION
# =============================================================================

def get_available_directories(base_paths=None):
    """Scans common system locations for potential data directories."""
    if base_paths is None:
        base_paths = ["/data", "/datasets", "/home", "/mnt", ".", os.path.expanduser("~")]
    
    available_dirs = set()
    for base_path in base_paths:
        if os.path.exists(base_path):
            try:
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
    
    for _, _, files in os.walk(path):
        for file in files:
            if file.lower().endswith('.dcm'):
                return []
    
    return ["No DICOM (.dcm) files found in the specified directory."]


def process_selected_path(selected_path):
    """Simply validates the path and returns it if valid."""
    if not selected_path or not os.path.isdir(selected_path):
        return None
    return selected_path


# =============================================================================
# DATA HANDLING AND ORGANIZATION
# =============================================================================

def organize_dicom_files(uploaded_files):
    """
    Extracts, organizes, and structures uploaded DICOM files by PatientID and SeriesUID.
    This function handles both individual files and ZIP archives.
    """
    try:
        temp_dir = tempfile.mkdtemp(prefix="radiomics_upload_")
        extract_path = os.path.join(temp_dir, "extracted")
        os.makedirs(extract_path, exist_ok=True)

        for uploaded_file in uploaded_files:
            if uploaded_file.name.lower().endswith('.zip'):
                with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
            else:
                with open(os.path.join(extract_path, uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())

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
                    continue

        if not dicom_files:
            shutil.rmtree(temp_dir)
            return None

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

        for patient_id, series_map in patient_series_map.items():
            patient_dir = os.path.join(organized_path, str(patient_id))
            for series_uid, files in series_map.items():
                modality = getattr(pydicom.dcmread(files[0], stop_before_pixels=True), 'Modality', 'UN')
                series_dir = os.path.join(patient_dir, f"{modality}_{series_uid[:8]}")
                os.makedirs(series_dir, exist_ok=True)
                for file_path in files:
                    shutil.copy(file_path, series_dir)
        
        shutil.rmtree(extract_path)
        
        return organized_path
        
    except Exception as e:
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return None


# =============================================================================
# NIFTI DATA PROCESSING
# =============================================================================

def calculate_filename_similarity(filename1: str, filename2: str) -> float:
    """Calculate similarity score between two filenames for pairing"""
    name1 = Path(filename1).stem.lower()
    name2 = Path(filename2).stem.lower()
    
    mask_indicators = ['mask', 'seg', 'roi', 'label', 'contour', 'structure', 'manual']
    for indicator in mask_indicators:
        name2 = name2.replace(indicator, '')
        name2 = name2.replace(f'_{indicator}', '').replace(f'{indicator}_', '')
    
    min_len = min(len(name1), len(name2))
    if min_len == 0:
        return 0.0
    
    common_chars = 0
    for i in range(min_len):
        if name1[i] == name2[i]:
            common_chars += 1
        else:
            break
    
    return common_chars / max(len(name1), len(name2))


def scan_nifti_data_for_analysis(data_path):
    """Scan NIfTI data directory for image/mask pairs with enhanced validation"""
    patient_data = {}
    processing_summary = {
        'total_patients': 0, 
        'valid_pairs': 0, 
        'errors': [],
        'format': 'nifti'
    }
    
    if not os.path.exists(data_path):
        return patient_data, processing_summary
    
    patient_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    processing_summary['total_patients'] = len(patient_dirs)
    
    progress_tracker = ProgressTracker(len(patient_dirs), "Scanning NIfTI data")
    
    for i, patient_id in enumerate(patient_dirs):
        progress_tracker.update(i, f"Scanning {patient_id}")
        patient_path = os.path.join(data_path, patient_id)
        
        try:
            nifti_files = []
            for file in os.listdir(patient_path):
                file_path = os.path.join(patient_path, file)
                if detect_file_type(file_path) == 'nifti':
                    is_mask = any(keyword in file.lower() for keyword in [
                        'mask', 'seg', 'roi', 'label', 'contour', 'structure', 'manual'
                    ])
                    
                    modality = 'Unknown'
                    if any(keyword in file.lower() for keyword in ['t1', 't1w']):
                        modality = 'MR_T1'
                    elif any(keyword in file.lower() for keyword in ['t2', 't2w']):
                        modality = 'MR_T2'
                    elif any(keyword in file.lower() for keyword in ['flair']):
                        modality = 'MR_FLAIR'
                    elif any(keyword in file.lower() for keyword in ['dwi', 'adc']):
                        modality = 'MR_DWI'
                    elif any(keyword in file.lower() for keyword in ['ct']):
                        modality = 'CT'
                    elif any(keyword in file.lower() for keyword in ['pet', 'fdg']):
                        modality = 'PT'
                    
                    nifti_files.append({
                        'filename': file,
                        'path': file_path,
                        'is_mask': is_mask,
                        'modality': modality
                    })
            
            images = [f for f in nifti_files if not f['is_mask']]
            masks = [f for f in nifti_files if f['is_mask']]
            
            if images and masks:
                valid_pairs = []
                for image in images:
                    best_mask = None
                    best_score = 0
                    
                    for mask in masks:
                        score = calculate_filename_similarity(image['filename'], mask['filename'])
                        is_valid, error_msg = validate_nifti_pair(image['path'], mask['path'])
                        
                        if is_valid and score > best_score:
                            best_mask = mask
                            best_score = score
                    
                    if best_mask:
                        valid_pairs.append({
                            'image': image,
                            'mask': best_mask,
                            'modality': image['modality'],
                            'similarity_score': best_score
                        })
                
                if valid_pairs:
                    patient_data[patient_id] = {
                        'pairs': valid_pairs,
                        'status': 'success',
                        'path': patient_path,
                        'available_modalities': list(set([pair['modality'] for pair in valid_pairs]))
                    }
                    processing_summary['valid_pairs'] += len(valid_pairs)
                else:
                    error_msg = f"No valid image/mask pairs found"
                    patient_data[patient_id] = {
                        'pairs': [],
                        'status': 'error',
                        'error': error_msg
                    }
                    processing_summary['errors'].append(f"{patient_id}: {error_msg}")
            else:
                error_msg = f"Insufficient files: {len(images)} images, {len(masks)} masks"
                patient_data[patient_id] = {
                    'pairs': [],
                    'status': 'error', 
                    'error': error_msg
                }
                processing_summary['errors'].append(f"{patient_id}: {error_msg}")
                
        except Exception as e:
            error_msg = f"Error processing patient {patient_id}: {str(e)}"
            patient_data[patient_id] = {
                'pairs': [],
                'status': 'error',
                'error': error_msg
            }
            processing_summary['errors'].append(error_msg)
    
    progress_tracker.complete(f"NIfTI scanning complete: {processing_summary['valid_pairs']} valid pairs found")
    
    return patient_data, processing_summary


def preprocess_nifti_data(data_path, selected_pairs):
    """Preprocess NIfTI data for feature extraction with enhanced validation"""
    dataset_records = []
    processing_summary = {
        'total_patients': 0,
        'successful_patients': 0,
        'failed_patients': {},
        'format': 'nifti'
    }
    
    output_dir = tempfile.mkdtemp(prefix="radiomics_nifti_processed_")
    st.session_state['temp_output_dir'] = output_dir
    
    progress_tracker = ProgressTracker(len(selected_pairs), "Processing NIfTI pairs")
    
    for i, pair_info in enumerate(selected_pairs):
        progress_tracker.update(i, f"Processing {pair_info['patient_id']}")
        
        try:
            patient_id = pair_info['patient_id']
            image_path = pair_info['image_path']
            mask_path = pair_info['mask_path']
            
            is_valid, error_msg = validate_nifti_pair(image_path, mask_path)
            
            if not is_valid:
                processing_summary['failed_patients'][patient_id] = {
                    'reason': 'Validation failed',
                    'details': error_msg
                }
                continue
            
            patient_output_dir = os.path.join(output_dir, patient_id)
            os.makedirs(patient_output_dir, exist_ok=True)
            
            image_sitk = sitk.ReadImage(image_path)
            mask_sitk = sitk.ReadImage(mask_path)
            
            mask_array = sitk.GetArrayFromImage(mask_sitk)
            
            unique_values = np.unique(mask_array)
            if len(unique_values) > 2:
                largest_label = 0
                largest_count = 0
                for val in unique_values:
                    if val > 0:
                        count = np.sum(mask_array == val)
                        if count > largest_count:
                            largest_count = count
                            largest_label = val
                mask_array = (mask_array == largest_label).astype(np.uint8)
            else:
                mask_array = (mask_array > 0).astype(np.uint8)
            
            mask_binary_sitk = sitk.GetImageFromArray(mask_array)
            mask_binary_sitk.CopyInformation(mask_sitk)
            
            output_image_path = os.path.join(patient_output_dir, "image.nii.gz")
            output_mask_path = os.path.join(patient_output_dir, "mask.nii.gz")
            
            sitk.WriteImage(image_sitk, output_image_path)
            sitk.WriteImage(mask_binary_sitk, output_mask_path)
            
            roi_voxel_count = np.sum(mask_array > 0)
            image_array = sitk.GetArrayFromImage(image_sitk)
            roi_intensities = image_array[mask_array > 0]
            
            dataset_records.append({
                'patient_id': patient_id,
                'image_path': output_image_path,
                'mask_path': output_mask_path,
                'roi_name': pair_info.get('roi_name', 'ROI'),
                'modality': pair_info.get('modality', 'NIfTI'),
                'original_image_path': image_path,
                'original_mask_path': mask_path,
                'roi_voxel_count': int(roi_voxel_count),
                'roi_mean_intensity': float(np.mean(roi_intensities)) if len(roi_intensities) > 0 else 0.0,
                'roi_std_intensity': float(np.std(roi_intensities)) if len(roi_intensities) > 0 else 0.0,
                'image_size': image_sitk.GetSize(),
                'voxel_spacing': image_sitk.GetSpacing()
            })
            
            processing_summary['successful_patients'] += 1
            
        except Exception as e:
            processing_summary['failed_patients'][pair_info['patient_id']] = {
                'reason': 'Processing error',
                'details': str(e)
            }
    
    progress_tracker.complete(f"NIfTI processing complete: {len(dataset_records)} patients processed")
    
    processing_summary['total_patients'] = len(selected_pairs)
    
    return pd.DataFrame(dataset_records), processing_summary


# =============================================================================
# ENHANCED DICOM SCANNING (Multi-Series Support)
# =============================================================================

def scan_uploaded_data_for_contours_enhanced(data_path, selected_modalities=['CT'], multi_series_mode=False):
    """
    Enhanced scanning with comprehensive multi-modality and longitudinal support.
    This function DOES NOT filter by ROI - it returns ALL available ROIs.
    ROI filtering happens later in preprocessing functions.
    """
    all_contours = set()
    patient_contour_data = {}
    patient_status = {}
    available_modalities = set()
    longitudinal_data = {}

    if not data_path or not os.path.isdir(data_path):
        return [], {}, {}, list(available_modalities), {}

    patient_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    
    progress_tracker = ProgressTracker(len(patient_dirs), "Enhanced DICOM scanning")
    
    with st.status("Enhanced scanning for multi-modality data...", expanded=True) as status:
        for i, patient_id in enumerate(patient_dirs):
            progress_tracker.update(i, f"Scanning {patient_id}")
            
            patient_path = os.path.join(data_path, patient_id)
            patient_status[patient_id] = {'status': 'error', 'issues': [], 'contours': []}
            
            try:
                rtstruct_files = []
                series_data = {}
                
                for dirpath, _, filenames in os.walk(patient_path):
                    for f in filenames:
                        if not f.lower().endswith(('.dcm', '.ima', '.dicom')) and '.' in f:
                            continue
                        
                        full_path = os.path.join(dirpath, f)
                        try:
                            dcm = pydicom.dcmread(full_path, stop_before_pixels=True)
                            modality = enhanced_modality_detection(full_path)
                            
                            if modality == 'RTSTRUCT':
                                rtstruct_files.append(full_path)
                            elif modality in get_supported_modalities():
                                available_modalities.add(modality)
                                series_uid = getattr(dcm, 'SeriesInstanceUID', 'UnknownSeries')
                                
                                study_date = getattr(dcm, 'StudyDate', '')
                                study_time = getattr(dcm, 'StudyTime', '')
                                
                                if study_date and study_time:
                                    timepoint = f"{study_date}_{study_time[:6]}"
                                elif study_date:
                                    timepoint = study_date
                                else:
                                    timepoint = 'TP_Unknown'
                                
                                if modality not in series_data:
                                    series_data[modality] = {}
                                if timepoint not in series_data[modality]:
                                    series_data[modality][timepoint] = {}
                                if series_uid not in series_data[modality][timepoint]:
                                    series_data[modality][timepoint][series_uid] = {
                                        'path': dirpath,
                                        'files': [],
                                        'series_description': getattr(dcm, 'SeriesDescription', ''),
                                        'study_date': study_date,
                                        'study_time': study_time,
                                        'slice_count': 0
                                    }
                                
                                series_data[modality][timepoint][series_uid]['files'].append(full_path)
                                series_data[modality][timepoint][series_uid]['slice_count'] += 1
                        except Exception:
                            continue

                compatible_pairs = []
                
                if multi_series_mode:
                    for modality in selected_modalities:
                        if modality in series_data:
                            for timepoint, timepoint_series in series_data[modality].items():
                                for series_uid, series_info in timepoint_series.items():
                                    for rt_path in rtstruct_files:
                                        try:
                                            rtstruct = RTStructBuilder.create_from(
                                                dicom_series_path=series_info['path'],
                                                rt_struct_path=rt_path
                                            )
                                            contours = rtstruct.get_roi_names()
                                            
                                            if contours:
                                                compatible_pairs.append({
                                                    'modality': modality,
                                                    'timepoint': timepoint,
                                                    'series_uid': series_uid,
                                                    'series_path': series_info['path'],
                                                    'rtstruct_path': rt_path,
                                                    'contours': contours,
                                                    'series_description': series_info['series_description'],
                                                    'study_date': series_info['study_date'],
                                                    'slice_count': series_info['slice_count']
                                                })
                                                
                                                all_contours.update(contours)
                                                
                                        except Exception:
                                            continue
                else:
                    for modality in selected_modalities:
                        if modality in series_data:
                            sorted_timepoints = sorted(series_data[modality].keys(), reverse=True)
                            latest_timepoint = sorted_timepoints[0]
                            timepoint_series = series_data[modality][latest_timepoint]
                            
                            best_series_uid = max(timepoint_series.keys(), 
                                                key=lambda x: timepoint_series[x]['slice_count'])
                            series_info = timepoint_series[best_series_uid]
                            
                            for rt_path in rtstruct_files:
                                try:
                                    rtstruct = RTStructBuilder.create_from(
                                        dicom_series_path=series_info['path'],
                                        rt_struct_path=rt_path
                                    )
                                    contours = rtstruct.get_roi_names()
                                    
                                    if contours:
                                        compatible_pairs.append({
                                            'modality': modality,
                                            'timepoint': latest_timepoint,
                                            'series_uid': best_series_uid,
                                            'series_path': series_info['path'],
                                            'rtstruct_path': rt_path,
                                            'contours': contours,
                                            'series_description': series_info['series_description'],
                                            'study_date': series_info['study_date'],
                                            'slice_count': series_info['slice_count']
                                        })
                                        
                                        all_contours.update(contours)
                                        break
                                        
                                except Exception:
                                    continue

                if compatible_pairs:
                    patient_contour_data[patient_id] = compatible_pairs[0]['contours']
                    patient_status[patient_id]['status'] = 'success'
                    patient_status[patient_id]['contours'] = compatible_pairs[0]['contours']
                    
                    longitudinal_data[patient_id] = {
                        'compatible_pairs': compatible_pairs,
                        'series_data': series_data,
                        'rtstruct_files': rtstruct_files,
                        'available_modalities': list(series_data.keys())
                    }
                    
                    st.success(f"  - ✅ Found {len(compatible_pairs)} compatible series for `{patient_id}`")
                else:
                    st.error(f"  - ❌ No compatible pairs found for `{patient_id}`")
                    patient_status[patient_id]['issues'].append("No compatible pairs found")
                    
                    longitudinal_data[patient_id] = {
                        'compatible_pairs': [],
                        'series_data': series_data,
                        'rtstruct_files': rtstruct_files,
                        'available_modalities': list(series_data.keys())
                    }

            except Exception as e:
                st.error(f"  - Critical error processing `{patient_id}`: {e}")
                patient_status[patient_id]['issues'].append(f"Critical error: {e}")
        
        progress_tracker.complete("Enhanced DICOM scanning complete")

    # processing.py (CONTINUATION - add this to the end of the previous code)

# The previous code cut off at line ~1850 inside scan_uploaded_data_for_contours_enhanced
# Here's the completion:

    return (
        sorted(list(all_contours)), 
        patient_contour_data, 
        patient_status, 
        sorted(list(available_modalities)), 
        longitudinal_data
    )


def scan_uploaded_data_for_contours(data_path, selected_modality='CT'):
    """
    Original scanning function for single modality processing (PRESERVED).
    This is the legacy function - kept for backward compatibility.
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
        
        for i, patient_id in enumerate(patient_dirs):
            status.update(label=f"Scanning patient {i+1}/{total_patients}: {patient_id}")
            patient_path = os.path.join(data_path, patient_id)
            
            patient_status[patient_id] = {'status': 'error', 'issues': [], 'contours': []}
            rtstruct_files = []
            series_dirs = {}

            try:
                st.write(f"  - Identifying all DICOM series for `{patient_id}`...")
                
                for dirpath, _, filenames in os.walk(patient_path):
                    for f in filenames:
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

                compatible_found = False
                for rt_path in rtstruct_files:
                    for series_path in filtered_series_dirs.values():
                        try:
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
                            break
                        except Exception:
                            continue
                    if compatible_found:
                        break

                if not compatible_found:
                    st.error(f"  - ❌ No compatible RTSTRUCT/{selected_modality} pair found for `{patient_id}`.")
                    patient_status[patient_id]['issues'].append(f"No compatible RTSTRUCT/{selected_modality} pair found")

            except Exception as e:
                st.error(f"  - A critical error occurred while processing `{patient_id}`: {e}")
                patient_status[patient_id]['issues'].append(f"Critical error: {e}")
        
        status.update(label="Scan complete!", state="complete")

    return sorted(list(all_contours)), patient_contour_data, patient_status, sorted(list(available_modalities))


# =============================================================================
# MAIN PREPROCESSING FUNCTIONS (MULTI-ROI ENABLED)
# =============================================================================

def preprocess_uploaded_data(
    uploaded_data_path,
    selected_roi,  # ✅ MULTI-ROI: Accepts ROI name (processes ONE ROI per call)
    primary_modality
):
    """
    Main preprocessing function - processes ONE ROI at a time with ULTIMATE ROBUST mask generation.
    
    Multi-ROI Strategy:
    - This function processes ONE ROI completely through the robust pipeline
    - UI layer calls this function multiple times (once per ROI) for multi-ROI processing
    - Each call generates one complete dataset row: (Patient, Series, ROI, Image, Mask)
    - UI concatenates results from multiple calls
    - ZERO changes to existing robust workflow - all 5 mask generation methods preserved
    
    Parameters:
        uploaded_data_path: Path to organized DICOM data
        selected_roi: Name of ROI to process (e.g., "GTV", "CTV", "Parotid_L")
        primary_modality: Imaging modality (e.g., "CT", "MR", "PT")
    
    Returns:
        (DataFrame, summary_dict): Dataset records and processing statistics
    """
    # Create output directory
    output_dir = tempfile.mkdtemp(prefix="radiomics_nifti_")
    st.session_state['temp_output_dir'] = output_dir

    dataset_records = []
    failed_patients = {}
    recovery_stats = {
        'robust_rt_utils': 0,
        'robust_sitk_skimage': 0,
        'robust_enhanced_coord_transform': 0,
        'robust_morphology_enhanced': 0,
        'robust_direct_dicom': 0,
        'alternative_roi': 0,
        'fallback_placeholder': 0,
        'conversion_rescued': 0,
        'alternative_format_saved': 0,
        'failed': 0
    }

    if not uploaded_data_path or not os.path.isdir(uploaded_data_path):
        return pd.DataFrame(), {
            'total_patients': 0,
            'successful_patients': 0,
            'failed_patients': {'error': 'Invalid uploaded_data_path'},
            'recovery_statistics': recovery_stats
        }

    patient_dirs = [d for d in os.listdir(uploaded_data_path) if os.path.isdir(os.path.join(uploaded_data_path, d))]
    total_patients = len(patient_dirs)

    # UI progress handles (may be None if not set)
    progress_bar = st.session_state.get('ui_progress_bar')
    progress_text = st.session_state.get('ui_progress_text')
    status_placeholder = st.session_state.get('ui_status_placeholder')

    for i, patient_id in enumerate(patient_dirs):
        try:
            current_progress = (i + 1) / max(1, total_patients)
            if progress_bar:
                progress_bar.progress(current_progress)
            if progress_text:
                progress_text.text(f"Processing patient {i+1}/{total_patients}: {patient_id} ({current_progress*100:.1f}%)")
            if status_placeholder:
                status_placeholder.info(f"🔄 Processing {patient_id} - ROI: {selected_roi or 'default'}...")

            patient_path = os.path.join(uploaded_data_path, patient_id)

            # Gather RTSTRUCTs and series candidates
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
                        elif modality == primary_modality:
                            series_uid = getattr(dcm, 'SeriesInstanceUID', 'UnknownSeries')
                            series_candidates.append({
                                'path': root,
                                'series_uid': series_uid,
                                'file_count': len([x for x in os.listdir(root) if x.lower().endswith(('.dcm', '.ima'))])
                            })
                    except Exception:
                        continue

            if not series_candidates:
                failed_patients[patient_id] = {'reason': f"No {primary_modality} image series found"}
                recovery_stats['failed'] += 1
                if status_placeholder:
                    status_placeholder.warning(f"⚠️ Skipping {patient_id}: no {primary_modality} series")
                continue

            # Choose best series by file_count
            unique_series = {}
            for cand in series_candidates:
                uid = cand['series_uid']
                if uid not in unique_series or cand['file_count'] > unique_series[uid]['file_count']:
                    unique_series[uid] = cand
            best_series = max(unique_series.values(), key=lambda x: x['file_count'])
            series_path = best_series['path']

            if not rtstruct_files:
                failed_patients[patient_id] = {'reason': "No RTSTRUCT files found"}
                recovery_stats['failed'] += 1
                if status_placeholder:
                    status_placeholder.warning(f"⚠️ Skipping {patient_id}: no RTSTRUCT")
                continue

            # Find compatible RTSTRUCT
            compatible_pair = None
            for rt_file in rtstruct_files:
                try:
                    rtstruct = RTStructBuilder.create_from(
                        dicom_series_path=series_path,
                        rt_struct_path=rt_file
                    )
                    compatible_pair = (series_path, rt_file, rtstruct)
                    break
                except Exception:
                    continue

            if not compatible_pair:
                failed_patients[patient_id] = {'reason': f"No compatible RTSTRUCT/{primary_modality} pair found"}
                recovery_stats['failed'] += 1
                if status_placeholder:
                    status_placeholder.warning(f"⚠️ Skipping {patient_id}: no compatible pair")
                continue

            series_path, rt_file, rtstruct = compatible_pair

            # ✅ MULTI-ROI: Find the selected ROI (case-insensitive matching)
            actual_roi_name = None
            available_rois = rtstruct.get_roi_names()
            
            if selected_roi:
                for roi in available_rois:
                    if selected_roi.lower() in roi.lower():
                        actual_roi_name = roi
                        break
                
                if not actual_roi_name:
                    # ROI not found - skip this patient for this ROI
                    failed_patients[patient_id] = {
                        'reason': f"Target contour '{selected_roi}' not found",
                        'details': f"Available contours: {available_rois}"
                    }
                    recovery_stats['failed'] += 1
                    if status_placeholder:
                        status_placeholder.warning(f"⚠️ Skipping {patient_id}: ROI '{selected_roi}' not found")
                    continue
            else:
                # Backward compatibility: if no ROI specified, use first available
                actual_roi_name = available_rois[0] if available_rois else None
                if actual_roi_name is None:
                    failed_patients[patient_id] = {'reason': "RTSTRUCT contains no ROIs"}
                    recovery_stats['failed'] += 1
                    continue

            # Validate ROI
            is_valid, validation_msg = validate_rtstruct_contours(rtstruct, actual_roi_name)
            if not is_valid:
                failed_patients[patient_id] = {'reason': f"ROI validation failed: {validation_msg}"}
                recovery_stats['failed'] += 1
                continue

            # Read image series
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(series_path)
            if not dicom_names:
                failed_patients[patient_id] = {'reason': "No DICOM files in series path"}
                recovery_stats['failed'] += 1
                continue
            reader.SetFileNames(dicom_names)
            image_sitk = reader.Execute()

            # ULTIMATE ROBUST MASK GENERATION (5 methods)
            if status_placeholder:
                status_placeholder.info(f"🎯 Generating mask for {patient_id} - ROI: {actual_roi_name}...")
            
            mask_3d, used_roi_name, recovery_method = ultimate_mask_recovery_robust(
                rtstruct, actual_roi_name, image_sitk, series_path, patient_id, status_placeholder
            )

            if recovery_method and recovery_method in recovery_stats:
                recovery_stats[recovery_method] += 1

            # Try alternative ROI names if all 5 methods failed
            if mask_3d is None:
                similar_rois = find_similar_roi_names(available_rois, selected_roi or actual_roi_name)
                for alt_roi, _, _ in similar_rois[:3]:
                    alt_mask, alt_used_roi, alt_recovery_method = ultimate_mask_recovery_robust(
                        rtstruct, alt_roi, image_sitk, series_path, patient_id, status_placeholder
                    )
                    if alt_mask is not None:
                        mask_3d = alt_mask
                        used_roi_name = alt_used_roi
                        recovery_method = f"alternative_roi_{alt_recovery_method}"
                        recovery_stats['alternative_roi'] += 1
                        if status_placeholder:
                            status_placeholder.success(f"  - ✅ Alternative ROI '{alt_roi}' succeeded")
                        break

            # Final fallback
            if mask_3d is None:
                mask_3d = create_fallback_mask(image_sitk, patient_id, status_placeholder)
                if mask_3d is not None:
                    recovery_method = "fallback_placeholder"
                    used_roi_name = f"{selected_roi or actual_roi_name}_fallback"
                    recovery_stats['fallback_placeholder'] += 1

            if mask_3d is None or np.sum(mask_3d > 0) == 0:
                failed_patients[patient_id] = {'reason': "Mask generation failed (all methods)"}
                recovery_stats['failed'] += 1
                if status_placeholder:
                    status_placeholder.error(f"💥 Mask generation failed for {patient_id}")
                continue

            # Conversion to SITK
            mask_sitk, conversion_debug = robust_mask_to_sitk_conversion(
                mask_3d, image_sitk, patient_id, status_placeholder
            )
            
            if mask_sitk is None:
                failed_patients[patient_id] = {
                    'reason': "Mask conversion to SITK failed",
                    'details': conversion_debug
                }
                recovery_stats['failed'] += 1
                continue

            # Prepare output paths
            patient_output_dir = os.path.join(output_dir, patient_id)
            os.makedirs(patient_output_dir, exist_ok=True)
            output_image_path = os.path.join(patient_output_dir, "image.nii.gz")
            output_mask_path = os.path.join(patient_output_dir, "mask.nii.gz")

            # Handle dimension mismatches
            try:
                if mask_sitk.GetSize() != image_sitk.GetSize():
                    mask_sitk, bypassed = bypass_resampling_when_possible(
                        mask_sitk, image_sitk, patient_id, status_placeholder
                    )
                    if not bypassed:
                        resampled_mask, final_voxels = smart_mask_resampling_with_coordinate_preservation(
                            mask_3d, mask_sitk, image_sitk, patient_id, status_placeholder
                        )
                        if resampled_mask is None or final_voxels == 0:
                            failed_patients[patient_id] = {'reason': "Smart resampling failed"}
                            recovery_stats['failed'] += 1
                            continue
                        mask_sitk = resampled_mask
            except Exception as e:
                failed_patients[patient_id] = {'reason': "Dimension handling error", 'details': str(e)}
                recovery_stats['failed'] += 1
                continue

            # Save image and mask
            try:
                sitk.WriteImage(image_sitk, output_image_path)
            except Exception as e:
                failed_patients[patient_id] = {'reason': "Image saving failed", 'details': str(e)}
                recovery_stats['failed'] += 1
                continue

            mask_saved_successfully, final_voxel_count, actual_mask_path = robust_mask_file_saving(
                mask_sitk, output_mask_path, patient_id, status_placeholder
            )

            if not mask_saved_successfully:
                alt_mask_path, alt_voxel_count = alternative_mask_saving_formats(
                    mask_3d, image_sitk, patient_output_dir, patient_id, status_placeholder
                )
                if alt_mask_path and alt_voxel_count > 0:
                    actual_mask_path = alt_mask_path
                    final_voxel_count = alt_voxel_count
                    mask_saved_successfully = True
                    recovery_stats['alternative_format_saved'] += 1

            if not mask_saved_successfully or final_voxel_count == 0:
                failed_patients[patient_id] = {'reason': "Mask saving failed"}
                recovery_stats['failed'] += 1
                continue
            # Get series description from DICOM
            try:
                dcm = pydicom.dcmread(dicom_names[0], stop_before_pixels=True)
                series_description = getattr(dcm, 'SeriesDescription', 'Unknown_Series')
            except Exception:
                series_description = 'Unknown_Series'
            # ✅ SUCCESS: Create dataset record with ROI tracking
            dataset_records.append({
                'patient_id': patient_id,
                'image_path': output_image_path,
                'mask_path': actual_mask_path,
                'roi_name': used_roi_name,  # ✅ Track which ROI was processed
                'series_description': series_description,  # ✅ ADD THIS if missing
                'modality': primary_modality,
                'recovery_method': recovery_method,
                'original_roi_target': selected_roi or actual_roi_name,  # ✅ Track which ROI was requested
                'needs_review': (recovery_method != "robust_rt_utils"),
                'voxel_count': int(final_voxel_count)
            })

            if status_placeholder:
                method_display = (recovery_method or "").replace('robust_', '').replace('_', ' ').title()
                status_placeholder.success(
                    f"🎉 SUCCESS: {patient_id} - ROI '{selected_roi or actual_roi_name}' "
                    f"processed ({final_voxel_count:,} voxels)"
                )

        except Exception as e:
            failed_patients[patient_id] = {'reason': "Critical processing error", 'details': str(e)}
            recovery_stats['failed'] += 1
            if status_placeholder:
                status_placeholder.error(f"💥 Critical error for {patient_id}: {e}")
            continue

    # Finalize
    if progress_bar:
        progress_bar.progress(1.0)
    if progress_text:
        success_count = len(dataset_records)
        progress_text.text(f"Processing complete! {success_count}/{total_patients} patients processed")

    processing_summary = {
        'total_patients': total_patients,
        'successful_patients': len(dataset_records),
        'failed_patients': failed_patients,
        'recovery_statistics': recovery_stats
    }

    return pd.DataFrame(dataset_records), processing_summary


def preprocess_uploaded_data_enhanced(
    uploaded_data_path,
    selected_roi,
    selected_modalities,
    multi_series_mode=True,
    selected_series=[]
):
    """
    Enhanced preprocessing for multi-series/multi-modality data.
    FIXED: Complete try-except blocks, proper indentation, metadata preservation.
    
    Multi-ROI Strategy:
    - Accepts selected_roi parameter (ONE ROI per call)
    - For multi-series: processes selected_roi across multiple series
    - Calls preprocess_uploaded_data() per series
    - Aggregates results with series metadata
    - UI layer orchestrates multi-ROI by calling this function multiple times
    
    Parameters:
        uploaded_data_path: Path to organized DICOM data
        selected_roi: Name of ONE ROI to process across all series
        selected_modalities: List of modalities to process
        multi_series_mode: Enable multi-series processing
        selected_series: List of specific series to process (optional)
    
    Returns:
        (DataFrame, summary_dict): Combined dataset and statistics
    """
    # Defensive defaults
    if selected_modalities is None or len(selected_modalities) == 0:
        selected_modalities = ['CT']
    if selected_series is None:
        selected_series = []

    all_records = []
    combined_summary = {
        'total_patients': 0,
        'successful_patients': 0,
        'failed_patients': {},
        'series_processed': 0,
        'processing_mode': 'multi_series' if multi_series_mode else 'single_series'
    }

    # Multi-series mode with specific series list
    if multi_series_mode and selected_series:
        total_series = len(selected_series)
        
        for idx, series_info in enumerate(selected_series):
            try:
                # Progress update
                if st.session_state.get('ui_status_placeholder'):
                    st.session_state['ui_status_placeholder'].info(
                        f"🔄 Processing series {idx+1}/{total_series} - "
                        f"Modality: {series_info.get('modality', 'Unknown')} - "
                        f"ROI: {selected_roi}"
                    )
                
                # Check if this series has the selected ROI
                series_has_roi = True
                if selected_roi:
                    series_contours = series_info.get('contours', [])
                    if series_contours:
                        series_has_roi = any(selected_roi.lower() in c.lower() for c in series_contours)
                        
                        if not series_has_roi:
                            # Skip this series - ROI not available
                            if st.session_state.get('ui_status_placeholder'):
                                st.session_state['ui_status_placeholder'].info(
                                    f"ℹ️ Skipping series {idx+1}: ROI '{selected_roi}' not available"
                                )
                            continue
                
                # Process this series with the selected ROI
                series_df, series_summary = preprocess_uploaded_data(
                    uploaded_data_path,
                    selected_roi,
                    series_info.get('modality', selected_modalities[0] if selected_modalities else 'CT')
                )

                # Add series metadata
                if not series_df.empty:
                    series_df = series_df.copy()
                    
                    # Ensure roi_name column exists
                    if 'roi_name' not in series_df.columns:
                        series_df['roi_name'] = selected_roi
                    
                    # Add comprehensive series metadata
                    series_df['series_description'] = series_info.get('series_description', 'Unknown')
                    series_df['timepoint'] = series_info.get('timepoint', '')
                    series_df['series_uid'] = series_info.get('series_uid', '')
                    series_df['study_date'] = series_info.get('study_date', '')
                    series_df['slice_count'] = series_info.get('slice_count', 0)
                    series_df['processing_order'] = idx + 1
                    series_df['modality'] = series_info.get('modality', selected_modalities[0] if selected_modalities else 'CT')
                    
                    all_records.append(series_df)

                combined_summary['total_patients'] += series_summary.get('total_patients', 0)
                combined_summary['successful_patients'] += series_summary.get('successful_patients', 0)
                combined_summary['series_processed'] += 1

                # Merge failed patients with series context
                for pid, failinfo in series_summary.get('failed_patients', {}).items():
                    key = f"{pid}_{series_info.get('modality', '')}_{series_info.get('timepoint', '')}"
                    combined_summary['failed_patients'][key] = failinfo

            except Exception as e:
                combined_summary['failed_patients'][f"series_idx_{idx}"] = {'reason': str(e)}
                if st.session_state.get('ui_status_placeholder'):
                    st.session_state['ui_status_placeholder'].error(f"❌ Series {idx+1} failed: {str(e)}")
                continue

        if all_records:
            final_df = pd.concat(all_records, ignore_index=True)
            final_df['multi_series_session'] = int(time.time())
        else:
            final_df = pd.DataFrame()

        return final_df, combined_summary

    else:
        # Single-series mode: call single preprocessing function
        result_df, result_summary = preprocess_uploaded_data(
            uploaded_data_path,
            selected_roi,
            selected_modalities[0] if selected_modalities else 'CT'
        )
        
        combined_summary.update(result_summary if isinstance(result_summary, dict) else {})
        return result_df, combined_summary


# =============================================================================
# ADVANCED SEARCH PREPROCESSING (USER-ASSISTED PATIENT / SERIES / ROI SELECTION)
# =============================================================================

def preprocess_selected_combinations(combinations, output_dir=None):
    """
    Process an explicit list of user-assembled (patient, series, ROI) combinations.

    Each combination is a dict describing a single unit of work for the robust
    mask generation pipeline. This is used by the "Advanced Search" workflow
    where the user has already categorized the scanned data into three boxes
    (patients, series, ROIs) and asked to apply the selection to all patients
    or to a selected subset.

    Required keys per combination:
        - patient_id (str)
        - series_path (str): directory containing the DICOM image series
        - rtstruct_path (str): path to RTSTRUCT DICOM file
        - roi_name (str): ROI name to process

    Optional keys per combination (propagated to the output dataset):
        - modality, timepoint, series_uid, series_description, study_date,
          slice_count

    Returns:
        (pd.DataFrame, summary_dict)
        DataFrame contains one row per successfully processed combination
        with columns compatible with the rest of the pipeline:
        patient_id, image_path, mask_path, roi_name, series_description,
        modality, series_uid, timepoint, recovery_method, needs_review,
        voxel_count, original_roi_target.
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="radiomics_advanced_")
    os.makedirs(output_dir, exist_ok=True)
    st.session_state['temp_output_dir'] = output_dir

    dataset_records = []
    failed_combinations = {}
    recovery_stats = {
        'robust_rt_utils': 0,
        'robust_sitk_skimage': 0,
        'robust_enhanced_coord_transform': 0,
        'robust_morphology_enhanced': 0,
        'robust_direct_dicom': 0,
        'alternative_roi': 0,
        'fallback_placeholder': 0,
        'alternative_format_saved': 0,
        'failed': 0
    }

    total = len(combinations)
    progress_bar = st.session_state.get('ui_progress_bar')
    progress_text = st.session_state.get('ui_progress_text')
    status_placeholder = st.session_state.get('ui_status_placeholder')

    patients_seen = set()

    for idx, combo in enumerate(combinations):
        patient_id = combo.get('patient_id', f'patient_{idx}')
        series_path = combo.get('series_path')
        rtstruct_path = combo.get('rtstruct_path')
        roi_name = combo.get('roi_name')
        modality = combo.get('modality', 'CT')
        series_uid = combo.get('series_uid', '')
        series_description = combo.get('series_description', 'Unknown_Series')
        timepoint = combo.get('timepoint', '')
        study_date = combo.get('study_date', '')
        slice_count = combo.get('slice_count', 0)

        combo_key = f"{patient_id}|{series_uid or series_description}|{roi_name}"
        patients_seen.add(patient_id)

        try:
            current_progress = (idx + 1) / max(1, total)
            if progress_bar:
                progress_bar.progress(current_progress)
            if progress_text:
                progress_text.text(
                    f"Processing {idx+1}/{total}: {patient_id} / "
                    f"{series_description or series_uid or 'series'} / ROI={roi_name}"
                )
            if status_placeholder:
                status_placeholder.info(
                    f"🔄 [{idx+1}/{total}] {patient_id} - series '{series_description}' - ROI '{roi_name}'"
                )

            if not series_path or not os.path.isdir(series_path):
                failed_combinations[combo_key] = {'reason': 'Invalid or missing series_path'}
                recovery_stats['failed'] += 1
                continue
            if not rtstruct_path or not os.path.isfile(rtstruct_path):
                failed_combinations[combo_key] = {'reason': 'Invalid or missing rtstruct_path'}
                recovery_stats['failed'] += 1
                continue
            if not roi_name:
                failed_combinations[combo_key] = {'reason': 'No ROI name provided'}
                recovery_stats['failed'] += 1
                continue

            try:
                rtstruct = RTStructBuilder.create_from(
                    dicom_series_path=series_path,
                    rt_struct_path=rtstruct_path
                )
            except Exception as e:
                failed_combinations[combo_key] = {
                    'reason': 'RTSTRUCT/series pair incompatible',
                    'details': str(e)
                }
                recovery_stats['failed'] += 1
                continue

            available_rois = rtstruct.get_roi_names()
            actual_roi_name = None
            for roi in available_rois:
                if roi == roi_name or roi_name.lower() == roi.lower():
                    actual_roi_name = roi
                    break
            if actual_roi_name is None:
                for roi in available_rois:
                    if roi_name.lower() in roi.lower():
                        actual_roi_name = roi
                        break
            if actual_roi_name is None:
                failed_combinations[combo_key] = {
                    'reason': f"ROI '{roi_name}' not found in RTSTRUCT",
                    'details': f"Available: {available_rois}"
                }
                recovery_stats['failed'] += 1
                continue

            is_valid, validation_msg = validate_rtstruct_contours(rtstruct, actual_roi_name)
            if not is_valid:
                failed_combinations[combo_key] = {'reason': f"ROI validation failed: {validation_msg}"}
                recovery_stats['failed'] += 1
                continue

            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(series_path)
            if not dicom_names:
                failed_combinations[combo_key] = {'reason': 'No DICOM files in series path'}
                recovery_stats['failed'] += 1
                continue
            reader.SetFileNames(dicom_names)
            image_sitk = reader.Execute()

            mask_3d, used_roi_name, recovery_method = ultimate_mask_recovery_robust(
                rtstruct, actual_roi_name, image_sitk, series_path, patient_id, status_placeholder
            )
            if recovery_method and recovery_method in recovery_stats:
                recovery_stats[recovery_method] += 1

            if mask_3d is None:
                similar_rois = find_similar_roi_names(available_rois, roi_name)
                for alt_roi, _, _ in similar_rois[:3]:
                    alt_mask, alt_used_roi, alt_recovery_method = ultimate_mask_recovery_robust(
                        rtstruct, alt_roi, image_sitk, series_path, patient_id, status_placeholder
                    )
                    if alt_mask is not None:
                        mask_3d = alt_mask
                        used_roi_name = alt_used_roi
                        recovery_method = f"alternative_roi_{alt_recovery_method}"
                        recovery_stats['alternative_roi'] += 1
                        break

            if mask_3d is None:
                mask_3d = create_fallback_mask(image_sitk, patient_id, status_placeholder)
                if mask_3d is not None:
                    recovery_method = 'fallback_placeholder'
                    used_roi_name = f"{roi_name}_fallback"
                    recovery_stats['fallback_placeholder'] += 1

            if mask_3d is None or np.sum(mask_3d > 0) == 0:
                failed_combinations[combo_key] = {'reason': 'Mask generation failed (all methods)'}
                recovery_stats['failed'] += 1
                continue

            mask_sitk, conversion_debug = robust_mask_to_sitk_conversion(
                mask_3d, image_sitk, patient_id, status_placeholder
            )
            if mask_sitk is None:
                failed_combinations[combo_key] = {
                    'reason': 'Mask conversion to SITK failed',
                    'details': conversion_debug
                }
                recovery_stats['failed'] += 1
                continue

            # Distinct output dir per (patient, series, ROI)
            safe_series = (series_description or series_uid or 'series').replace('/', '_').replace(' ', '_')[:40]
            safe_roi = (used_roi_name or roi_name).replace('/', '_').replace(' ', '_')[:40]
            combo_output_dir = os.path.join(output_dir, patient_id, f"{safe_series}__{safe_roi}")
            os.makedirs(combo_output_dir, exist_ok=True)
            output_image_path = os.path.join(combo_output_dir, "image.nii.gz")
            output_mask_path = os.path.join(combo_output_dir, "mask.nii.gz")

            try:
                if mask_sitk.GetSize() != image_sitk.GetSize():
                    mask_sitk, bypassed = bypass_resampling_when_possible(
                        mask_sitk, image_sitk, patient_id, status_placeholder
                    )
                    if not bypassed:
                        resampled_mask, final_voxels = smart_mask_resampling_with_coordinate_preservation(
                            mask_3d, mask_sitk, image_sitk, patient_id, status_placeholder
                        )
                        if resampled_mask is None or final_voxels == 0:
                            failed_combinations[combo_key] = {'reason': 'Smart resampling failed'}
                            recovery_stats['failed'] += 1
                            continue
                        mask_sitk = resampled_mask
            except Exception as e:
                failed_combinations[combo_key] = {'reason': 'Dimension handling error', 'details': str(e)}
                recovery_stats['failed'] += 1
                continue

            try:
                sitk.WriteImage(image_sitk, output_image_path)
            except Exception as e:
                failed_combinations[combo_key] = {'reason': 'Image saving failed', 'details': str(e)}
                recovery_stats['failed'] += 1
                continue

            mask_saved_successfully, final_voxel_count, actual_mask_path = robust_mask_file_saving(
                mask_sitk, output_mask_path, patient_id, status_placeholder
            )
            if not mask_saved_successfully:
                alt_mask_path, alt_voxel_count = alternative_mask_saving_formats(
                    mask_3d, image_sitk, combo_output_dir, patient_id, status_placeholder
                )
                if alt_mask_path and alt_voxel_count > 0:
                    actual_mask_path = alt_mask_path
                    final_voxel_count = alt_voxel_count
                    mask_saved_successfully = True
                    recovery_stats['alternative_format_saved'] += 1

            if not mask_saved_successfully or final_voxel_count == 0:
                failed_combinations[combo_key] = {'reason': 'Mask saving failed'}
                recovery_stats['failed'] += 1
                continue

            if not series_description or series_description == 'Unknown_Series':
                try:
                    dcm = pydicom.dcmread(dicom_names[0], stop_before_pixels=True)
                    series_description = getattr(dcm, 'SeriesDescription', series_description)
                except Exception:
                    pass

            dataset_records.append({
                'patient_id': patient_id,
                'image_path': output_image_path,
                'mask_path': actual_mask_path,
                'roi_name': used_roi_name or roi_name,
                'original_roi_target': roi_name,
                'series_description': series_description,
                'series_uid': series_uid,
                'timepoint': timepoint,
                'study_date': study_date,
                'slice_count': slice_count,
                'modality': modality,
                'recovery_method': recovery_method,
                'needs_review': (recovery_method != 'robust_rt_utils'),
                'voxel_count': int(final_voxel_count)
            })

        except Exception as e:
            failed_combinations[combo_key] = {'reason': 'Critical processing error', 'details': str(e)}
            recovery_stats['failed'] += 1
            continue

    if progress_bar:
        progress_bar.progress(1.0)
    if progress_text:
        progress_text.text(f"Advanced preprocessing complete: {len(dataset_records)}/{total} combinations succeeded")

    summary = {
        'total_combinations': total,
        'successful_combinations': len(dataset_records),
        'total_patients': len(patients_seen),
        'successful_patients': len({r['patient_id'] for r in dataset_records}),
        'failed_patients': failed_combinations,
        'recovery_statistics': recovery_stats,
        'processing_mode': 'advanced_search'
    }
    return pd.DataFrame(dataset_records), summary


# =============================================================================
# END OF FILE
# =============================================================================
