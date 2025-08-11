# processing.py
"""
This module contains all functions related to data loading, validation,
organization, and pre-processing. It prepares the raw DICOM data
for the feature extraction step.

Updated: 2025-07-23 18:35:12 UTC by Medphysicist
Added robust mask conversion and saving system to fix "Even after successful mask generation, final file is empty".
Includes multi-library mask generation + comprehensive saving/conversion debugging and fallbacks.
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
import cv2  # Added for specific OpenCV error handling
from skimage import draw
import scipy.ndimage as ndimage

# --- ROBUST MASK CONVERSION AND SAVING SYSTEM ---
def smart_mask_resampling_with_coordinate_preservation(mask_3d, mask_sitk, image_sitk, patient_id, status_placeholder=None):
    """
    Smart resampling that preserves mask voxels by trying multiple approaches.
    """
    
    if status_placeholder:
        status_placeholder.info(f"  - Smart resampling for {patient_id}...")
    
    # Get original voxel count
    original_voxels = np.sum(mask_3d > 0)
    
    # Method 1: Skip resampling if sizes are close enough
    mask_size = mask_sitk.GetSize()
    image_size = image_sitk.GetSize()
    
    size_diff = max(abs(a - b) for a, b in zip(mask_size, image_size))
    if size_diff <= 2:  # Allow small differences
        if status_placeholder:
            status_placeholder.info(f"  - Size difference is small ({size_diff}), skipping resampling")
        return mask_sitk, original_voxels
    
    # Method 2: Manual coordinate scaling (most reliable)
    try:
        if status_placeholder:
            status_placeholder.info(f"  - Trying manual coordinate scaling...")
        
        # Create new mask with exact image properties
        new_mask_array = np.zeros(sitk.GetArrayFromImage(image_sitk).shape, dtype=np.uint8)
        mask_array = sitk.GetArrayFromImage(mask_sitk)
        
        # Calculate scaling factors
        scale_z = new_mask_array.shape[0] / mask_array.shape[0]
        scale_y = new_mask_array.shape[1] / mask_array.shape[1]
        scale_x = new_mask_array.shape[2] / mask_array.shape[2]
        
        # Scale each voxel
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
        
        # Create SimpleITK image with exact image properties
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
            
            # Threshold to convert back to binary
            resampled_mask = sitk.BinaryThreshold(resampled_mask, 0.5, 2.0, 1, 0)
            
            # Check result
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
    
    # All methods failed
    return None, 0


def bypass_resampling_when_possible(mask_sitk, image_sitk, patient_id, status_placeholder=None):
    """
    Try to avoid resampling altogether.
    """
    mask_size = mask_sitk.GetSize()
    image_size = image_sitk.GetSize()
    
    if mask_size == image_size:
        if status_placeholder:
            status_placeholder.info(f"  - Sizes match exactly, no resampling needed for {patient_id}")
        return mask_sitk, True
    
    # Allow small differences
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

def debug_mask_conversion_process(mask_3d, patient_id, status_placeholder=None):
    """
    Debug the mask conversion and saving process step by step.
    """
    debug_info = {}
    
    try:
        # Step 1: Check input mask
        debug_info['input_mask_type'] = type(mask_3d).__name__
        debug_info['input_mask_shape'] = mask_3d.shape
        debug_info['input_mask_dtype'] = mask_3d.dtype
        debug_info['input_nonzero_voxels'] = int(np.sum(mask_3d > 0))
        debug_info['input_unique_values'] = np.unique(mask_3d).tolist()
        debug_info['input_min_max'] = [float(np.min(mask_3d)), float(np.max(mask_3d))]
        
        if status_placeholder:
            status_placeholder.info(f"  - DEBUG {patient_id}: Input mask has {debug_info['input_nonzero_voxels']} voxels, dtype={debug_info['input_mask_dtype']}")
        
        # Step 2: Convert to uint8
        mask_uint8 = mask_3d.astype(np.uint8)
        debug_info['uint8_nonzero_voxels'] = int(np.sum(mask_uint8 > 0))
        debug_info['uint8_unique_values'] = np.unique(mask_uint8).tolist()
        
        # Step 3: Check for conversion issues
        if debug_info['input_nonzero_voxels'] != debug_info['uint8_nonzero_voxels']:
            debug_info['conversion_issue'] = True
            if status_placeholder:
                status_placeholder.warning(f"  - ⚠️ CONVERSION ISSUE: {debug_info['input_nonzero_voxels']} -> {debug_info['uint8_nonzero_voxels']} voxels")
        
        return debug_info, mask_uint8
        
    except Exception as e:
        debug_info['conversion_error'] = str(e)
        return debug_info, None


def robust_mask_to_sitk_conversion(mask_3d, image_sitk, patient_id, status_placeholder=None):
    """
    Robust conversion of mask array to SimpleITK image with multiple fallback methods.
    """
    
    # Debug the conversion process
    debug_info, mask_uint8 = debug_mask_conversion_process(mask_3d, patient_id, status_placeholder)
    
    if mask_uint8 is None:
        return None, debug_info
    
    conversion_methods = [
        ("standard_conversion", lambda: sitk.GetImageFromArray(mask_uint8)),
        ("explicit_uint8_conversion", lambda: sitk.GetImageFromArray(mask_uint8.astype(np.uint8))),
        ("binary_conversion", lambda: sitk.GetImageFromArray((mask_uint8 > 0).astype(np.uint8))),
        ("clipped_conversion", lambda: sitk.GetImageFromArray(np.clip(mask_uint8, 0, 1).astype(np.uint8))),
        ("explicit_copy_conversion", lambda: sitk.GetImageFromArray(np.array(mask_uint8, dtype=np.uint8, copy=True)))
    ]
    
    for method_name, conversion_func in conversion_methods:
        try:
            if status_placeholder:
                status_placeholder.info(f"  - Trying {method_name} for {patient_id}...")
            
            # Convert to SimpleITK
            mask_sitk = conversion_func()
            
            # Copy spatial properties
            mask_sitk.SetOrigin(image_sitk.GetOrigin())
            mask_sitk.SetSpacing(image_sitk.GetSpacing())
            mask_sitk.SetDirection(image_sitk.GetDirection())
            
            # Verify the conversion worked
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
    """
    Robust mask file saving with multiple methods and verification.
    """
    
    saving_methods = [
        ("standard_write", lambda: sitk.WriteImage(mask_sitk, output_path)),
        ("explicit_cast_write", lambda: sitk.WriteImage(sitk.Cast(mask_sitk, sitk.sitkUInt8), output_path)),
        ("compressed_write", lambda: sitk.WriteImage(mask_sitk, output_path, True)),  # Enable compression
        ("uncompressed_write", lambda: sitk.WriteImage(mask_sitk, output_path.replace('.nii.gz', '.nii'))),
    ]
    
    for method_name, saving_func in saving_methods:
        try:
            if status_placeholder:
                status_placeholder.info(f"  - Trying {method_name} for {patient_id}...")
            
            # Save the file
            saving_func()
            
            # Determine actual output path (in case of uncompressed_write)
            actual_output_path = output_path
            if method_name == "uncompressed_write":
                actual_output_path = output_path.replace('.nii.gz', '.nii')
            
            # Verify the saved file
            if os.path.exists(actual_output_path):
                # Try to read it back
                test_mask = sitk.ReadImage(actual_output_path)
                test_array = sitk.GetArrayFromImage(test_mask)
                saved_voxels = np.sum(test_array > 0)
                
                if saved_voxels > 0:
                    if status_placeholder:
                        status_placeholder.success(f"  - ✅ {method_name} successfully saved {patient_id} ({saved_voxels} voxels)")
                    return True, saved_voxels, actual_output_path
                else:
                    if status_placeholder:
                        status_placeholder.error(f"  - ❌ {method_name} saved empty file for {patient_id}")
                    # Try to remove the empty file
                    try:
                        os.remove(actual_output_path)
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
    """
    Try saving in alternative formats if NIfTI fails.
    """
    alternative_formats = [
        ("mask.mha", "MetaImage format"),
        ("mask.nrrd", "NRRD format"), 
        ("mask.vtk", "VTK format"),
        ("mask_uncompressed.nii", "Uncompressed NIfTI")
    ]
    
    for filename, description in alternative_formats:
        try:
            output_path = os.path.join(patient_output_dir, filename)
            
            if status_placeholder:
                status_placeholder.info(f"  - Trying {description} for {patient_id}...")
            
            # Convert mask
            mask_sitk, debug_info = robust_mask_to_sitk_conversion(mask_3d, image_sitk, patient_id, status_placeholder)
            
            if mask_sitk is not None:
                # Save in alternative format
                sitk.WriteImage(mask_sitk, output_path)
                
                # Verify
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


# --- MULTI-LIBRARY ROBUST MASK GENERATION ---

def create_mask_using_sitk_skimage(ds, roi_name, image_sitk, series_path):
    """
    Use SimpleITK + scikit-image for robust polygon processing.
    This handles coordinate transformations better than manual methods.
    """
    try:
        # Get image properties
        image_array = sitk.GetArrayFromImage(image_sitk)
        
        # Initialize mask
        mask_array = np.zeros(image_array.shape, dtype=np.uint8)
        
        # Find ROI number
        roi_number = None
        for roi_struct in ds.StructureSetROISequence:
            if getattr(roi_struct, 'ROIName', '').lower() == roi_name.lower():
                roi_number = getattr(roi_struct, 'ROINumber', None)
                break
        
        if roi_number is None:
            print(f"ROI '{roi_name}' not found in StructureSetROISequence")
            return None
        
        # Process contours with scikit-image
        processed_slices = 0
        total_points_processed = 0
        
        for roi_contour in ds.ROIContourSequence:
            if getattr(roi_contour, 'ReferencedROINumber', None) == roi_number:
                contour_sequence = getattr(roi_contour, 'ContourSequence', [])
                print(f"Found {len(contour_sequence)} contour slices for ROI '{roi_name}'")
                
                for contour in contour_sequence:
                    contour_data = getattr(contour, 'ContourData', [])
                    if len(contour_data) < 9:
                        continue
                    
                    # Get world coordinates
                    points_world = np.array(contour_data).reshape(-1, 3)
                    
                    # Transform using SimpleITK's built-in transformation
                    points_image = []
                    valid_points = 0
                    
                    for world_point in points_world:
                        try:
                            # Use SimpleITK's robust coordinate transformation
                            image_point = image_sitk.TransformPhysicalPointToIndex(world_point.tolist())
                            
                            # Check if point is within image bounds
                            if (0 <= image_point[0] < image_sitk.GetSize()[0] and
                                0 <= image_point[1] < image_sitk.GetSize()[1] and
                                0 <= image_point[2] < image_sitk.GetSize()[2]):
                                # Convert to ZYX ordering for numpy array
                                points_image.append([image_point[2], image_point[1], image_point[0]])
                                valid_points += 1
                        except:
                            # Skip points that can't be transformed
                            continue
                    
                    if len(points_image) < 3:
                        print(f"Insufficient valid points: {len(points_image)} (need at least 3)")
                        continue
                    
                    points_image = np.array(points_image)
                    total_points_processed += len(points_image)
                    
                    # Group by slice and use scikit-image for robust polygon filling
                    slice_groups = {}
                    for point in points_image:
                        z_idx = int(round(point[0]))
                        if 0 <= z_idx < mask_array.shape[0]:
                            if z_idx not in slice_groups:
                                slice_groups[z_idx] = []
                            slice_groups[z_idx].append([point[1], point[2]])  # Y, X for scikit-image
                    
                    # Fill polygons using scikit-image (more robust than cv2)
                    for z_idx, slice_points in slice_groups.items():
                        if len(slice_points) >= 3:
                            points_array = np.array(slice_points)
                            
                            try:
                                # Use scikit-image's robust polygon function
                                rr, cc = draw.polygon(points_array[:, 0], points_array[:, 1], 
                                                    shape=mask_array[z_idx].shape)
                                
                                # Ensure indices are within bounds
                                valid_indices = (rr >= 0) & (rr < mask_array.shape[1]) & \
                                              (cc >= 0) & (cc < mask_array.shape[2])
                                
                                if np.any(valid_indices):
                                    mask_array[z_idx, rr[valid_indices], cc[valid_indices]] = 1
                                    processed_slices += 1
                                
                            except Exception as e:
                                print(f"Failed to fill polygon on slice {z_idx}: {e}")
                                continue
                
                break
        
        final_voxel_count = np.sum(mask_array > 0)
        print(f"SimpleITK + scikit-image: {processed_slices} slices processed, {total_points_processed} points, {final_voxel_count} voxels created")
        
        return mask_array if final_voxel_count > 0 else None
        
    except Exception as e:
        print(f"SimpleITK + scikit-image approach failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_mask_using_morphology_enhancement(ds, roi_name, image_sitk, series_path):
    """
    Try the direct DICOM approach but with morphological enhancement.
    """
    try:
        # Start with direct DICOM processing
        mask_array = create_mask_from_dicom_contours_direct(ds, roi_name, image_sitk)
        
        if mask_array is None or np.sum(mask_array > 0) == 0:
            return None
            
        # Apply morphological operations to enhance the mask
        print(f"Applying morphological enhancement to mask with {np.sum(mask_array > 0)} initial voxels")
        
        # Close small holes
        mask_closed = ndimage.binary_closing(mask_array, structure=np.ones((3,3,3)))
        
        # Fill holes within each slice
        for z in range(mask_closed.shape[0]):
            if np.any(mask_closed[z]):
                mask_closed[z] = ndimage.binary_fill_holes(mask_closed[z])
        
        # Light dilation to ensure connectivity
        mask_dilated = ndimage.binary_dilation(mask_closed, structure=np.ones((2,2,2)))
        
        final_mask = mask_dilated.astype(np.uint8)
        final_voxel_count = np.sum(final_mask > 0)
        
        print(f"Morphology enhanced mask: {final_voxel_count} voxels (increased from {np.sum(mask_array > 0)})")
        
        return final_mask if final_voxel_count > 0 else None
        
    except Exception as e:
        print(f"Morphology enhancement failed: {e}")
        return None


def create_mask_from_dicom_contours_direct(ds, roi_name, image_sitk):
    """
    Bypass rt-utils completely and create mask directly from DICOM contour data.
    """
    try:
        # Get image properties
        image_array = sitk.GetArrayFromImage(image_sitk)
        spacing = np.array(image_sitk.GetSpacing())
        origin = np.array(image_sitk.GetOrigin())
        direction = np.array(image_sitk.GetDirection()).reshape(3, 3)
        
        # Initialize empty mask
        mask_array = np.zeros(image_array.shape, dtype=np.uint8)
        
        # Step 1: Find ROI number from StructureSetROISequence  
        roi_number = None
        for roi_struct in ds.StructureSetROISequence:
            if getattr(roi_struct, 'ROIName', '').lower() == roi_name.lower():
                roi_number = getattr(roi_struct, 'ROINumber', None)
                break
        
        if roi_number is None:
            print(f"ROI '{roi_name}' not found in StructureSetROISequence")
            return None
        
        # Step 2: Find contour data from ROIContourSequence
        contour_found = False
        processed_slices = 0
        
        for roi_contour in ds.ROIContourSequence:
            if getattr(roi_contour, 'ReferencedROINumber', None) == roi_number:
                contour_sequence = getattr(roi_contour, 'ContourSequence', [])
                print(f"Found {len(contour_sequence)} contour slices for ROI '{roi_name}'")
                
                for contour in contour_sequence:
                    contour_data = getattr(contour, 'ContourData', [])
                    if len(contour_data) < 9:  # Need at least 3 points (x,y,z each)
                        continue
                    
                    # Step 3: Extract 3D world coordinates
                    points_world = np.array(contour_data).reshape(-1, 3)
                    
                    # Step 4: Transform world coordinates to image coordinates
                    points_image = []
                    for world_point in points_world:
                        # Apply inverse transformation: image = (world - origin) / spacing
                        relative_pos = world_point - origin
                        
                        # Apply direction matrix if not identity
                        if not np.allclose(direction, np.eye(3)):
                            direction_inv = np.linalg.inv(direction)
                            rotated_pos = direction_inv @ relative_pos
                        else:
                            rotated_pos = relative_pos
                        
                        image_coord = rotated_pos / spacing
                        
                        # Convert to ZYX ordering for SimpleITK (reverse of XYZ)
                        points_image.append([image_coord[2], image_coord[1], image_coord[0]])
                    
                    points_image = np.array(points_image)
                    
                    # Step 5: Group points by slice (Z coordinate)
                    slice_groups = {}
                    for point in points_image:
                        z_idx = int(round(point[0]))
                        if 0 <= z_idx < mask_array.shape[0]:
                            if z_idx not in slice_groups:
                                slice_groups[z_idx] = []
                            # Store as [x, y] for 2D polygon filling
                            slice_groups[z_idx].append([int(round(point[2])), int(round(point[1]))])
                    
                    # Step 6: Fill polygons for each slice
                    for z_idx, slice_points in slice_groups.items():
                        if len(slice_points) >= 3:  # Need at least 3 points for polygon
                            points_2d = np.array(slice_points, dtype=np.int32)
                            
                            # Ensure points are within image bounds
                            points_2d[:, 0] = np.clip(points_2d[:, 0], 0, mask_array.shape[2] - 1)
                            points_2d[:, 1] = np.clip(points_2d[:, 1], 0, mask_array.shape[1] - 1)
                            
                            # Fill the polygon on this slice
                            cv2.fillPoly(mask_array[z_idx], [points_2d], 1)
                            contour_found = True
                            processed_slices += 1
                
                break
        
        if not contour_found:
            print(f"No valid contour data found for ROI '{roi_name}'")
            return None
        
        final_voxel_count = np.sum(mask_array > 0)
        print(f"Direct DICOM processing: {processed_slices} slices processed, {final_voxel_count} voxels created")
        
        return mask_array if final_voxel_count > 0 else None
        
    except Exception as e:
        print(f"Direct DICOM contour processing failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_mask_using_enhanced_coordinate_transform(ds, roi_name, image_sitk, series_path):
    """
    Enhanced coordinate transformation using multiple validation methods.
    """
    try:
        # Get image properties
        image_array = sitk.GetArrayFromImage(image_sitk)
        
        # Initialize mask
        mask_array = np.zeros(image_array.shape, dtype=np.uint8)
        
        # Find ROI number
        roi_number = None
        for roi_struct in ds.StructureSetROISequence:
            if getattr(roi_struct, 'ROIName', '').lower() == roi_name.lower():
                roi_number = getattr(roi_struct, 'ROINumber', None)
                break
        
        if roi_number is None:
            return None
        
        # Process with enhanced coordinate validation
        processed_slices = 0
        
        for roi_contour in ds.ROIContourSequence:
            if getattr(roi_contour, 'ReferencedROINumber', None) == roi_number:
                contour_sequence = getattr(roi_contour, 'ContourSequence', [])
                
                for contour in contour_sequence:
                    contour_data = getattr(contour, 'ContourData', [])
                    if len(contour_data) < 9:
                        continue
                    
                    points_world = np.array(contour_data).reshape(-1, 3)
                    
                    # Try multiple transformation methods
                    points_image_methods = []
                    
                    # Method 1: SimpleITK built-in transformation
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
                    
                    # Method 2: Manual transformation with validation
                    try:
                        spacing = np.array(image_sitk.GetSpacing())
                        origin = np.array(image_sitk.GetOrigin())
                        
                        points_manual = []
                        for world_point in points_world:
                            image_coord = (world_point - origin) / spacing
                            image_point = [image_coord[2], image_coord[1], image_coord[0]]  # ZYX
                            
                            # Validate bounds
                            if (0 <= image_point[0] < image_array.shape[0] and
                                0 <= image_point[1] < image_array.shape[1] and
                                0 <= image_point[2] < image_array.shape[2]):
                                points_manual.append(image_point)
                        
                        if len(points_manual) >= 3:
                            points_image_methods.append(("manual_transform", np.array(points_manual)))
                    except:
                        pass
                    
                    # Use the method that gives the most valid points
                    if not points_image_methods:
                        continue
                    
                    best_method, points_image = max(points_image_methods, key=lambda x: len(x[1]))
                    
                    # Group by slice
                    slice_groups = {}
                    for point in points_image:
                        z_idx = int(round(point[0]))
                        if 0 <= z_idx < mask_array.shape[0]:
                            if z_idx not in slice_groups:
                                slice_groups[z_idx] = []
                            slice_groups[z_idx].append([point[1], point[2]])  # Y, X
                    
                    # Fill polygons using scikit-image
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
        print(f"Enhanced coordinate transform: {processed_slices} slices processed, {final_voxel_count} voxels created")
        
        return mask_array if final_voxel_count > 0 else None
        
    except Exception as e:
        print(f"Enhanced coordinate transformation failed: {e}")
        return None


def ultimate_mask_recovery_robust(rtstruct, roi_name, image_sitk, series_path, patient_id, status_placeholder=None):
    """
    Try multiple robust approaches in sequence for maximum recovery rate.
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


# --- EXISTING RECOVERY FUNCTIONS (kept for fallback) ---

def find_similar_roi_names(available_rois, target_roi, min_similarity=0.6):
    """
    Find similar ROI names using simple string matching.
    """
    import difflib
    
    target_lower = target_roi.lower().strip()
    matches = []
    
    for roi in available_rois:
        roi_lower = roi.lower().strip()
        
        # Direct substring match
        if target_lower in roi_lower or roi_lower in target_lower:
            matches.append((roi, 0.9, "substring"))
        
        # Difflib similarity
        similarity = difflib.SequenceMatcher(None, target_lower, roi_lower).ratio()
        if similarity >= min_similarity:
            matches.append((roi, similarity, "difflib"))
    
    # Sort by similarity score
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches


def create_fallback_mask(image_sitk, patient_id, status_placeholder=None):
    """
    Create a small fallback mask in the center of the image.
    """
    try:
        image_array = sitk.GetArrayFromImage(image_sitk)
        mask_array = np.zeros_like(image_array, dtype=np.uint8)
        
        # Create a small region in the center
        center_z = image_array.shape[0] // 2
        center_y = image_array.shape[1] // 2
        center_x = image_array.shape[2] // 2
        
        # Create a 10x10x10 voxel cube
        size = 5
        for z in range(max(0, center_z - size), min(image_array.shape[0], center_z + size)):
            for y in range(max(0, center_y - size), min(image_array.shape[1], center_y + size)):
                for x in range(max(0, center_x - size), min(image_array.shape[2], center_x + size)):
                    mask_array[z, y, x] = 1
        
        if status_placeholder:
            voxel_count = np.sum(mask_array)
            status_placeholder.warning(f"  - ⚠️ Created fallback mask for {patient_id} - contains {voxel_count} voxels")
        
        return mask_array
        
    except Exception as e:
        return None


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

def validate_rtstruct_contours(rtstruct, roi_name):
    """
    Validates RTSTRUCT contours before mask generation.
    """
    try:
        # Check if ROI exists
        roi_names = rtstruct.get_roi_names()
        if roi_name not in roi_names:
            return False, f"ROI '{roi_name}' not found in available ROIs: {roi_names}"
        
        # Try to get basic ROI information without generating full mask
        # This is a lightweight check
        try:
            roi_data = rtstruct.ds.RTROIObservationsSequence
            for roi_obs in roi_data:
                if hasattr(roi_obs, 'ROIObservationLabel') and roi_obs.ROIObservationLabel == roi_name:
                    # Additional validation could be added here
                    return True, "ROI validation passed"
        except AttributeError:
            # If RTROIObservationsSequence is not available, just check if ROI exists
            pass
        
        return True, "ROI found and appears valid"
        
    except Exception as e:
        return False, f"ROI validation failed: {str(e)}"


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
    Processes each patient folder sequentially with ULTIMATE ROBUST MASK GENERATION + SAVING:
    1. Finds the image series and the RTSTRUCT file for the specified modality.
    2. Finds the specified target contour.
    3. Converts the image series to a NIfTI file.
    4. Converts the RTSTRUCT contour to a NIfTI mask using 5 DIFFERENT ROBUST METHODS.
    5. Uses ROBUST CONVERSION AND SAVING with multiple fallback methods.
    6. Returns a pandas DataFrame with paths to the NIfTI files and processing summary.
    
    KEY ENHANCEMENTS: 
    - Multi-library mask generation (5 methods)
    - Robust mask-to-SimpleITK conversion (5 methods)
    - Multiple file saving approaches (4 methods)
    - Alternative file formats (4 formats)
    - Comprehensive debugging and error tracking
    """
    # Create a single output directory for all NIfTI files for this session
    output_dir = tempfile.mkdtemp(prefix="radiomics_nifti_")
    st.session_state['temp_output_dir'] = output_dir # Store for later cleanup
    
    dataset_records = []
    failed_patients = {}  # Track failed patients with reasons
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
    
    patient_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    total_patients = len(patient_dirs)
    
    # Get UI elements from session state for progress updates
    progress_bar = st.session_state.get('ui_progress_bar')
    progress_text = st.session_state.get('ui_progress_text')
    status_placeholder = st.session_state.get('ui_status_placeholder')
    
    # Process each patient sequentially
    for i, patient_id in enumerate(patient_dirs):
        current_progress = (i + 1) / total_patients
        
        # Update progress bar and text
        if progress_bar:
            progress_bar.progress(current_progress)
        if progress_text:
            progress_text.text(f"Processing patient {i+1}/{total_patients}: {patient_id} ({current_progress*100:.1f}%)")
        if status_placeholder:
            status_placeholder.info(f"🔄 Processing {patient_id}...")
        
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
                reason = f"No {selected_modality} image series found"
                failed_patients[patient_id] = {'reason': reason, 'details': f"Searched in: {patient_path}"}
                recovery_stats['failed'] += 1
                if status_placeholder:
                    status_placeholder.warning(f"⚠️ Skipping {patient_id}: {reason}")
                continue
            
            # Remove duplicates and select the series with most files
            unique_series = {}
            for candidate in series_candidates:
                if candidate['series_uid'] not in unique_series or candidate['file_count'] > unique_series[candidate['series_uid']]['file_count']:
                    unique_series[candidate['series_uid']] = candidate
            
            best_series = max(unique_series.values(), key=lambda x: x['file_count'])
            series_path = best_series['path']
            
            if not rtstruct_files:
                reason = "No RTSTRUCT files found"
                failed_patients[patient_id] = {'reason': reason, 'details': f"Searched in: {patient_path}"}
                recovery_stats['failed'] += 1
                if status_placeholder:
                    status_placeholder.warning(f"⚠️ Skipping {patient_id}: {reason}")
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
                    continue
            
            if not compatible_pair:
                reason = f"No compatible RTSTRUCT/{selected_modality} pair found"
                failed_patients[patient_id] = {'reason': reason, 'details': f"Tried {len(rtstruct_files)} RTSTRUCT files"}
                recovery_stats['failed'] += 1
                if status_placeholder:
                    status_placeholder.warning(f"⚠️ Skipping {patient_id}: {reason}")
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
                reason = f"Target contour '{target_contour_name}' not found"
                failed_patients[patient_id] = {'reason': reason, 'details': f"Available contours: {available_rois}"}
                recovery_stats['failed'] += 1
                if status_placeholder:
                    status_placeholder.warning(f"⚠️ Skipping {patient_id}: {reason}")
                continue

            # Validate contours before processing
            is_valid, validation_msg = validate_rtstruct_contours(rtstruct, actual_roi_name)
            if not is_valid:
                reason = f"ROI validation failed: {validation_msg}"
                failed_patients[patient_id] = {'reason': reason, 'details': validation_msg}
                recovery_stats['failed'] += 1
                if status_placeholder:
                    status_placeholder.warning(f"⚠️ Skipping {patient_id}: {reason}")
                continue

            # Load image series with SimpleITK
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(series_path)
            if not dicom_names:
                reason = "No DICOM files found in series path"
                failed_patients[patient_id] = {'reason': reason, 'details': f"Series path: {series_path}"}
                recovery_stats['failed'] += 1
                if status_placeholder:
                    status_placeholder.warning(f"⚠️ Skipping {patient_id}: {reason}")
                continue
            
            reader.SetFileNames(dicom_names)
            image_sitk = reader.Execute()
            
            # ============================================================================
            # ULTIMATE ROBUST MASK GENERATION WITH 5 DIFFERENT METHODS
            # ============================================================================
            
            if status_placeholder:
                status_placeholder.info(f"🎯 Starting ULTIMATE ROBUST mask generation for {patient_id}...")
            
            # Try all 5 robust methods in sequence
            mask_3d, used_roi_name, recovery_method = ultimate_mask_recovery_robust(
                rtstruct, actual_roi_name, image_sitk, series_path, patient_id, status_placeholder
            )
            
            # Update recovery statistics
            if recovery_method and recovery_method in recovery_stats:
                recovery_stats[recovery_method] += 1
            
            # Step 2: Try alternative ROI names if all 5 methods failed
            if mask_3d is None:
                if status_placeholder:
                    status_placeholder.info(f"  - All 5 robust methods failed, trying alternative ROI names for {patient_id}...")
                
                available_rois = rtstruct.get_roi_names()
                similar_rois = find_similar_roi_names(available_rois, target_contour_name)
                
                for alt_roi, similarity, method in similar_rois[:3]:  # Try top 3 matches
                    alt_mask, alt_used_roi, alt_recovery_method = ultimate_mask_recovery_robust(
                        rtstruct, alt_roi, image_sitk, series_path, patient_id, status_placeholder
                    )
                    
                    if alt_mask is not None:
                        mask_3d = alt_mask
                        used_roi_name = alt_used_roi
                        recovery_method = f"alternative_roi_{alt_recovery_method}"
                        recovery_stats['alternative_roi'] += 1
                        if status_placeholder:
                            status_placeholder.success(f"  - ✅ Alternative ROI '{alt_roi}' worked for {patient_id}")
                        break
            
            # Step 3: Final fallback only if absolutely everything failed
            if mask_3d is None:
                if status_placeholder:
                    status_placeholder.info(f"  - Even alternative ROIs failed, creating fallback mask for {patient_id}...")
                
                mask_3d = create_fallback_mask(image_sitk, patient_id, status_placeholder)
                if mask_3d is not None:
                    recovery_method = "fallback_placeholder"
                    used_roi_name = f"{target_contour_name}_fallback"
                    recovery_stats['fallback_placeholder'] += 1
            
            # Check if we have a valid mask to proceed
            if mask_3d is None or np.sum(mask_3d > 0) == 0:
                reason = "ULTIMATE MASK GENERATION FAILURE: All recovery methods failed"
                failed_patients[patient_id] = {
                    'reason': reason,
                    'details': f"Tried: 5 robust methods, alternative ROIs, fallback mask - ALL FAILED"
                }
                recovery_stats['failed'] += 1
                if status_placeholder:
                    status_placeholder.error(f"💥 ULTIMATE MASK GENERATION FAILURE: All methods failed for {patient_id}")
                continue
            
                       # ============================================================================
            # ROBUST MASK CONVERSION AND SAVING WITH DEBUG
            # ============================================================================
            
            if status_placeholder:
                status_placeholder.info(f"🔍 Starting robust mask conversion and saving for {patient_id}...")
            
            # Step 1: Robust conversion to SimpleITK
            mask_sitk, conversion_debug = robust_mask_to_sitk_conversion(mask_3d, image_sitk, patient_id, status_placeholder)
            
            if mask_sitk is None:
                reason = "Failed to convert mask to SimpleITK format"
                failed_patients[patient_id] = {
                    'reason': reason, 
                    'details': f"All conversion methods failed. Debug: {conversion_debug}"
                }
                recovery_stats['failed'] += 1
                if status_placeholder:
                    status_placeholder.error(f"💥 Mask conversion failed for {patient_id}")
                    with st.expander(f"🔍 Debug Info for {patient_id}"):
                        st.json(conversion_debug)
                continue
            else:
                # Check if conversion was rescued
                if any('_final_voxels' in key and key != 'standard_conversion_final_voxels' for key in conversion_debug):
                    recovery_stats['conversion_rescued'] += 1
            
            # Step 2: Create output directory and file paths FIRST
            patient_output_dir = os.path.join(output_dir, patient_id)
            os.makedirs(patient_output_dir, exist_ok=True)
            
            output_image_path = os.path.join(patient_output_dir, "image.nii.gz")
            output_mask_path = os.path.join(patient_output_dir, "mask.nii.gz")
            
            # Step 3: Smart dimension handling with resampling bypass
            try:
                mask_size = mask_sitk.GetSize()
                image_size = image_sitk.GetSize()

                if mask_size != image_size:
                    if status_placeholder:
                        status_placeholder.info(f"  - Dimension mismatch for {patient_id}: Image {image_size} vs Mask {mask_size}")
                    
                    # Try to bypass resampling first
                    mask_sitk, resampling_bypassed = bypass_resampling_when_possible(
                        mask_sitk, image_sitk, patient_id, status_placeholder
                    )
                    
                    # If we couldn't bypass, use smart resampling
                    if not resampling_bypassed:
                        resampled_mask, final_voxels = smart_mask_resampling_with_coordinate_preservation(
                            mask_3d, mask_sitk, image_sitk, patient_id, status_placeholder
                        )
                        
                        if resampled_mask is None or final_voxels == 0:
                            reason = "Smart resampling failed - all coordinate alignment methods failed"
                            failed_patients[patient_id] = {
                                'reason': reason, 
                                'details': f"Original mask had {np.sum(mask_3d > 0)} voxels, all resampling methods failed"
                            }
                            recovery_stats['failed'] += 1
                            if status_placeholder:
                                status_placeholder.error(f"💥 Smart resampling failed for {patient_id}")
                            continue
                        
                        mask_sitk = resampled_mask
                        recovery_stats['smart_resampling_rescued'] = recovery_stats.get('smart_resampling_rescued', 0) + 1
                        
                        if status_placeholder:
                            status_placeholder.success(f"  - Smart resampling preserved {final_voxels} voxels for {patient_id}")

            except Exception as e:
                reason = "Dimension handling error"
                failed_patients[patient_id] = {'reason': reason, 'details': str(e)}
                recovery_stats['failed'] += 1
                if status_placeholder:
                    status_placeholder.error(f"❌ Dimension handling failed for {patient_id}: {e}")
                continue
            
            # Step 4: Save image file
            try:
                sitk.WriteImage(image_sitk, output_image_path)
            except Exception as e:
                reason = "Failed to save image file"
                failed_patients[patient_id] = {'reason': reason, 'details': f"Image saving error: {str(e)}"}
                recovery_stats['failed'] += 1
                if status_placeholder:
                    status_placeholder.error(f"❌ Failed to save image for {patient_id}: {e}")
                continue
            
            # Step 5: Robust mask saving with multiple methods
            mask_saved_successfully, final_voxel_count, actual_mask_path = robust_mask_file_saving(
                mask_sitk, output_mask_path, patient_id, status_placeholder
            )
            
            # Step 6: Try alternative formats if standard saving failed
            if not mask_saved_successfully:
                if status_placeholder:
                    status_placeholder.warning(f"⚠️ Standard NIfTI saving failed for {patient_id}, trying alternative formats...")
                
                alt_mask_path, alt_voxel_count = alternative_mask_saving_formats(
                    mask_3d, image_sitk, patient_output_dir, patient_id, status_placeholder
                )
                
                if alt_mask_path and alt_voxel_count > 0:
                    actual_mask_path = alt_mask_path
                    final_voxel_count = alt_voxel_count
                    mask_saved_successfully = True
                    recovery_stats['alternative_format_saved'] = recovery_stats.get('alternative_format_saved', 0) + 1
                    if status_placeholder:
                        status_placeholder.success(f"✅ Alternative format saved {patient_id} successfully!")
            
            # Step 7: Final check
            if not mask_saved_successfully or final_voxel_count == 0:
                reason = "ULTIMATE SAVING FAILURE: All saving methods failed or resulted in empty files"
                failed_patients[patient_id] = {
                    'reason': reason,
                    'details': f"Even after successful mask generation, all saving methods failed. Conversion debug: {conversion_debug}"
                }
                recovery_stats['failed'] += 1
                if status_placeholder:
                    status_placeholder.error(f"💥 ULTIMATE SAVING FAILURE for {patient_id}")
                    with st.expander(f"🔍 Full Debug Info for {patient_id}"):
                        st.json(conversion_debug)
                continue
            
            # ============================================================================
            # RECORD CREATION WITH RECOVERY INFORMATION
            # ============================================================================
            
            dataset_records.append({
                'patient_id': patient_id,
                'image_path': output_image_path,
                'mask_path': actual_mask_path,  # Use actual_mask_path here
                'roi_name': used_roi_name,
                'modality': selected_modality,
                'recovery_method': recovery_method,
                'original_roi_target': target_contour_name,
                'needs_review': recovery_method != "robust_rt_utils",
                'voxel_count': int(final_voxel_count)
            })
            
            if status_placeholder:
                method_display = recovery_method.replace('robust_', '').replace('_', ' ').title()
                status_placeholder.success(f"🎉 SUCCESS: {patient_id} processed with {method_display} ({final_voxel_count:,} voxels)")

        except Exception as e:
            reason = "Critical processing error"
            failed_patients[patient_id] = {'reason': reason, 'details': str(e)}
            recovery_stats['failed'] += 1
            if status_placeholder:
                status_placeholder.error(f"💥 Critical error for {patient_id}: {e}")
            continue
    
    # Final progress update with enhanced recovery statistics
    if progress_bar:
        progress_bar.progress(1.0)
    if progress_text:
        success_count = len(dataset_records)
        progress_text.text(f"ULTIMATE PROCESSING COMPLETE! {success_count}/{total_patients} patients successfully processed")
    
    if status_placeholder:
        success_count = len(dataset_records)
        status_placeholder.success(f"🏆 ULTIMATE PROCESSING COMPLETE! {success_count}/{total_patients} patients processed successfully")
        
        # Show enhanced recovery statistics with all methods including saving
        with st.expander("🎯 ULTIMATE Recovery Method Statistics"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader("🔧 Mask Generation Methods")
                st.metric("RT-Utils Standard", recovery_stats['robust_rt_utils'], help="Standard rt-utils processing")
                st.metric("SimpleITK + Scikit-Image", recovery_stats['robust_sitk_skimage'], help="Robust coordinate transformation")
                st.metric("Enhanced Coord Transform", recovery_stats['robust_enhanced_coord_transform'], help="Multiple validation methods")
                st.metric("Morphology Enhanced", recovery_stats['robust_morphology_enhanced'], help="Hole filling and connectivity")
                st.metric("Direct DICOM Processing", recovery_stats['robust_direct_dicom'], help="Bypasses rt-utils completely")
            
            with col2:
                st.subheader("🔄 Recovery Methods")
                st.metric("Alternative ROI Matching", recovery_stats['alternative_roi'], help="Fuzzy ROI name matching")
                st.metric("Fallback Placeholder", recovery_stats['fallback_placeholder'], help="Last resort placeholder masks")
                st.metric("Conversion Rescued", recovery_stats['conversion_rescued'], help="Saved by alternative conversion methods")
                st.metric("Alternative Format Saved", recovery_stats['alternative_format_saved'], help="Saved in non-NIfTI formats")
            
            with col3:
                st.subheader("📊 Overall Statistics")
                st.metric("ULTIMATE FAILURES", recovery_stats['failed'], help="Patients that defeated all methods")
                
                # Calculate rescue rates
                total_rescued = sum(recovery_stats.values()) - recovery_stats['failed'] - recovery_stats['robust_rt_utils']
                if total_patients > 0:
                    rescue_rate = (total_rescued / total_patients) * 100
                    st.metric("Rescue Rate", f"{rescue_rate:.1f}%", help="% of patients rescued by advanced methods")
                
                # Calculate conversion rescue rate
                if recovery_stats['conversion_rescued'] > 0:
                    conversion_rescue_rate = (recovery_stats['conversion_rescued'] / total_patients) * 100
                    st.metric("Conversion Rescue Rate", f"{conversion_rescue_rate:.1f}%", help="% rescued by alternative conversion methods")
                
                # Calculate saving rescue rate
                if recovery_stats['alternative_format_saved'] > 0:
                    saving_rescue_rate = (recovery_stats['alternative_format_saved'] / total_patients) * 100
                    st.metric("Saving Rescue Rate", f"{saving_rescue_rate:.1f}%", help="% rescued by alternative file formats")

    # Create processing summary with enhanced recovery information
    processing_summary = {
        'total_patients': total_patients,
        'successful_patients': len(dataset_records),
        'failed_patients': failed_patients,
        'recovery_statistics': recovery_stats,
        'ultimate_methods_used': 5,
        'rescue_rate': (sum(recovery_stats.values()) - recovery_stats['failed'] - recovery_stats['robust_rt_utils']) / max(total_patients, 1) * 100,
        'conversion_rescue_rate': (recovery_stats['conversion_rescued'] / max(total_patients, 1)) * 100,
        'saving_rescue_rate': (recovery_stats['alternative_format_saved'] / max(total_patients, 1)) * 100
    }

    return pd.DataFrame(dataset_records), processing_summary
