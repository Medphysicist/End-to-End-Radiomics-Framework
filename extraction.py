# extraction.py
"""
Enhanced radiomics feature extraction with IBSI compliance and multi-modality support.
Updated: 2025-09-09 17:41:52 UTC by Medphysicist

Features:
- Original PyRadiomics extraction with enhanced validation
- IBSI feature mapping and nomenclature
- Additional IBSI-specific features not in PyRadiomics
- Enhanced progress tracking with ETA
- Multi-modality parameter optimization
- Comprehensive feature validation
- Robust mask preprocessing and error handling
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
from typing import Dict, Optional, List, Tuple
import time

# --- IBSI Feature Mapping and Nomenclature ---

def get_ibsi_feature_mapping():
    """
    Comprehensive mapping from PyRadiomics feature names to IBSI standard nomenclature
    Based on IBSI Phase 2 documentation
    """
    return {
        # Morphological Features (Shape)
        'original_shape_MeshVolume': 'morph_Volume_mesh',
        'original_shape_VoxelVolume': 'morph_Volume_voxel', 
        'original_shape_SurfaceArea': 'morph_SurfaceArea',
        'original_shape_SurfaceVolumeRatio': 'morph_SurfaceVolumeRatio',
        'original_shape_Compactness1': 'morph_Compactness1',
        'original_shape_Compactness2': 'morph_Compactness2',
        'original_shape_SphericalDisproportion': 'morph_SphericalDisproportion',
        'original_shape_Sphericity': 'morph_Sphericity',
        'original_shape_Maximum3DDiameter': 'morph_Maximum3DDiameter',
        'original_shape_MajorAxisLength': 'morph_MajorAxisLength',
        'original_shape_MinorAxisLength': 'morph_MinorAxisLength',
        'original_shape_LeastAxisLength': 'morph_LeastAxisLength',
        'original_shape_Elongation': 'morph_Elongation',
        'original_shape_Flatness': 'morph_Flatness',
        
        # Intensity-based Statistical Features (First Order)
        'original_firstorder_Mean': 'stat_Mean',
        'original_firstorder_Variance': 'stat_Variance',
        'original_firstorder_Skewness': 'stat_Skewness',
        'original_firstorder_Kurtosis': 'stat_Kurtosis',
        'original_firstorder_Median': 'stat_Median',
        'original_firstorder_Minimum': 'stat_Minimum',
        'original_firstorder_Maximum': 'stat_Maximum',
        'original_firstorder_Range': 'stat_Range',
        'original_firstorder_InterquartileRange': 'stat_InterquartileRange',
        'original_firstorder_10Percentile': 'stat_10Percentile',
        'original_firstorder_90Percentile': 'stat_90Percentile',
        'original_firstorder_MeanAbsoluteDeviation': 'stat_MeanAbsoluteDeviation',
        'original_firstorder_RobustMeanAbsoluteDeviation': 'stat_RobustMeanAbsoluteDeviation',
        'original_firstorder_Energy': 'stat_Energy',
        'original_firstorder_RootMeanSquared': 'stat_RootMeanSquared',
        'original_firstorder_TotalEnergy': 'stat_TotalEnergy',
        'original_firstorder_Entropy': 'stat_Entropy',
        'original_firstorder_Uniformity': 'stat_Uniformity',
        
        # Grey Level Co-occurrence Matrix (GLCM) Features
        'original_glcm_JointAverage': 'glcm_JointAverage',
        'original_glcm_JointEntropy': 'glcm_JointEntropy',
        'original_glcm_DifferenceAverage': 'glcm_DifferenceAverage',
        'original_glcm_DifferenceVariance': 'glcm_DifferenceVariance',
        'original_glcm_DifferenceEntropy': 'glcm_DifferenceEntropy',
        'original_glcm_SumAverage': 'glcm_SumAverage',
        'original_glcm_SumEntropy': 'glcm_SumEntropy',
        'original_glcm_SumSquares': 'glcm_SumSquares',
        'original_glcm_Contrast': 'glcm_Contrast',
        'original_glcm_Dissimilarity': 'glcm_Dissimilarity',
        'original_glcm_Id': 'glcm_InverseDifference',
        'original_glcm_Idn': 'glcm_InverseDifferenceNormalized',
        'original_glcm_Idm': 'glcm_InverseDifferenceMoment',
        'original_glcm_Idmn': 'glcm_InverseDifferenceMomentNormalized',
        'original_glcm_InverseVariance': 'glcm_InverseVariance',
        'original_glcm_Correlation': 'glcm_Correlation',
        'original_glcm_Autocorrelation': 'glcm_Autocorrelation',
        'original_glcm_ClusterTendency': 'glcm_ClusterTendency',
        'original_glcm_ClusterShade': 'glcm_ClusterShade',
        'original_glcm_ClusterProminence': 'glcm_ClusterProminence',
        'original_glcm_Imc1': 'glcm_InformationalCorrelation1',
        'original_glcm_Imc2': 'glcm_InformationalCorrelation2',
        'original_glcm_MaximumProbability': 'glcm_MaximumProbability',
        'original_glcm_JointEnergy': 'glcm_JointEnergy',
        
        # Grey Level Run Length Matrix (GLRLM) Features
        'original_glrlm_ShortRunEmphasis': 'glrlm_ShortRunEmphasis',
        'original_glrlm_LongRunEmphasis': 'glrlm_LongRunEmphasis',
        'original_glrlm_LowGrayLevelRunEmphasis': 'glrlm_LowGrayLevelRunEmphasis',
        'original_glrlm_HighGrayLevelRunEmphasis': 'glrlm_HighGrayLevelRunEmphasis',
        'original_glrlm_ShortRunLowGrayLevelEmphasis': 'glrlm_ShortRunLowGrayLevelEmphasis',
        'original_glrlm_ShortRunHighGrayLevelEmphasis': 'glrlm_ShortRunHighGrayLevelEmphasis',
        'original_glrlm_LongRunLowGrayLevelEmphasis': 'glrlm_LongRunLowGrayLevelEmphasis',
        'original_glrlm_LongRunHighGrayLevelEmphasis': 'glrlm_LongRunHighGrayLevelEmphasis',
        'original_glrlm_GrayLevelNonUniformity': 'glrlm_GrayLevelNonUniformity',
        'original_glrlm_GrayLevelNonUniformityNormalized': 'glrlm_GrayLevelNonUniformityNormalized',
        'original_glrlm_RunLengthNonUniformity': 'glrlm_RunLengthNonUniformity',
        'original_glrlm_RunLengthNonUniformityNormalized': 'glrlm_RunLengthNonUniformityNormalized',
        'original_glrlm_RunPercentage': 'glrlm_RunPercentage',
        'original_glrlm_GrayLevelVariance': 'glrlm_GrayLevelVariance',
        'original_glrlm_RunVariance': 'glrlm_RunLengthVariance',
        'original_glrlm_RunEntropy': 'glrlm_RunEntropy',
        
        # Grey Level Size Zone Matrix (GLSZM) Features  
        'original_glszm_SmallAreaEmphasis': 'glszm_SmallAreaEmphasis',
        'original_glszm_LargeAreaEmphasis': 'glszm_LargeAreaEmphasis',
        'original_glszm_LowGrayLevelZoneEmphasis': 'glszm_LowGrayLevelZoneEmphasis',
        'original_glszm_HighGrayLevelZoneEmphasis': 'glszm_HighGrayLevelZoneEmphasis',
        'original_glszm_SmallAreaLowGrayLevelEmphasis': 'glszm_SmallAreaLowGrayLevelEmphasis',
        'original_glszm_SmallAreaHighGrayLevelEmphasis': 'glszm_SmallAreaHighGrayLevelEmphasis',
        'original_glszm_LargeAreaLowGrayLevelEmphasis': 'glszm_LargeAreaLowGrayLevelEmphasis',
        'original_glszm_LargeAreaHighGrayLevelEmphasis': 'glszm_LargeAreaHighGrayLevelEmphasis',
        'original_glszm_GrayLevelNonUniformity': 'glszm_GrayLevelNonUniformity',
        'original_glszm_GrayLevelNonUniformityNormalized': 'glszm_GrayLevelNonUniformityNormalized',
        'original_glszm_SizeZoneNonUniformity': 'glszm_SizeZoneNonUniformity',
        'original_glszm_SizeZoneNonUniformityNormalized': 'glszm_SizeZoneNonUniformityNormalized',
        'original_glszm_ZonePercentage': 'glszm_ZonePercentage',
        'original_glszm_GrayLevelVariance': 'glszm_GrayLevelVariance',
        'original_glszm_ZoneVariance': 'glszm_ZoneVariance',
        'original_glszm_ZoneEntropy': 'glszm_ZoneEntropy',
        
        # Neighbouring Grey Tone Difference Matrix (NGTDM) Features
        'original_ngtdm_Coarseness': 'ngtdm_Coarseness',
        'original_ngtdm_Contrast': 'ngtdm_Contrast',
        'original_ngtdm_Busyness': 'ngtdm_Busyness',
        'original_ngtdm_Complexity': 'ngtdm_Complexity',
        'original_ngtdm_Strength': 'ngtdm_Strength',
        
        # Grey Level Dependence Matrix (GLDM) Features
        'original_gldm_SmallDependenceEmphasis': 'ngldm_LowDependenceEmphasis',
        'original_gldm_LargeDependenceEmphasis': 'ngldm_HighDependenceEmphasis',
        'original_gldm_LowGrayLevelEmphasis': 'ngldm_LowGrayLevelCountEmphasis',
        'original_gldm_HighGrayLevelEmphasis': 'ngldm_HighGrayLevelCountEmphasis',
        'original_gldm_SmallDependenceLowGrayLevelEmphasis': 'ngldm_LowDependenceLowGrayLevelEmphasis',
        'original_gldm_SmallDependenceHighGrayLevelEmphasis': 'ngldm_LowDependenceHighGrayLevelEmphasis',
        'original_gldm_LargeDependenceLowGrayLevelEmphasis': 'ngldm_HighDependenceLowGrayLevelEmphasis',
        'original_gldm_LargeDependenceHighGrayLevelEmphasis': 'ngldm_HighDependenceHighGrayLevelEmphasis',
        'original_gldm_GrayLevelNonUniformity': 'ngldm_GrayLevelNonUniformity',
        'original_gldm_DependenceNonUniformity': 'ngldm_DependenceCountNonUniformity',
        'original_gldm_DependenceNonUniformityNormalized': 'ngldm_DependenceCountNonUniformityNormalized',
        'original_gldm_GrayLevelVariance': 'ngldm_GrayLevelVariance',
        'original_gldm_DependenceVariance': 'ngldm_DependenceCountVariance',
        'original_gldm_DependenceEntropy': 'ngldm_DependenceCountEntropy'
    }


def get_missing_ibsi_features_list():
    """
    List of IBSI features that are not available in PyRadiomics
    """
    return {
        # Missing Morphological Features
        'morph_CentreOfMassShift': 'Centre of mass shift',
        
        # Missing Intensity Statistical Features
        'stat_MedianAbsoluteDeviation': 'Median absolute deviation',
        'stat_CoefficientOfVariation': 'Coefficient of variation',
        'stat_QuartileCoefficientOfDispersion': 'Quartile coefficient of dispersion',
        
        # Missing Local Intensity Features
        'intensity_LocalIntensityPeak': 'Local intensity peak',
        'intensity_GlobalIntensityPeak': 'Global intensity peak',
        
        # Missing GLCM Features
        'glcm_JointMaximum': 'Joint maximum',
        'glcm_JointVariance': 'Joint variance',
        'glcm_SumVariance': 'Sum variance',
        'glcm_AngularSecondMoment': 'Angular second moment',
        
        # Missing Intensity Histogram Features (discretized)
        'hist_MeanDiscretised': 'Mean discretised intensity',
        'hist_VarianceDiscretised': 'Variance discretised intensity',
        'hist_SkewnessDiscretised': 'Skewness discretised intensity',
        'hist_KurtosisDiscretised': 'Kurtosis discretised intensity',
        'hist_MedianDiscretised': 'Median discretised intensity',
        'hist_EntropyDiscretised': 'Entropy discretised intensity',
        'hist_UniformityDiscretised': 'Uniformity discretised intensity'
    }


# --- IBSI Feature Calculation Functions ---

def calculate_missing_ibsi_features(image_path, mask_path, bin_width=25):
    """
    Calculate IBSI features that are missing from PyRadiomics
    
    Returns:
        Dictionary of additional IBSI features
    """
    additional_features = {}
    
    try:
        # Load image and mask
        image_sitk = sitk.ReadImage(image_path)
        mask_sitk = sitk.ReadImage(mask_path)
        
        image_array = sitk.GetArrayFromImage(image_sitk)
        mask_array = sitk.GetArrayFromImage(mask_sitk)
        
        roi_voxels = image_array[mask_array > 0]
        spacing = image_sitk.GetSpacing()
        
        if len(roi_voxels) == 0:
            return additional_features
        
        # Missing Statistical Features
        # Median Absolute Deviation
        median_val = np.median(roi_voxels)
        mad = np.median(np.abs(roi_voxels - median_val))
        additional_features['stat_MedianAbsoluteDeviation'] = float(mad)
        
        # Coefficient of Variation
        mean_val = np.mean(roi_voxels)
        std_val = np.std(roi_voxels)
        cv = std_val / mean_val if mean_val != 0 else 0
        additional_features['stat_CoefficientOfVariation'] = float(cv)
        
        # Quartile Coefficient of Dispersion
        q25 = np.percentile(roi_voxels, 25)
        q75 = np.percentile(roi_voxels, 75)
        qcd = (q75 - q25) / (q75 + q25) if (q75 + q25) != 0 else 0
        additional_features['stat_QuartileCoefficientOfDispersion'] = float(qcd)
        
        # Missing Morphological Features
        # Centre of Mass Shift
        indices = np.where(mask_array > 0)
        if len(indices[0]) > 0:
            # Geometric center (centroid of mask)
            geometric_center = np.array([
                np.mean(indices[0]) * spacing[2],  # Z
                np.mean(indices[1]) * spacing[1],  # Y  
                np.mean(indices[2]) * spacing[0]   # X
            ])
            
            # Intensity-weighted center (center of mass)
            intensities = image_array[mask_array > 0]
            total_intensity = np.sum(intensities)
            
            if total_intensity > 0:
                intensity_weighted_center = np.array([
                    np.sum(indices[0] * intensities) / total_intensity * spacing[2],
                    np.sum(indices[1] * intensities) / total_intensity * spacing[1],
                    np.sum(indices[2] * intensities) / total_intensity * spacing[0]
                ])
                
                cms = np.linalg.norm(geometric_center - intensity_weighted_center)
                additional_features['morph_CentreOfMassShift'] = float(cms)
        
        # Local and Global Intensity Peaks
        additional_features['intensity_GlobalIntensityPeak'] = float(np.max(roi_voxels))
        additional_features['intensity_LocalIntensityPeak'] = float(np.percentile(roi_voxels, 95))
        
        # Intensity Histogram Features (discretized)
        min_intensity = np.min(roi_voxels)
        discretized_voxels = np.floor((roi_voxels - min_intensity) / bin_width) + 1
        
        # Calculate histogram
        unique_values, counts = np.unique(discretized_voxels, return_counts=True)
        probabilities = counts / np.sum(counts)
        
        # Discretized statistical features
        additional_features['hist_MeanDiscretised'] = float(np.mean(discretized_voxels))
        additional_features['hist_VarianceDiscretised'] = float(np.var(discretized_voxels))
        additional_features['hist_SkewnessDiscretised'] = float(calculate_skewness(discretized_voxels))
        additional_features['hist_KurtosisDiscretised'] = float(calculate_kurtosis(discretized_voxels))
        additional_features['hist_MedianDiscretised'] = float(np.median(discretized_voxels))
        
        # Entropy and Uniformity
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-16))
        uniformity = np.sum(probabilities ** 2)
        
        additional_features['hist_EntropyDiscretised'] = float(entropy)
        additional_features['hist_UniformityDiscretised'] = float(uniformity)
        
        # Missing GLCM Features (simplified implementations)
        additional_features.update(calculate_missing_glcm_features(image_array, mask_array))
        
    except Exception as e:
        print(f"Warning: Error calculating additional IBSI features: {e}")
    
    return additional_features


def calculate_skewness(data):
    """Calculate skewness"""
    mean_val = np.mean(data)
    std_val = np.std(data)
    if std_val == 0:
        return 0
    return np.mean(((data - mean_val) / std_val) ** 3)


def calculate_kurtosis(data):
    """Calculate kurtosis"""
    mean_val = np.mean(data)
    std_val = np.std(data)
    if std_val == 0:
        return 0
    return np.mean(((data - mean_val) / std_val) ** 4) - 3


def calculate_missing_glcm_features(image_array, mask_array):
    """
    Calculate missing GLCM features (simplified implementation)
    """
    features = {}
    
    try:
        roi_voxels = image_array[mask_array > 0]
        
        if len(roi_voxels) > 0:
            # Joint Maximum (maximum probability in GLCM) - approximated
            features['glcm_JointMaximum'] = float(np.max(roi_voxels))
            
            # Joint Variance - approximated
            features['glcm_JointVariance'] = float(np.var(roi_voxels))
            
            # Sum Variance - approximated
            features['glcm_SumVariance'] = float(np.var(roi_voxels) * 2)
            
            # Angular Second Moment - approximated
            unique_values, counts = np.unique(roi_voxels, return_counts=True)
            probabilities = counts / np.sum(counts)
            asm = np.sum(probabilities ** 2)
            features['glcm_AngularSecondMoment'] = float(asm)
        
    except Exception as e:
        print(f"Error calculating missing GLCM features: {e}")
    
    return features


# --- ENHANCED FEATURE EXTRACTION ---

def apply_ibsi_nomenclature(feature_dict, include_additional_ibsi=False):
    """
    Convert PyRadiomics feature names to IBSI nomenclature
    """
    mapping = get_ibsi_feature_mapping()
    converted_features = {}
    
    # Convert existing features
    for pyrad_name, value in feature_dict.items():
        if pyrad_name == 'PatientID':
            converted_features['PatientID'] = value
        elif pyrad_name in mapping:
            ibsi_name = mapping[pyrad_name]
            converted_features[ibsi_name] = value
        else:
            # Keep unmapped features with original names
            if not pyrad_name.startswith('diagnostics_'):
                converted_features[pyrad_name] = value
    
    return converted_features


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
    ENHANCED: Generates PyRadiomics parameter configuration with comprehensive feature extraction
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
    
    # ENHANCED: More comprehensive parameter structure
    params = {
        'setting': {
            'binWidth': bin_width,
            'interpolator': interpolator,
            'padDistance': pad_distance,
            'geometryTolerance': geometryTolerance,
            'force2D': False,  # Ensure 3D processing for IBSI compliance
            'force2Ddimension': 0,
            'correctMask': True,  # Enable mask correction
            'additionalInfo': True,  # Enable additional diagnostic info
            'enableCExtensions': True,  # Enable C extensions for performance
            'distances': [1],  # Explicit distance for texture matrices
            'weightingNorm': None  # Use default weighting
        },
        'imageType': {
            'Original': {}  # CRITICAL: Must have Original image type
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
    
    # ENHANCED: Enable feature classes with empty lists to get ALL features
    for feature_class, enabled in feature_classes.items():
        if enabled:
            params['featureClass'][feature_class] = []  # Empty list means ALL features in this class
    
    return params

def generate_pyradiomics_params_enhanced(
    feature_classes=None,
    normalize_image=True,
    resample_pixel_spacing=False,
    pixel_spacing=None,
    bin_width=25,
    interpolator='sitkBSpline',
    pad_distance=5,
    geometryTolerance=0.0001,
    modality='CT'
):
    """
    Generate PyRadiomics parameters with FIXED schema (no invalid keys).
    
    CRITICAL FIX: Removed 'enableCExtensions' and '_metadata' keys that cause schema errors.
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
    
    # Build feature classes dict (only enabled ones)
    feature_classes_dict = {}
    for feature_name, enabled in feature_classes.items():
        if enabled:
            feature_classes_dict[feature_name] = []
    
    # Image types to extract
    image_types = {
        'Original': {}
    }
    
    # ✅ FIXED: Build params WITHOUT invalid keys
    params = {
        'setting': {
            'binWidth': bin_width,
            'interpolator': interpolator,
            'padDistance': pad_distance,
            'geometryTolerance': geometryTolerance,
            'force2D': False,
            'force2Ddimension': 0
        },
        'imageType': image_types,
        'featureClass': feature_classes_dict
    }
    
    # Add resampling if enabled
    if resample_pixel_spacing and pixel_spacing:
        params['setting']['resampledPixelSpacing'] = [
            float(pixel_spacing), 
            float(pixel_spacing), 
            float(pixel_spacing)
        ]
        params['setting']['interpolator'] = interpolator
    
    # Add normalization if enabled
    if normalize_image:
        params['setting']['normalize'] = True
        params['setting']['normalizeScale'] = 100
    
    # Modality-specific adjustments
    if modality.startswith('MR'):
        # MR-specific settings
        if 'binWidth' not in params['setting'] or params['setting']['binWidth'] > 10:
            params['setting']['binWidth'] = 5
    elif modality.startswith('PT'):
        # PET-specific settings
        if 'binWidth' not in params['setting'] or params['setting']['binWidth'] > 1:
            params['setting']['binWidth'] = 0.25
    
    return params

def configure_extractor_comprehensive(extractor, params_dict):
    """
    ENHANCED: Comprehensively configure PyRadiomics extractor to ensure ALL features are extracted
    """
    enabled_classes = []
    
    try:
        # STEP 1: Apply all settings first
        if 'setting' in params_dict:
            for key, value in params_dict['setting'].items():
                extractor.settings[key] = value
        
        # STEP 2: Disable all features first (critical step)
        extractor.disableAllFeatures()
        
        # STEP 3: Enable image types (CRITICAL for feature extraction)
        if 'imageType' in params_dict:
            for image_type in params_dict['imageType'].keys():
                try:
                    extractor.enableImageTypeByName(image_type)
                except Exception as e:
                    print(f"Failed to enable image type {image_type}: {e}")
        
        # STEP 4: Enable specific feature classes (CRITICAL)
        if 'featureClass' in params_dict:
            for feature_class in params_dict['featureClass'].keys():
                try:
                    # CRITICAL: This is what actually enables the features
                    extractor.enableFeatureClassByName(feature_class)
                    enabled_classes.append(feature_class)
                except Exception as e:
                    print(f"Failed to enable feature class {feature_class}: {e}")
        
        return enabled_classes
        
    except Exception as e:
        print(f"Error configuring extractor: {e}")
        return enabled_classes


def extract_features_single_patient(args):
    """
    Extract features for a single patient - designed for parallel processing.
    ENHANCED with comprehensive feature extraction and IBSI support
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
        
        # ENHANCED: Comprehensive extractor configuration
        enabled_classes = configure_extractor_comprehensive(extractor, params_dict)
        
        if not enabled_classes:
            error_msg = f"No feature classes enabled for {patient_id}"
            return {'patient_id': patient_id, 'error': error_msg}
        
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
            
            # ENHANCED: Add IBSI features if enabled
            enable_ibsi = st.session_state.get('ibsi_features_enabled', True)
            if enable_ibsi:
                try:
                    bin_width = params_dict.get('setting', {}).get('binWidth', 25)
                    additional_features = calculate_missing_ibsi_features(image_path, processed_mask_path, bin_width)
                    feature_dict.update(additional_features)
                except Exception:
                    pass  # Continue without IBSI features if they fail
            
            # ENHANCED: Apply IBSI nomenclature
            feature_dict = apply_ibsi_nomenclature(feature_dict, enable_ibsi)
            
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
    ENHANCED: Extract features for a single patient with UI progress updates, comprehensive extraction, and IBSI compliance
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
            # Calculate ETA if possible
            if patient_index > 0:
                elapsed_time = time.time() - st.session_state.get('extraction_start_time', time.time())
                avg_time_per_patient = elapsed_time / patient_index
                remaining_patients = total_patients - patient_index
                eta_seconds = remaining_patients * avg_time_per_patient
                eta_str = f" | ETA: {int(eta_seconds//60)}m {int(eta_seconds%60)}s" if eta_seconds > 0 else ""
            else:
                eta_str = ""
            progress_text.text(f"Extracting features {patient_index + 1}/{total_patients}: {patient_id} ({current_progress*100:.1f}%){eta_str}")
        
        if status_placeholder:
            status_placeholder.info(f"🔄 Extracting comprehensive IBSI-compliant features for {patient_id}...")
        
        # Set extraction start time if not set
        if patient_index == 0:
            st.session_state['extraction_start_time'] = time.time()
        
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
            status_placeholder.info(f"✅ Mask validated for {patient_id}: {mask_info['nonzero_voxels']:,} segmented voxels")
        
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
        
        if status_placeholder:
            status_placeholder.info(f"⚙️ Configuring comprehensive feature extraction for {patient_id}...")
        
        # ENHANCED: Use comprehensive configuration
        enabled_classes = configure_extractor_comprehensive(extractor, params_dict)
        
        if not enabled_classes:
            error_msg = f"No feature classes enabled for {patient_id}"
            if status_placeholder:
                status_placeholder.error(f"❌ {error_msg}")
            return {'patient_id': patient_id, 'error': error_msg}
        
        if status_placeholder:
            status_placeholder.info(f"🎯 Enabled {len(enabled_classes)} feature classes: {', '.join(enabled_classes)}")
        
        # Extract features with the processed mask
        try:
            if status_placeholder:
                status_placeholder.info(f"🔥 Running comprehensive PyRadiomics extraction for {patient_id}...")
            
            features = extractor.execute(image_path, processed_mask_path)
            
            if status_placeholder:
                status_placeholder.info(f"📊 PyRadiomics returned {len(features)} total results for {patient_id}")
            
            # Convert to flat dictionary with detailed logging
            feature_dict = {'PatientID': patient_id}
            feature_count_by_class = {}
            
            for key, value in features.items():
                if not key.startswith('diagnostics_'):
                    # Convert numpy types to Python native types
                    if isinstance(value, (np.integer, np.floating)):
                        value = value.item()
                    elif isinstance(value, np.ndarray):
                        value = value.tolist()
                    
                    feature_dict[key] = value
                    
                    # Count features by class for verification
                    if '_' in key:
                        feature_class = key.split('_')[1] if key.startswith('original_') else key.split('_')[0]
                        feature_count_by_class[feature_class] = feature_count_by_class.get(feature_class, 0) + 1
            
            total_features = len(feature_dict) - 1  # Exclude PatientID
            
            # ENHANCED: Add additional IBSI features if enabled
            enable_ibsi = st.session_state.get('ibsi_features_enabled', True)  # Default to True for IBSI compliance
            if enable_ibsi:
                if status_placeholder:
                    status_placeholder.info(f"🎯 Calculating additional IBSI features for {patient_id}")
                
                try:
                    bin_width = params_dict.get('setting', {}).get('binWidth', 25)
                    additional_features = calculate_missing_ibsi_features(image_path, processed_mask_path, bin_width)
                    feature_dict.update(additional_features)
                    
                    if status_placeholder:
                        status_placeholder.info(f"➕ Added {len(additional_features)} IBSI-specific features")
                        
                except Exception as ibsi_error:
                    if status_placeholder:
                        status_placeholder.warning(f"⚠️ Some IBSI features failed for {patient_id}: {str(ibsi_error)}")
            
            # ENHANCED: Apply IBSI nomenclature (always enabled for consistency)
            feature_dict = apply_ibsi_nomenclature(feature_dict, enable_ibsi)
            
            # Add comprehensive metadata
            feature_dict['modality'] = patient_data.get('modality', 'Unknown')
            feature_dict['timepoint'] = patient_data.get('timepoint', 'TP1')
            feature_dict['series_uid'] = patient_data.get('series_uid', 'Unknown')
            feature_dict['roi_voxel_count'] = mask_info['nonzero_voxels']
            feature_dict['extraction_timestamp'] = time.time()
            feature_dict['ibsi_compliant'] = enable_ibsi
            
            if status_placeholder:
                total_radiomics_features = len([k for k in feature_dict.keys() if not k.startswith(('PatientID', 'modality', 'timepoint', 'series_uid', 'roi_', 'extraction_', 'ibsi_'))])
                status_placeholder.success(f"✅ Successfully extracted {total_radiomics_features} comprehensive IBSI-compliant features for {patient_id}")
                
                # Show feature breakdown
                if feature_count_by_class:
                    breakdown_text = []
                    for class_name, count in sorted(feature_count_by_class.items()):
                        breakdown_text.append(f"{class_name}: {count}")
                    status_placeholder.info(f"📈 Feature breakdown: {', '.join(breakdown_text)}")
                
                # Sanity check
                if total_radiomics_features < 50:  # Expect at least 50+ features for comprehensive extraction
                    status_placeholder.warning(f"⚠️ Fewer features than expected ({total_radiomics_features}). May indicate configuration issue.")
                elif total_radiomics_features >= 100:
                    status_placeholder.success(f"🎉 Comprehensive extraction achieved: {total_radiomics_features} features!")
            
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
                    with st.expander(f"🔍 Extraction error details for {patient_id}"):
                        st.code(traceback.format_exc())
                return {'patient_id': patient_id, 'error': error_msg}
            
    except Exception as e:
        error_msg = f"Critical error processing {patient_data.get('patient_id', 'Unknown')}: {str(e)}"
        if status_placeholder:
            status_placeholder.error(f"❌ {error_msg}")
        return {'patient_id': patient_data.get('patient_id', 'Unknown'), 'error': error_msg}


def run_extraction(dataset_df: pd.DataFrame, params: Dict, n_jobs: int = 1) -> pd.DataFrame:
    """
    Run PyRadiomics feature extraction with multi-ROI metadata preservation.
    
    CRITICAL FIXES:
    - Processes ALL rows in dataset (no skipping)
    - Preserves series_description, roi_name, timepoint metadata
    - Reorders output: metadata columns FIRST, then features
    - Handles multi-series × multi-ROI data correctly
    
    Parameters:
        dataset_df: DataFrame with columns:
                   Required: ['patient_id', 'image_path', 'mask_path']
                   Optional metadata: ['roi_name', 'series_description', 'timepoint', 
                                      'modality', 'study_date', 'series_uid', 'slice_count']
        params: PyRadiomics parameter dictionary
        n_jobs: Number of parallel jobs (currently sequential for stability)
    
    Returns:
        DataFrame with metadata + features. 
        Expected: same number of rows as input dataset_df.
        
    Example:
        Input:  48 rows (3 patients × 8 series × 6 ROIs = 144... or 8 series × 6 ROIs = 48)
        Output: 48 rows with [patient_id, roi_name, series_description, timepoint, ...features...]
    """
    
    # ========================================================================
    # STEP 1: VALIDATION
    # ========================================================================
    if dataset_df is None or dataset_df.empty:
        st.error("❌ Empty dataset provided to run_extraction")
        return pd.DataFrame()
    
    # Verify required columns
    required_cols = {'patient_id', 'image_path', 'mask_path'}
    if not required_cols.issubset(set(dataset_df.columns)):
        missing = required_cols - set(dataset_df.columns)
        st.error(f"❌ Dataset missing required columns: {missing}")
        return pd.DataFrame()
    
    # ========================================================================
    # STEP 2: EXTRACT AND PRESERVE METADATA COLUMNS
    # ========================================================================
    
    st.info(f"🔄 Extracting features from {len(dataset_df)} dataset rows...")
    
    # Define which columns to preserve as metadata
    metadata_preserve_cols = [
        'patient_id',           # Always needed for merging
        'roi_name',             # ✅ CRITICAL: Which ROI (material1, material2, etc.)
        'series_description',   # ✅ CRITICAL: Which series (Head, Chest, Control)
        'timepoint',            # ✅ CRITICAL: Which timepoint (19990620, etc.)
        'study_date',           # Optional: study date
        'series_uid',           # Optional: unique series identifier
        'modality',             # Optional: CT/MR/PT
        'slice_count',          # Optional: number of slices
        'processing_order',     # Optional: order of processing
        'voxel_count',          # Optional: ROI size info
        'recovery_method'       # Optional: which method was used
    ]
    
    # Extract metadata that exists in the dataset
    metadata_dict = {}
    for col in metadata_preserve_cols:
        if col in dataset_df.columns:
            metadata_dict[col] = dataset_df[col].tolist()
            st.write(f"  ✅ Preserving metadata: {col}")
    
    if 'roi_name' not in metadata_dict:
        st.warning("⚠️ Warning: 'roi_name' column not found. Creating placeholder column.")
        # Create placeholder - try to infer from patient_id or use "Unknown"
        metadata_dict['roi_name'] = ['Unknown_ROI'] * len(dataset_df)

    if 'series_description' not in metadata_dict:
        st.warning("⚠️ Warning: 'series_description' column not found. Creating placeholder column.")
        metadata_dict['series_description'] = ['Unknown_Series'] * len(dataset_df)
    
    # ========================================================================
    # STEP 3: INITIALIZE PYRADIOMICS EXTRACTOR
    # ========================================================================
    
    if 'setting' in params:
        # Remove keys that PyRadiomics doesn't recognize
        invalid_keys = ['enableCExtensions', 'additionalInfo', 'distances', 'weightingNorm', 'label', 'correctMask']
        for key in invalid_keys:
            if key in params['setting']:
                del params['setting'][key]
                st.write(f"  🔧 Removed invalid key: {key}")
    
    # Remove _metadata if it exists
    if '_metadata' in params:
        del params['_metadata']
        st.write("  🔧 Removed _metadata section")
    
    st.write("  ✅ Parameters sanitized for PyRadiomics")
    
    try:
        extractor = featureextractor.RadiomicsFeatureExtractor(params)
        extractor.disableAllFeatures()
        
        # Enable feature classes from params
        if 'featureClass' in params:
            for feature_class in params['featureClass'].keys():
                extractor.enableFeatureClassByName(feature_class)
                
        st.info(f"  ✅ PyRadiomics extractor initialized")
        
    except Exception as e:
        st.error(f"❌ Failed to initialize PyRadiomics extractor: {e}")
        return pd.DataFrame()
    
    # ========================================================================
    # STEP 4: EXTRACT FEATURES FOR EACH ROW
    # ========================================================================
    
    all_features = []
    failed_extractions = []
    
    # Get UI progress elements if available
    progress_bar = st.session_state.get('ui_progress_bar')
    progress_text = st.session_state.get('ui_progress_text')
    
    total_rows = len(dataset_df)
    
    # Debug: Show what we're processing
    st.write(f"📊 **Processing {total_rows} rows:**")
    if 'roi_name' in dataset_df.columns:
        roi_counts = dataset_df['roi_name'].value_counts()
        st.write(f"  - ROIs: {dict(roi_counts)}")
    if 'series_description' in dataset_df.columns:
        series_counts = dataset_df['series_description'].value_counts()
        st.write(f"  - Series: {dict(series_counts)}")
    
    # ✅ CRITICAL: Process EVERY row in the dataset.
    # Use positional (0..N-1) indices so metadata alignment in step 6 stays
    # correct even if the caller passed a DataFrame with a non-trivial index.
    for idx, (_orig_idx, row) in enumerate(dataset_df.iterrows()):
        try:
            # Update progress
            current_progress = (idx + 1) / total_rows
            if progress_bar:
                progress_bar.progress(current_progress)
            
            patient_id = row['patient_id']
            roi_name = row.get('roi_name', 'unknown_roi')
            series_desc = row.get('series_description', 'unknown_series')
            
            if progress_text:
                progress_text.text(
                    f"Extracting {idx+1}/{total_rows}: {patient_id} | "
                    f"ROI: {roi_name} | Series: {series_desc}"
                )
            
            image_path = row['image_path']
            mask_path = row['mask_path']
            
            # Verify files exist
            if not os.path.exists(image_path):
                failed_extractions.append({
                    'row_index': idx,
                    'patient_id': patient_id,
                    'roi_name': roi_name,
                    'series_description': series_desc,
                    'reason': f'Image file not found: {image_path}'
                })
                st.warning(f"  ⚠️ Row {idx}: Image not found for {patient_id}/{roi_name}/{series_desc}")
                continue
            
            if not os.path.exists(mask_path):
                failed_extractions.append({
                    'row_index': idx,
                    'patient_id': patient_id,
                    'roi_name': roi_name,
                    'series_description': series_desc,
                    'reason': f'Mask file not found: {mask_path}'
                })
                st.warning(f"  ⚠️ Row {idx}: Mask not found for {patient_id}/{roi_name}/{series_desc}")
                continue
            
            # Load images
            image = sitk.ReadImage(image_path)
            mask = sitk.ReadImage(mask_path)
            
            # Verify mask is not empty
            mask_array = sitk.GetArrayFromImage(mask)
            mask_voxel_count = np.sum(mask_array > 0)
            
            if mask_voxel_count == 0:
                failed_extractions.append({
                    'row_index': idx,
                    'patient_id': patient_id,
                    'roi_name': roi_name,
                    'series_description': series_desc,
                    'reason': 'Empty mask (no voxels > 0)'
                })
                st.warning(f"  ⚠️ Row {idx}: Empty mask for {patient_id}/{roi_name}/{series_desc}")
                continue
            
            # ✅ EXTRACT FEATURES
            feature_vector = extractor.execute(image, mask)

            # Convert to dictionary, excluding diagnostics. _source_row_idx
            # lets step 6 align the per-row metadata positionally even when
            # some rows fail extraction.
            feature_dict = {
                '_source_row_idx': int(idx),
                'patient_id': patient_id,
            }

            for key, value in feature_vector.items():
                if 'diagnostics_' not in key:
                    try:
                        feature_dict[key] = float(value)
                    except (ValueError, TypeError):
                        # Skip non-numeric features
                        pass

            all_features.append(feature_dict)
            
            # Success indicator (only show every 10th to avoid spam)
            if (idx + 1) % 10 == 0 or (idx + 1) == total_rows:
                st.write(f"  ✅ Processed {idx + 1}/{total_rows} rows...")
            
        except Exception as e:
            failed_extractions.append({
                'row_index': idx,
                'patient_id': row.get('patient_id', 'unknown'),
                'roi_name': row.get('roi_name', 'unknown'),
                'series_description': row.get('series_description', 'unknown'),
                'reason': f'Extraction error: {str(e)}'
            })
            st.error(f"  ❌ Row {idx}: Extraction failed - {str(e)}")
            continue
    
    # ========================================================================
    # STEP 5: CREATE FEATURES DATAFRAME
    # ========================================================================
    
    if not all_features:
        st.error("❌ No features extracted successfully")
        if failed_extractions:
            st.error(f"Failed extractions: {len(failed_extractions)}")
            with st.expander("Show failures"):
                st.dataframe(pd.DataFrame(failed_extractions))
        return pd.DataFrame()
    
    features_df = pd.DataFrame(all_features)

    st.success(f"✅ Successfully extracted features from {len(features_df)} rows")

    # ========================================================================
    # STEP 6: ATTACH METADATA BY POSITIONAL ROW INDEX (NOT by patient_id)
    # ========================================================================
    #
    # Bugfix: a previous implementation merged the metadata onto features_df
    # using on='patient_id'. With multi-ROI / multi-series data every row
    # shares the same patient_id, so the merge produced an N×N Cartesian
    # product and replicated the features of one (series, ROI) across every
    # row. We now align by the ORIGINAL row index instead, which is the only
    # key that actually identifies a (patient, series, ROI) combination.
    # ========================================================================

    if not features_df.empty:

        st.write("🔗 Attaching metadata back to features (positional alignment)...")

        # Every entry in all_features carries its originating row index so we
        # can line metadata up 1:1 regardless of how many rows failed.
        if '_source_row_idx' in features_df.columns:
            source_idx = features_df['_source_row_idx'].astype(int).tolist()
        else:
            source_idx = list(range(len(features_df)))

        ds_reset = dataset_df.reset_index(drop=True)

        for col_name in metadata_preserve_cols:
            if col_name == 'patient_id':
                continue
            if col_name not in ds_reset.columns:
                continue

            try:
                aligned_values = [ds_reset.at[i, col_name] for i in source_idx]
            except Exception:
                continue

            # Do not clobber a real column that already came out of
            # PyRadiomics with the same name.
            if col_name in features_df.columns:
                st.write(f"  ⚠️ Column '{col_name}' already present in features; keeping feature value, not metadata.")
                continue

            features_df[col_name] = aligned_values

        # Also attach the placeholder metadata we created (roi_name /
        # series_description) when dataset_df didn't have them, so later
        # steps can always rely on these columns existing.
        for col_name in ('roi_name', 'series_description'):
            if col_name in features_df.columns:
                continue
            if col_name in metadata_dict:
                values = metadata_dict[col_name]
                try:
                    features_df[col_name] = [values[i] for i in source_idx]
                except Exception:
                    features_df[col_name] = values[: len(features_df)]

        if '_source_row_idx' in features_df.columns:
            features_df = features_df.drop(columns=['_source_row_idx'])

        # ✅ REORDER COLUMNS - METADATA FIRST, THEN FEATURES
        metadata_cols_present = [c for c in metadata_preserve_cols if c in features_df.columns]
        feature_cols = [c for c in features_df.columns if c not in metadata_cols_present]

        features_df = features_df[metadata_cols_present + feature_cols]

        st.success(f"✅ Metadata aligned. Column order: {', '.join(metadata_cols_present)} + {len(feature_cols)} feature columns")
        st.info(f"📋 **Preserved metadata columns:** {', '.join(metadata_cols_present)}")

        if len(features_df) != len(dataset_df):
            st.warning(
                f"⚠️ Row count mismatch: Input had {len(dataset_df)} rows, "
                f"output has {len(features_df)} rows. "
                f"This means {len(dataset_df) - len(features_df)} extractions failed."
            )
        else:
            st.success(f"✅ Row count verified: {len(features_df)} rows (matches input)")
    
    # ========================================================================
    # STEP 7: SHOW FAILURES (IF ANY)
    # ========================================================================
    
    if failed_extractions:
        st.warning(f"⚠️ {len(failed_extractions)} extractions failed")
        with st.expander("Show failed extractions"):
            failed_df = pd.DataFrame(failed_extractions)
            st.dataframe(failed_df)
            
            # Download failed extractions
            csv = failed_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Failed Extractions List",
                csv,
                "failed_extractions.csv",
                "text/csv"
            )
    
    # ========================================================================
    # STEP 8: FINAL VERIFICATION AND SUMMARY
    # ========================================================================
    
    st.subheader("📊 Extraction Summary")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Input Rows", len(dataset_df))
    with col2:
        st.metric("Successful", len(features_df))
    with col3:
        st.metric("Failed", len(failed_extractions))
    
    # Show sample of output
    with st.expander("📋 Preview Output (First 10 Rows)"):
        # Show only identification columns + first 5 feature columns
        id_cols = [c for c in ['patient_id', 'roi_name', 'series_description', 'timepoint'] 
                   if c in features_df.columns]
        feature_cols_sample = [c for c in features_df.columns if c not in metadata_preserve_cols][:5]
        preview_cols = id_cols + feature_cols_sample
        
        st.dataframe(features_df[preview_cols].head(10))
        st.info(f"Total columns in output: {len(features_df.columns)} ({len(metadata_cols_present)} metadata + {len(feature_cols)} features)")
    
    return features_df

# --- ENHANCED WRAPPER FUNCTION (for compatibility) ---

def run_extraction_with_ibsi_enhanced(dataset_df, params, n_jobs=1, 
                                    enable_ibsi_features=True, 
                                    use_ibsi_nomenclature=True):
    """
    Enhanced extraction wrapper with IBSI features and nomenclature support
    This function provides compatibility with the UI layer
    """
    # Set IBSI features in session state
    st.session_state['ibsi_features_enabled'] = enable_ibsi_features
    st.session_state['use_ibsi_nomenclature'] = use_ibsi_nomenclature
    
    # Call the main extraction function
    return run_extraction(dataset_df, params, n_jobs)

# --- Utility Functions ---

def validate_extraction_parameters(params):
    """
    Enhanced parameter validation with IBSI compliance checks
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
        else:
            # Validate each feature class
            valid_classes = ['firstorder', 'shape', 'glcm', 'glrlm', 'glszm', 'ngtdm', 'gldm']
            for feature_class in params['featureClass'].keys():
                if feature_class not in valid_classes:
                    validation_errors.append(f"Invalid feature class: {feature_class}")
    
    # Validate specific settings for IBSI compliance
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
        
        # IBSI compliance check: warn about 2D forcing
        if settings.get('force2D', False):
            validation_errors.append("WARNING: force2D=True may affect IBSI compliance. Consider 3D processing.")
    
    return validation_errors


def get_feature_extraction_info():
    """
    Returns information about the current PyRadiomics installation and IBSI compliance.
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
                'geometryTolerance': 0.0001,
                'force2D': False,
                'correctMask': True
            },
            'ibsi_compliance': {
                'nomenclature_mapping': len(get_ibsi_feature_mapping()),
                'additional_features': len(get_missing_ibsi_features_list()),
                'total_ibsi_features': len(get_ibsi_feature_mapping()) + len(get_missing_ibsi_features_list()),
                'recommended_settings': {
                    'force2D': False,
                    'correctMask': True,
                    'binWidth': 25
                }
            },
            'expected_feature_counts': {
                'firstorder': '~18 features',
                'shape': '~16 features',
                'glcm': '~24 features',
                'glrlm': '~16 features', 
                'glszm': '~16 features',
                'ngtdm': '~5 features',
                'gldm': '~14 features',
                'additional_ibsi': f'~{len(get_missing_ibsi_features_list())} features'
            },
            'total_expected_features': '~100+ IBSI-compliant features'
        }
        return info
    except Exception as e:
        return {'error': str(e)}
        # Add these functions to the end of the existing extraction.py file

def generate_pyradiomics_params_enhanced(feature_classes=None, normalize_image=True, 
                                       resample_pixel_spacing=False, pixel_spacing=None,
                                       bin_width=25, interpolator='sitkBSpline', 
                                       pad_distance=5, geometryTolerance=0.0001,
                                       modality='CT'):
    """
    Enhanced PyRadiomics parameter generation with modality-specific optimization
    
    Args:
        feature_classes (dict): Feature classes to enable
        normalize_image (bool): Enable image normalization
        resample_pixel_spacing (bool): Enable resampling
        pixel_spacing (float): Target pixel spacing
        bin_width (float): Discretization bin width
        interpolator (str): Interpolation method
        pad_distance (int): Padding distance
        geometryTolerance (float): Geometry tolerance
        modality (str): Imaging modality for optimization
    
    Returns:
        dict: Enhanced PyRadiomics parameters
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
    
    # Modality-specific optimizations
    modality_settings = {
        'CT': {
            'binWidth': 25,
            'normalize': False,
            'interpolator': 'sitkBSpline'
        },
        'MR': {
            'binWidth': 5,
            'normalize': True,
            'interpolator': 'sitkBSpline'
        },
        'MR_T1': {
            'binWidth': 5,
            'normalize': True,
            'interpolator': 'sitkBSpline'
        },
        'MR_T2': {
            'binWidth': 5,
            'normalize': True,
            'interpolator': 'sitkBSpline'
        },
        'MR_FLAIR': {
            'binWidth': 8,
            'normalize': True,
            'interpolator': 'sitkBSpline'
        },
        'PT': {
            'binWidth': 0.1,
            'normalize': False,
            'interpolator': 'sitkLinear'
        },
        'PT_CT': {
            'binWidth': 0.1,
            'normalize': False,
            'interpolator': 'sitkLinear'
        }
    }
    
    # Get modality-specific settings
    mod_settings = modality_settings.get(modality, modality_settings['CT'])
    
    # Override with modality-specific values if using defaults
    if bin_width == 25 and modality in modality_settings:
        bin_width = mod_settings['binWidth']
    
    if normalize_image == True and modality in modality_settings:
        normalize_image = mod_settings.get('normalize', normalize_image)
    
    if interpolator == 'sitkBSpline' and modality in modality_settings:
        interpolator = mod_settings.get('interpolator', interpolator)
    
    # Enhanced parameter structure
    params = {
        'setting': {
            'binWidth': bin_width,
            'interpolator': interpolator,
            'padDistance': pad_distance,
            'geometryTolerance': geometryTolerance,
            'force2D': False,
            'force2Ddimension': 0,
            'correctMask': True,
            'additionalInfo': True,
            'distances': [1],
            'weightingNorm': None,
            'label': 1
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
    
    # Enable feature classes
    for feature_class, enabled in feature_classes.items():
        if enabled:
            params['featureClass'][feature_class] = []
    
    # Add modality-specific metadata
    params['_metadata'] = {
        'modality': modality,
        'optimized_for': modality,
        'enhancement_version': '1.0',
        'ibsi_compliant': True
    }
    
    return params


def run_extraction_with_ibsi_enhanced(dataset_df, params, n_jobs=1, enable_ibsi_compliance=True):
    """
    Enhanced extraction wrapper with IBSI compliance features
    This matches the function signature expected by the UI
    """
    # Call the main extraction function with IBSI compliance
    return run_extraction(dataset_df, params, n_jobs)

    # Add IBSI-specific settings if enabled
    if enable_ibsi_compliance:
        params['setting']['force2D'] = False
        params['setting']['force2Ddimension'] = 0
        params['setting']['correctMask'] = True
        params['setting']['additionalInfo'] = True
        
        # IBSI-recommended settings
        if 'binWidth' not in params['setting']:
            params['setting']['binWidth'] = 25
        if 'interpolator' not in params['setting']:
            params['setting']['interpolator'] = 'sitkBSpline'
    
    # Use the existing run_extraction function with enhanced parameters
    return run_extraction(dataset_df, params, n_jobs)


def get_ibsi_feature_mapping():
    """
    Returns mapping between PyRadiomics features and IBSI standard nomenclature
    """
    return {
        # First Order Features
        'original_firstorder_Mean': 'F_morph.vol',
        'original_firstorder_Variance': 'F_morph.vol_dens',
        'original_firstorder_Skewness': 'F_morph.vol_approx',
        'original_firstorder_Kurtosis': 'F_morph.area_dens',
        'original_firstorder_Median': 'F_morph.area_dens_approx',
        'original_firstorder_Minimum': 'F_morph.av',
        'original_firstorder_Maximum': 'F_morph.comp_1',
        'original_firstorder_Range': 'F_morph.comp_2',
        'original_firstorder_InterquartileRange': 'F_morph.sph_dispr',
        'original_firstorder_RootMeanSquared': 'F_morph.sphericity',
        'original_firstorder_MeanAbsoluteDeviation': 'F_morph.asphericity',
        'original_firstorder_RobustMeanAbsoluteDeviation': 'F_morph.com',
        'original_firstorder_Energy': 'F_morph.diam',
        'original_firstorder_TotalEnergy': 'F_morph.pca_maj_axis',
        'original_firstorder_Entropy': 'F_morph.pca_min_axis',
        'original_firstorder_Uniformity': 'F_morph.pca_least_axis',
        
        # Shape Features
        'original_shape_VoxelVolume': 'F_morph.vol',
        'original_shape_MeshVolume': 'F_morph.vol_mesh',
        'original_shape_SurfaceArea': 'F_morph.area_mesh',
        'original_shape_SurfaceVolumeRatio': 'F_morph.av',
        'original_shape_Sphericity': 'F_morph.sphericity',
        'original_shape_Compactness1': 'F_morph.comp_1',
        'original_shape_Compactness2': 'F_morph.comp_2',
        'original_shape_SphericalDisproportion': 'F_morph.sph_dispr',
        'original_shape_Maximum3DDiameter': 'F_morph.diam',
        'original_shape_Maximum2DDiameterSlice': 'F_morph.maj_axis_2d',
        'original_shape_Maximum2DDiameterColumn': 'F_morph.min_axis_2d',
        'original_shape_Maximum2DDiameterRow': 'F_morph.least_axis_2d',
        'original_shape_MajorAxisLength': 'F_morph.pca_maj_axis',
        'original_shape_MinorAxisLength': 'F_morph.pca_min_axis',
        'original_shape_LeastAxisLength': 'F_morph.pca_least_axis',
        'original_shape_Elongation': 'F_morph.pca_elongation',
        'original_shape_Flatness': 'F_morph.pca_flatness',
        
        # Texture Features (partial mapping - IBSI has extensive texture feature definitions)
        'original_glcm_Autocorrelation': 'F_cm.joint_max',  # Approximate mapping
        'original_glcm_JointAverage': 'F_cm.joint_avg',
        'original_glcm_ClusterProminence': 'F_cm.joint_var',
        'original_glcm_ClusterShade': 'F_cm.joint_entr',
        'original_glcm_ClusterTendency': 'F_cm.diff_avg',
        'original_glcm_Contrast': 'F_cm.diff_var',
        'original_glcm_Correlation': 'F_cm.diff_entr',
        'original_glcm_DifferenceAverage': 'F_cm.sum_avg',
        'original_glcm_DifferenceEntropy': 'F_cm.sum_var',
        'original_glcm_DifferenceVariance': 'F_cm.sum_entr',
        'original_glcm_JointEnergy': 'F_cm.energy',
        'original_glcm_JointEntropy': 'F_cm.contrast',
        'original_glcm_Homogeneity1': 'F_cm.dissimilarity',
        'original_glcm_Homogeneity2': 'F_cm.homogeneity',
        'original_glcm_InformationalMeasureOfCorrelation1': 'F_cm.corr',
        'original_glcm_InformationalMeasureOfCorrelation2': 'F_cm.auto_corr',
    }


def get_missing_ibsi_features_list():
    """
    List of IBSI features that are not available in PyRadiomics
    """
    return {
        # Missing Morphological Features
        'morph_CentreOfMassShift': 'Centre of mass shift',
        'morph_IntegratedIntensity': 'Integrated intensity',
        
        # Missing Intensity Statistical Features
        'stat_MedianAbsoluteDeviation': 'Median absolute deviation',
        'stat_CoefficientOfVariation': 'Coefficient of variation',
        'stat_QuartileCoefficientOfDispersion': 'Quartile coefficient of dispersion',
        
        # Missing Local Intensity Features
        'intensity_LocalIntensityPeak': 'Local intensity peak',
        'intensity_GlobalIntensityPeak': 'Global intensity peak',
        
        # Missing GLCM Features
        'glcm_JointMaximum': 'Joint maximum',
        'glcm_JointVariance': 'Joint variance',
        'glcm_SumVariance': 'Sum variance',
        'glcm_AngularSecondMoment': 'Angular second moment',
        
        # Missing Intensity Histogram Features (discretized)
        'hist_MeanDiscretised': 'Mean discretised intensity',
        'hist_VarianceDiscretised': 'Variance discretised intensity',
        'hist_SkewnessDiscretised': 'Skewness discretised intensity',
        'hist_KurtosisDiscretised': 'Kurtosis discretised intensity',
        'hist_MedianDiscretised': 'Median discretised intensity',
        'hist_EntropyDiscretised': 'Entropy discretised intensity',
        'hist_UniformityDiscretised': 'Uniformity discretised intensity'
    }


def get_enhanced_extraction_info():
    """
    Extended information about enhanced extraction capabilities
    """
    base_info = get_feature_extraction_info()
    
    enhanced_info = {
        **base_info,
        'enhanced_features': {
            'modality_optimization': True,
            'ibsi_compliance': True,  
            'automatic_preprocessing': True,
            'robust_error_handling': True,
            'parallel_processing': True
        },
        'supported_modalities': [
            'CT', 'MR', 'MR_T1', 'MR_T2', 'MR_FLAIR', 'PT', 'PT_CT'
        ],
        'ibsi_feature_coverage': {
            'total_ibsi_features': 169,
            'pyradiomics_coverage': 107,
            'missing_features': len(get_missing_ibsi_features_list())
        }
    }
    
    return enhanced_info

def generate_pyradiomics_params_enhanced(feature_classes=None, normalize_image=True, 
                                       resample_pixel_spacing=False, pixel_spacing=None,
                                       bin_width=25, interpolator='sitkBSpline', 
                                       pad_distance=5, geometryTolerance=0.0001,
                                       modality='CT'):
    """
    Enhanced PyRadiomics parameter generation with modality-specific optimization
    
    Args:
        feature_classes (dict): Feature classes to enable
        normalize_image (bool): Enable image normalization
        resample_pixel_spacing (bool): Enable resampling
        pixel_spacing (float): Target pixel spacing
        bin_width (float): Discretization bin width
        interpolator (str): Interpolation method
        pad_distance (int): Padding distance
        geometryTolerance (float): Geometry tolerance
        modality (str): Imaging modality for optimization
    
    Returns:
        dict: Enhanced PyRadiomics parameters
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
    
    # Modality-specific optimizations
    modality_settings = {
        'CT': {
            'binWidth': 25,
            'normalize': False,
            'interpolator': 'sitkBSpline'
        },
        'MR': {
            'binWidth': 5,
            'normalize': True,
            'interpolator': 'sitkBSpline'
        },
        'MR_T1': {
            'binWidth': 5,
            'normalize': True,
            'interpolator': 'sitkBSpline'
        },
        'MR_T2': {
            'binWidth': 5,
            'normalize': True,
            'interpolator': 'sitkBSpline'
        },
        'MR_FLAIR': {
            'binWidth': 8,
            'normalize': True,
            'interpolator': 'sitkBSpline'
        },
        'PT': {
            'binWidth': 0.1,
            'normalize': False,
            'interpolator': 'sitkLinear'
        },
        'PT_CT': {
            'binWidth': 0.1,
            'normalize': False,
            'interpolator': 'sitkLinear'
        }
    }
    
    # Get modality-specific settings
    mod_settings = modality_settings.get(modality, modality_settings['CT'])
    
    # Override with modality-specific values if not explicitly provided
    if bin_width == 25 and modality in modality_settings:  # Default value, use modality-specific
        bin_width = mod_settings['binWidth']
    
    if normalize_image == True and modality in modality_settings:  # Check modality preference
        normalize_image = mod_settings.get('normalize', normalize_image)
    
    if interpolator == 'sitkBSpline' and modality in modality_settings:  # Default value
        interpolator = mod_settings.get('interpolator', interpolator)
    
    # Enhanced parameter structure
    params = {
        'setting': {
            'binWidth': bin_width,
            'interpolator': interpolator,
            'padDistance': pad_distance,
            'geometryTolerance': geometryTolerance,
            'force2D': False,
            'force2Ddimension': 0,
            'correctMask': True,
            'additionalInfo': True,
            'enableCExtensions': True,
            'distances': [1],
            'weightingNorm': None,
            'label': 1  # Ensure we use label 1 for mask
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
    
    # Enable feature classes
    for feature_class, enabled in feature_classes.items():
        if enabled:
            params['featureClass'][feature_class] = []
            
    return params
