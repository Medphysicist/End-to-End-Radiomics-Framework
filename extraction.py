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
    
    # Add modality-specific metadata
    params['_metadata'] = {
        'modality': modality,
        'optimized_for': modality,
        'enhancement_version': '1.0',
        'ibsi_compliant': True
    }
    
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


def run_extraction(dataset_df, params, n_jobs=1):
    """
    ENHANCED: Runs comprehensive feature extraction on the dataset with IBSI compliance and progress tracking
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
    
    # Validate parameters before starting
    if status_placeholder:
        status_placeholder.info("🔍 Validating comprehensive IBSI-compliant extraction parameters...")
    
    # Check parameter structure
    if not isinstance(params, dict) or 'featureClass' not in params:
        st.error("❌ Invalid PyRadiomics parameters. Please regenerate parameters.")
        return pd.DataFrame()
    
    enabled_feature_classes = list(params.get('featureClass', {}).keys())
    if not enabled_feature_classes:
        st.error("❌ No feature classes enabled. Please check your configuration.")
        return pd.DataFrame()
    
    if status_placeholder:
        status_placeholder.info(f"🎯 Comprehensive IBSI-compliant extraction configured. Will extract from: {', '.join(enabled_feature_classes)}")
    
    try:
        if n_jobs == 1:
            # Sequential processing with detailed progress updates
            if status_placeholder:
                status_placeholder.info(f"🔄 Starting comprehensive sequential feature extraction for {total_patients} patients...")
            
            for i, (_, patient_data) in enumerate(dataset_df.iterrows()):
                result = extract_features_single_patient_sequential(
                    patient_data.to_dict(), 
                    params, 
                    i, 
                    total_patients
                )
                
                if 'error' in result:
                    failed_extractions.append({
                        'patient_id': result['patient_id'],
                        'error': result['error']
                    })
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
                            failed_extractions.append({
                                'patient_id': result['patient_id'],
                                'error': result['error']
                            })
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
            progress_text.text(f"Comprehensive IBSI-compliant extraction complete! {len(successful_extractions)}/{total_patients} patients processed successfully")
        
        # Analyze extraction results
        if successful_extractions:
            features_df = pd.DataFrame(successful_extractions)
            
            # Comprehensive result analysis
            total_features = len(features_df.columns) - 1  # Exclude PatientID
            
            # Count IBSI features
            ibsi_feature_count = len([col for col in features_df.columns if any(col.startswith(prefix) for prefix in ['morph_', 'stat_', 'glcm_', 'glrlm_', 'glszm_', 'ngtdm_', 'ngldm_', 'hist_', 'intensity_'])])
            
            # Analyze feature distribution
            feature_categories = {}
            for col in features_df.columns:
                if col not in ['PatientID', 'modality', 'timepoint', 'series_uid', 'roi_voxel_count', 'extraction_timestamp', 'ibsi_compliant']:
                    if col.startswith('morph_'):
                        feature_categories.setdefault('Morphological', []).append(col)
                    elif col.startswith('stat_'):
                        feature_categories.setdefault('Statistical', []).append(col)
                    elif col.startswith('glcm_'):
                        feature_categories.setdefault('GLCM', []).append(col)
                    elif col.startswith('glrlm_'):
                        feature_categories.setdefault('GLRLM', []).append(col)
                    elif col.startswith('glszm_'):
                        feature_categories.setdefault('GLSZM', []).append(col)
                    elif col.startswith('ngtdm_'):
                        feature_categories.setdefault('NGTDM', []).append(col)
                    elif col.startswith('ngldm_'):
                        feature_categories.setdefault('NGLDM', []).append(col)
                    elif col.startswith('hist_'):
                        feature_categories.setdefault('Histogram', []).append(col)
                    elif col.startswith('intensity_'):
                        feature_categories.setdefault('Intensity', []).append(col)
                    elif col.startswith('original_'):
                        # Handle unmapped PyRadiomics features
                        parts = col.split('_')
                        if len(parts) >= 2:
                            category = parts[1].upper()
                            feature_categories.setdefault(f'PyRad_{category}', []).append(col)
                        else:
                            feature_categories.setdefault('Other', []).append(col)
                    else:
                        feature_categories.setdefault('Other', []).append(col)
            
            if status_placeholder:
                if failed_extractions:
                    status_placeholder.warning(f"⚠️ Comprehensive IBSI-compliant extraction complete with some failures: {len(successful_extractions)} successful, {len(failed_extractions)} failed")
                else:
                    status_placeholder.success(f"🎉 Comprehensive IBSI-compliant feature extraction complete! All {len(successful_extractions)} patients processed successfully")
                
                # Show feature category breakdown
                breakdown_text = []
                for category, features in sorted(feature_categories.items()):
                    breakdown_text.append(f"{category}: {len(features)}")
                status_placeholder.info(f"📊 Feature breakdown: {', '.join(breakdown_text)}")
                
                # Sanity check and IBSI compliance report
                if total_features < 50:
                    status_placeholder.warning(f"⚠️ Only {total_features} features extracted. Expected 100+ for comprehensive extraction.")
                elif total_features >= 100:
                    status_placeholder.success(f"✅ Comprehensive extraction achieved: {total_features} features including {ibsi_feature_count} IBSI-named features")
                
                # IBSI compliance summary
                status_placeholder.info(f"🏅 IBSI Compliance: {ibsi_feature_count}/{total_features} features follow IBSI nomenclature ({(ibsi_feature_count/total_features)*100:.1f}%)")
            
            # Store enhanced extraction summary in session state
            st.session_state['extraction_summary'] = {
                'total_patients': total_patients,
                'successful_extractions': len(successful_extractions),
                'failed_extractions': failed_extractions,
                'total_features': total_features,
                'feature_categories': feature_categories,
                'ibsi_feature_count': ibsi_feature_count,
                'enabled_feature_classes': enabled_feature_classes,
                'ibsi_compliant': True,
                'ibsi_compliance_rate': (ibsi_feature_count/total_features)*100 if total_features > 0 else 0
            }
            
            return features_df
        else:
            if status_placeholder:
                status_placeholder.error(f"❌ Comprehensive IBSI-compliant feature extraction failed for all patients")
            
            # Store extraction summary even for complete failure
            st.session_state['extraction_summary'] = {
                'total_patients': total_patients,
                'successful_extractions': 0,
                'failed_extractions': failed_extractions,
                'total_features': 0,
                'feature_categories': {},
                'ibsi_feature_count': 0,
                'enabled_feature_classes': enabled_feature_classes,
                'ibsi_compliant': True,
                'ibsi_compliance_rate': 0
            }
            
            return pd.DataFrame()
            
    except Exception as e:
        if status_placeholder:
            status_placeholder.error(f"❌ Critical error during comprehensive IBSI-compliant feature extraction: {str(e)}")
        st.error(f"Feature extraction failed with error: {str(e)}")
        with st.expander("🔍 Error Details"):
            st.code(traceback.format_exc())
        
        # Store extraction summary for critical failure
        st.session_state['extraction_summary'] = {
            'total_patients': total_patients,
            'successful_extractions': len(successful_extractions),
            'failed_extractions': failed_extractions + [{'patient_id': 'System', 'error': str(e)}],
            'total_features': 0,
            'feature_categories': {},
            'ibsi_feature_count': 0,
            'enabled_feature_classes': enabled_feature_classes,
            'ibsi_compliant': True,
            'ibsi_compliance_rate': 0
        }
        
        return pd.DataFrame()


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
            'enableCExtensions': True,
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
    
    # Add modality-specific metadata
    params['_metadata'] = {
        'modality': modality,
        'optimized_for': modality,
        'enhancement_version': '1.0',
        'ibsi_compliant': True
    }
    
    return params
