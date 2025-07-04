# extraction.py
"""
This module handles the radiomics feature extraction process.
It uses the PyRadiomics library to calculate features based on a
user-defined configuration.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yaml
import traceback
from radiomics import featureextractor, setVerbosity
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil
import multiprocessing as mp

# Set PyRadiomics verbosity. Errors will be logged to a file.
# This prevents the Streamlit app from being cluttered with PyRadiomics logs.
setVerbosity(logging.ERROR)

# Set up logging for radiomics
logging.getLogger('radiomics').setLevel(logging.ERROR)


def check_system_resources():
    """
    Check available system resources.
    
    Returns:
        dict: System resource information
    """
    try:
        # Get memory info
        memory = psutil.virtual_memory()
        available_ram_gb = memory.available / (1024**3)
        
        # Get CPU count
        cpu_count = mp.cpu_count()
        
        return {
            'available_ram_gb': available_ram_gb,
            'cpu_count': cpu_count,
            'total_ram_gb': memory.total / (1024**3)
        }
    except Exception as e:
        print(f"Error checking system resources: {e}")
        return {
            'available_ram_gb': 4.0,
            'cpu_count': 1,
            'total_ram_gb': 8.0
        }


def extract_features_single_patient(args):
    """
    Extract features for a single patient. This function is designed to be
    used with multiprocessing.
    
    Args:
        args (tuple): Tuple containing (patient_data, params_dict)
    
    Returns:
        tuple: (success, patient_id, features_dict or error_message)
    """
    patient_data, params = args
    patient_id = patient_data.get('PatientID', 'Unknown')
    
    try:
        print(f"[Worker] Processing patient {patient_id}")
        
        # Get file paths
        image_path = patient_data['image_path']
        mask_path = patient_data['mask_path']
        
        print(f"[Worker] Image: {image_path}")
        print(f"[Worker] Mask: {mask_path}")
        
        # Check if files exist
        import os
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        
        # Initialize feature extractor
        extractor = featureextractor.RadiomicsFeatureExtractor(params)
        
        # Extract features
        features = extractor.execute(image_path, mask_path)
        
        print(f"[Worker] Extracted {len(features)} raw features for {patient_id}")
        
        # Convert to regular dictionary and add patient ID
        feature_dict = {'PatientID': patient_id}
        feature_count = 0
        for key, value in features.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                feature_dict[key] = float(value)
                feature_count += 1
            else:
                feature_dict[key] = str(value)
        
        print(f"[Worker] Converted {feature_count} numeric features for {patient_id}")
        return True, patient_id, feature_dict
        
    except Exception as e:
        error_msg = f"Error type: {type(e).__name__}, Message: {str(e)}"
        print(f"[Worker] ❌ Failed {patient_id}: {error_msg}")
        return False, patient_id, error_msg


def generate_pyradiomics_params(feature_classes=None, normalize_image=True, 
                               resample_pixel_spacing=False, pixel_spacing=None,
                               bin_width=25, interpolator='sitkBSpline', 
                               pad_distance=5, geometryTolerance=0.0001):
    """
    Generate PyRadiomics parameter configuration.
    
    Args:
        feature_classes (dict): Dictionary of feature classes to enable
        normalize_image (bool): Whether to normalize the image
        resample_pixel_spacing (bool): Whether to resample pixel spacing
        pixel_spacing (float): Pixel spacing for resampling
        bin_width (int): Bin width for discretization
        interpolator (str): Interpolation method
        pad_distance (int): Padding distance
        geometryTolerance (float): Geometry tolerance
    
    Returns:
        dict: PyRadiomics parameter configuration
    """
    
    # Default feature classes if none provided
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
    
    # Create the parameter configuration
    params = {
        'setting': {
            'binWidth': bin_width,
            'interpolator': interpolator,
            'padDistance': pad_distance,
            'geometryTolerance': geometryTolerance
        },
        'imageType': {
            'Original': {}  # Original image type
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


def build_tab2_feature_extraction():
    """Builds the UI for the second tab: Feature Extraction."""
    
    # Check if preprocessing is done
    if not st.session_state.get('preprocessing_done', False):
        st.warning("⚠️ Please complete the data upload and pre-processing steps first.")
        return
    
    st.header("Step 2: Radiomics Feature Extraction")
    
    # PyRadiomics configuration
    st.subheader("🔧 PyRadiomics Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Feature Classes to Extract:**")
        feature_classes = {}
        feature_classes['firstorder'] = st.checkbox("First Order Statistics", value=True)
        feature_classes['shape'] = st.checkbox("Shape Features", value=True)
        feature_classes['glcm'] = st.checkbox("GLCM Features", value=True)
        feature_classes['glrlm'] = st.checkbox("GLRLM Features", value=True)
        feature_classes['glszm'] = st.checkbox("GLSZM Features", value=True)
        feature_classes['ngtdm'] = st.checkbox("NGTDM Features", value=True)
        feature_classes['gldm'] = st.checkbox("GLDM Features", value=True)
    
    with col2:
        st.write("**Preprocessing Settings:**")
        normalize_image = st.checkbox("Normalize Image", value=True)
        resample_pixel_spacing = st.checkbox("Resample Pixel Spacing", value=False)
        bin_width = st.number_input("Bin Width", min_value=1, max_value=100, value=25)
        
        if resample_pixel_spacing:
            pixel_spacing = st.number_input("Pixel Spacing (mm)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        else:
            pixel_spacing = None
    
    # Advanced settings
    with st.expander("⚙️ Advanced Settings"):
        interpolator = st.selectbox(
            "Interpolator",
            ["sitkBSpline", "sitkLinear", "sitkNearestNeighbor"],
            index=0
        )
        
        pad_distance = st.number_input("Pad Distance", min_value=0, max_value=20, value=5)
        
        geometryTolerance = st.number_input(
            "Geometry Tolerance", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.0001, 
            format="%.6f"
        )
    
    # Generate parameter file
    if st.button("🔄 Generate PyRadiomics Parameters", type="secondary"):
        with st.spinner("Generating parameter configuration..."):
            try:
                params = generate_pyradiomics_params(
                    feature_classes=feature_classes,
                    normalize_image=normalize_image,
                    resample_pixel_spacing=resample_pixel_spacing,
                    pixel_spacing=pixel_spacing,
                    bin_width=bin_width,
                    interpolator=interpolator,
                    pad_distance=pad_distance,
                    geometryTolerance=geometryTolerance
                )
                
                st.session_state.pyradiomics_params = params
                st.success("✅ PyRadiomics parameters generated successfully!")
                
                # Display the parameters
                with st.expander("📋 Generated Parameters"):
                    st.code(yaml.dump(params, default_flow_style=False), language='yaml')
                    
            except Exception as e:
                st.error(f"❌ Error generating parameters: {str(e)}")
                with st.expander("🔍 Error Details"):
                    st.code(traceback.format_exc())
    
    # Feature extraction
    if st.session_state.get('pyradiomics_params'):
        st.divider()
        st.subheader("🚀 Run Feature Extraction")
        
        # Resource check with error handling
        try:
            resource_info = check_system_resources()
            col1, col2, col3 = st.columns(3)
            with col1:
                ram_gb = resource_info.get('available_ram_gb', 0)
                st.metric("Available RAM", f"{ram_gb:.1f} GB")
            with col2:
                cpu_count = resource_info.get('cpu_count', 1)
                st.metric("CPU Cores", cpu_count)
            with col3:
                st.metric("Patients to Process", len(st.session_state.dataset_df))
        except Exception as e:
            st.warning(f"Could not get system resources: {str(e)}")
            cpu_count = 1
            ram_gb = 0
        
        # Parallel processing options
        use_parallel = st.checkbox("Enable Parallel Processing", value=True if cpu_count > 1 else False)
        if use_parallel:
            n_jobs = st.slider(
                "Number of parallel jobs:", 
                min_value=1, 
                max_value=min(cpu_count, len(st.session_state.dataset_df)), 
                value=min(4, cpu_count)
            )
        else:
            n_jobs = 1
        
        if st.button("🔥 Start Feature Extraction", type="primary"):
            with st.spinner("Extracting radiomics features... This may take several minutes."):
                try:
                    features_df = run_extraction(
                        st.session_state.dataset_df,
                        st.session_state.pyradiomics_params,
                        n_jobs=n_jobs
                    )
                    
                    if not features_df.empty:
                        st.success(f"✅ Successfully extracted {features_df.shape[1]-1} features from {features_df.shape[0]} patients!")
                        st.session_state.features_df = features_df
                        st.session_state.extraction_done = True
                        
                        # Display results
                        st.subheader("📊 Extraction Results")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Features", features_df.shape[1]-1)
                        with col2:
                            st.metric("Patients", features_df.shape[0])
                        with col3:
                            st.metric("Success Rate", f"{len(features_df)/len(st.session_state.dataset_df)*100:.1f}%")
                        
                        st.dataframe(features_df.head())
                        
                        # Download extracted features
                        csv = features_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📥 Download Extracted Features",
                            csv,
                            f"radiomics_features_{st.session_state.get('selected_modality', 'unknown')}.csv",
                            "text/csv",
                            key='download-features'
                        )
                    else:
                        st.error("❌ Feature extraction failed. Please check your data and parameters.")
                        
                except Exception as e:
                    st.error(f"❌ Feature extraction failed with error: {str(e)}")
                    with st.expander("🔍 Error Details"):
                        st.code(traceback.format_exc())


def run_extraction(dataset_df, params, n_jobs=1):
    """
    Run radiomics feature extraction on a dataset.
    
    Args:
        dataset_df (pd.DataFrame): DataFrame containing patient data with image_path and mask_path
        params (dict): PyRadiomics parameter configuration
        n_jobs (int): Number of parallel jobs to run (default: 1)
    
    Returns:
        pd.DataFrame: DataFrame containing extracted features
    """
    
    print(f"Starting feature extraction for {len(dataset_df)} patients with {n_jobs} parallel jobs...")
    print(f"Available columns in dataset: {list(dataset_df.columns)}")
    
    # Validate input DataFrame with flexible column naming
    # Try to find the patient ID column with different possible names
    patient_id_column = None
    possible_patient_id_names = ['PatientID', 'Patient_ID', 'patient_id', 'ID', 'Subject', 'subject_id', 'SubjectID']
    
    for col_name in possible_patient_id_names:
        if col_name in dataset_df.columns:
            patient_id_column = col_name
            break
    
    if patient_id_column is None:
        # If no patient ID column found, create one using the index
        print("Warning: No patient ID column found. Creating PatientID from index.")
        dataset_df = dataset_df.copy()
        dataset_df['PatientID'] = dataset_df.index.astype(str)
        patient_id_column = 'PatientID'
    
    # Check for required path columns
    required_path_columns = ['image_path', 'mask_path']
    missing_columns = [col for col in required_path_columns if col not in dataset_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}. Available columns: {list(dataset_df.columns)}")
    
    print(f"Using '{patient_id_column}' as patient ID column")
    
    feature_list = []
    failed_patients = []
    
    if n_jobs == 1:
        # Sequential processing
        print("Using sequential processing...")
        extractor = featureextractor.RadiomicsFeatureExtractor(params)
        
        for idx, row in dataset_df.iterrows():
            try:
                patient_id = row[patient_id_column]
                image_path = row['image_path']
                mask_path = row['mask_path']
                
                print(f"Processing patient {patient_id} ({idx+1}/{len(dataset_df)})")
                print(f"  Image path: {image_path}")
                print(f"  Mask path: {mask_path}")
                
                # Check if files exist
                import os
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                if not os.path.exists(mask_path):
                    raise FileNotFoundError(f"Mask file not found: {mask_path}")
                
                print(f"  Files exist - extracting features...")
                
                # Extract features
                features = extractor.execute(image_path, mask_path)
                
                print(f"  Extracted {len(features)} raw features")
                
                # Convert to regular dictionary and add patient ID
                feature_dict = {'PatientID': patient_id}
                feature_count = 0
                for key, value in features.items():
                    if isinstance(value, (int, float, np.integer, np.floating)):
                        feature_dict[key] = float(value)
                        feature_count += 1
                    else:
                        feature_dict[key] = str(value)
                
                print(f"  Converted {feature_count} numeric features")
                
                feature_list.append(feature_dict)
                print(f"✅ Successfully processed patient {patient_id} - Total features in dict: {len(feature_dict)}")
                print(f"  Current feature_list length: {len(feature_list)}")
                
            except Exception as e:
                patient_id = row[patient_id_column] if patient_id_column in row else f"Unknown_{idx}"
                print(f"❌ Error extracting features for patient {patient_id}: {str(e)}")
                print(f"  Error type: {type(e).__name__}")
                import traceback
                print(f"  Full traceback: {traceback.format_exc()}")
                failed_patients.append(patient_id)
                continue
    
    else:
        # Parallel processing
        print(f"Using parallel processing with {n_jobs} workers...")
        print(f"Total patients to process: {len(dataset_df)}")
        
        # Prepare arguments for parallel processing
        args_list = []
        for idx, row in dataset_df.iterrows():
            row_dict = row.to_dict()
            # Ensure PatientID is in the dictionary for parallel processing
            if 'PatientID' not in row_dict:
                row_dict['PatientID'] = row_dict[patient_id_column]
            args_list.append((row_dict, params))
        
        print(f"Prepared {len(args_list)} jobs for parallel processing")
        
        try:
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                # Submit all jobs
                future_to_patient = {}
                for i, args in enumerate(args_list):
                    future = executor.submit(extract_features_single_patient, args)
                    future_to_patient[future] = args[0]['PatientID']
                    print(f"Submitted job {i+1}/{len(args_list)} for patient {args[0]['PatientID']}")
                
                print(f"All {len(future_to_patient)} jobs submitted. Waiting for results...")
                
                # Collect results
                completed_count = 0
                for future in as_completed(future_to_patient):
                    patient_id = future_to_patient[future]
                    completed_count += 1
                    try:
                        success, returned_patient_id, result = future.result()
                        if success:
                            feature_list.append(result)
                            print(f"✅ Completed patient {returned_patient_id} ({completed_count}/{len(future_to_patient)})")
                        else:
                            failed_patients.append(returned_patient_id)
                            print(f"❌ Failed patient {returned_patient_id}: {result}")
                    except Exception as e:
                        failed_patients.append(patient_id)
                        print(f"❌ Exception for patient {patient_id}: {str(e)}")
                        
        except Exception as e:
            print(f"❌ Error in parallel processing: {str(e)}")
            print("Falling back to sequential processing...")
            # Fallback to sequential processing
            return run_extraction(dataset_df, params, n_jobs=1)
    
    # Final results summary
    print(f"\n=== EXTRACTION SUMMARY ===")
    print(f"Total patients in dataset: {len(dataset_df)}")
    print(f"Successfully processed: {len(feature_list)}")
    print(f"Failed: {len(failed_patients)}")
    
    if failed_patients:
        print(f"Failed patients: {failed_patients}")
    
    if not feature_list:
        print("❌ Error: No successful feature extractions")
        return pd.DataFrame()
    
    # Convert to DataFrame
    print("Converting results to DataFrame...")
    features_df = pd.DataFrame(feature_list)
    
    # Sort columns: PatientID first, then alphabetically
    columns = ['PatientID'] + sorted([col for col in features_df.columns if col != 'PatientID'])
    features_df = features_df[columns]
    
    print(f"✅ Successfully extracted {features_df.shape[1]-1} features from {features_df.shape[0]} patients")
    print(f"Final DataFrame shape: {features_df.shape}")
    
    return features_df
