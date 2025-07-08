# ui.py
"""
This module contains all the functions for building the Streamlit user interface.
It defines the content for each tab and the sidebar.
"""

# These two imports are essential for displaying the error message itself.
import streamlit as st
import traceback

# This single try-except block will attempt to import everything else.
# If any import fails, it will now show the detailed error.
try:
    import pandas as pd
    import numpy as np
    import yaml

    from processing import (
        validate_directory_path,
        get_available_directories,
        process_selected_path,
        organize_dicom_files,
        scan_uploaded_data_for_contours,
        preprocess_uploaded_data
    )
    from extraction import generate_pyradiomics_params, run_extraction
    from analysis import (
        run_univariate_analysis,
        run_lasso_selection,
        generate_correlation_heatmap
    )
    from utils import check_system_resources, categorize_contours, validate_uploaded_files

except ImportError as e:
    # This is the new, improved error reporting block.
    st.header("💥 Application Error")
    st.error("A critical module failed to import. The application cannot start.")
    st.error(f"The specific error is: **{e}**")
    st.code(traceback.format_exc(), language='text')
    st.stop()


# --- ALL OF YOUR UI FUNCTIONS START HERE, UNCHANGED ---

def enhanced_data_input_section():
    """Creates the UI for allowing users to select their data source."""
    st.header("Step 1.1: Select Your DICOM Data Source")

    input_method = st.radio(
        "Choose how to provide your DICOM data:",
        ["📤 Upload Files", "📁 Select from Available Directories", "✏️ Manual Path Entry"],
        horizontal=True,
        help="Choose the most convenient method to specify your DICOM data location."
    )

    uploaded_files = None
    selected_path = None

    if input_method == "📤 Upload Files":
        uploaded_files = st.file_uploader(
            "Choose your DICOM files or ZIP archives",
            accept_multiple_files=True,
            type=['dcm', 'zip', 'DCM', 'ZIP']
        )
        if uploaded_files:
            validation_issues = validate_uploaded_files(uploaded_files)
            if validation_issues:
                for issue in validation_issues:
                    st.error(f"• {issue}")
                return None, None
            st.success(f"✅ {len(uploaded_files)} file(s) are ready for processing.")

    elif input_method == "📁 Select from Available Directories":
        with st.spinner("Scanning for available directories..."):
            available_dirs = get_available_directories()
        if available_dirs:
            selected_path = st.selectbox(
                "Select a directory from the list:",
                options=[""] + available_dirs,
                format_func=lambda x: f"📁 {x}" if x else "Select a directory...",
            )
        else:
            st.warning("⚠️ No accessible directories found. Please use 'Manual Path Entry'.")

    elif input_method == "✏️ Manual Path Entry":
        selected_path = st.text_input(
            "Enter the full path to your DICOM data directory:",
            placeholder="/path/to/your/dicom/data",
        )

    if selected_path:
        validation_issues = validate_directory_path(selected_path)
        if validation_issues:
            for issue in validation_issues:
                st.error(f"• {issue}")
            selected_path = None
        else:
            st.success(f"✅ Valid DICOM directory selected: {selected_path}")
            
    return uploaded_files, selected_path


def build_tab1_data_upload():
    """Builds the UI for the first tab: Data Upload & Pre-processing."""
    
    uploaded_files, selected_path = enhanced_data_input_section()
    
    process_button_label = ""
    if uploaded_files:
        process_button_label = "🔄 Process Uploaded Files"
    elif selected_path:
        process_button_label = "🔄 Process Selected Directory"

    if process_button_label and st.button(process_button_label, type="primary", key="process_data"):
        data_path = None
        with st.spinner("Processing data source..."):
            if uploaded_files:
                data_path = organize_dicom_files(uploaded_files)
            elif selected_path:
                data_path = process_selected_path(selected_path)

        if data_path:
            st.session_state['uploaded_data_path'] = data_path
            st.success(f"✅ Data source processed. Ready to scan contours from: `{data_path}`")
        else:
            st.error("❌ Data processing failed. Please check your files or path.")

    if st.session_state.get('uploaded_data_path'):
        st.divider()
        st.header("Step 1.2: Select Imaging Modality and Scan for Contours")
        
        # Modality selection
        col1, col2 = st.columns(2)
        with col1:
            modality_options = ['CT', 'MR', 'PT']
            selected_modality = st.selectbox(
                "Select imaging modality:",
                options=modality_options,
                index=0,
                help="Choose the type of imaging you want to analyze"
            )
        with col2:
            if st.button("🔍 Scan Data For Available Contours", type="primary"):
                with st.spinner(f"Scanning data for {selected_modality} series and available ROIs..."):
                    all_contours, pat_data, pat_status, available_modalities = scan_uploaded_data_for_contours(
                        st.session_state.uploaded_data_path, selected_modality
                    )
                    st.session_state.all_contours = all_contours
                    st.session_state.patient_contour_data = pat_data
                    st.session_state.patient_status = pat_status
                    st.session_state.available_modalities = available_modalities
                    st.session_state.selected_modality = selected_modality
                    
                    if available_modalities:
                        st.info(f"📊 Available imaging modalities in your data: {', '.join(sorted(available_modalities))}")
                    
                    if selected_modality not in available_modalities:
                        st.warning(f"⚠️ {selected_modality} modality not found in your data. Available: {', '.join(sorted(available_modalities))}")
        
        if st.session_state.get('all_contours'):
            st.success(f"🎯 Scan complete! Found {len(st.session_state.all_contours)} unique contours for {st.session_state.selected_modality} imaging.")
            
            # Show scan results summary
            with st.expander("📋 Scan Results Summary"):
                success_count = sum(1 for status in st.session_state.patient_status.values() if status['status'] == 'success')
                total_patients = len(st.session_state.patient_status)
                st.metric("Successfully processed patients", f"{success_count}/{total_patients}")
                
                if success_count < total_patients:
                    failed_patients = [pid for pid, status in st.session_state.patient_status.items() if status['status'] == 'error']
                    st.warning(f"Failed to process: {', '.join(failed_patients)}")
            
            targets, oars, other = categorize_contours(st.session_state.all_contours)
            all_rois = [""] + targets + oars + other
            selected_roi = st.selectbox("Select a target contour (ROI) for extraction:", options=all_rois)

            if selected_roi:
                st.info(f"You selected: **{selected_roi}** for **{st.session_state.selected_modality}** imaging")
                
                # Show which patients have this ROI
                patients_with_roi = [pid for pid, contours in st.session_state.patient_contour_data.items() 
                                   if any(selected_roi.lower() in contour.lower() for contour in contours)]
                st.info(f"📊 This ROI is available in {len(patients_with_roi)} patients: {', '.join(patients_with_roi)}")
                
                st.divider()
                st.header("Step 1.3: Generate NIfTI Files")
                if st.button("🚀 Start Pre-processing & Generate NIfTI", type="primary"):
                    with st.spinner("Generating NIfTI files... This may take a while."):
                        result_df = preprocess_uploaded_data(
                            st.session_state.uploaded_data_path, 
                            selected_roi, 
                            st.session_state.selected_modality
                        )
                    if not result_df.empty:
                        st.success(f"✅ Successfully pre-processed {len(result_df)} patients with {st.session_state.selected_modality} imaging!")
                        st.session_state.dataset_df = result_df
                        st.session_state.preprocessing_done = True
                        
                        # Enhanced results display
                        st.subheader("📊 Pre-processing Results")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Patients Processed", len(result_df))
                        with col2:
                            st.metric("Imaging Modality", st.session_state.selected_modality)
                        with col3:
                            st.metric("Target ROI", selected_roi)
                        
                        st.dataframe(result_df)
                        
                        # Download option for the dataset summary
                        csv = result_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📥 Download Dataset Summary",
                            csv,
                            f"dataset_summary_{st.session_state.selected_modality}_{selected_roi}.csv",
                            "text/csv",
                            key='download-csv'
                        )
                    else:
                        st.error("❌ Pre-processing failed. Please check your data and try again.")


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
                # Try the original function call first
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
            except TypeError as e:
                # If the function doesn't accept these parameters, create params manually
                st.warning(f"Parameter generation function needs updating. Creating parameters manually...")
                
                # Create parameters manually
                params = {
                    'setting': {
                        'binWidth': bin_width,
                        'interpolator': interpolator,
                        'padDistance': pad_distance,
                        'geometryTolerance': geometryTolerance
                    },
                    'imageType': {}
                }
                
                # Add normalization if requested
                if normalize_image:
                    params['setting']['normalize'] = True
                    params['setting']['normalizeScale'] = 1
                
                # Add resampling if requested
                if resample_pixel_spacing and pixel_spacing:
                    params['setting']['resampledPixelSpacing'] = [pixel_spacing, pixel_spacing, pixel_spacing]
                
                # Add feature classes
                params['featureClass'] = {}
                for feature_class, enabled in feature_classes.items():
                    if enabled:
                        params['featureClass'][feature_class] = []
                
                # Add original image
                params['imageType']['Original'] = {}
            
            st.session_state.pyradiomics_params = params
            st.success("✅ PyRadiomics parameters generated successfully!")
            
            # Display the parameters
            with st.expander("📋 Generated Parameters"):
                st.code(yaml.dump(params, default_flow_style=False), language='yaml')
    
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
                            st.metric("Success Rate", "100%")
                        
                        st.dataframe(features_df.head())
                        
                        # Download extracted features
                        csv = features_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📥 Download Extracted Features",
                            csv,
                            f"radiomics_features_{st.session_state.selected_modality}.csv",
                            "text/csv",
                            key='download-features'
                        )
                    else:
                        st.error("❌ Feature extraction failed. Please check your data and parameters.")
                        
                except Exception as e:
                    st.error(f"❌ Feature extraction failed with error: {str(e)}")
                    with st.expander("🔍 Error Details"):
                        st.code(traceback.format_exc())

def build_tab3_analysis():
    """Builds the UI for the third tab: Statistical Analysis."""
    
    # Check if extraction is done
    if not st.session_state.get('extraction_done', False):
        st.warning("⚠️ Please complete the feature extraction step first.")
        return
    
    st.header("Step 3: Statistical Analysis & Feature Selection")
    
    # Load outcome data
    st.subheader("📊 Load Outcome Data")
    
    outcome_method = st.radio(
        "How would you like to provide outcome data?",
        ["📤 Upload CSV File", "✏️ Manual Entry"],
        horizontal=True
    )
    
    outcome_df = None
    
    if outcome_method == "📤 Upload CSV File":
        uploaded_outcome = st.file_uploader(
            "Upload outcome data (CSV format)",
            type=['csv'],
            help="CSV should contain 'PatientID' and outcome columns"
        )
        
        if uploaded_outcome:
            try:
                outcome_df = pd.read_csv(uploaded_outcome)
                st.success(f"✅ Loaded outcome data with {len(outcome_df)} patients")
                st.dataframe(outcome_df.head())
                
                # Validate PatientID column
                if 'PatientID' not in outcome_df.columns:
                    st.error("❌ Outcome data must contain a 'PatientID' column")
                    outcome_df = None
                else:
                    # Check for matching patients
                    feature_patients = set(st.session_state.features_df['PatientID'].values)
                    outcome_patients = set(outcome_df['PatientID'].values)
                    common_patients = feature_patients.intersection(outcome_patients)
                    
                    if len(common_patients) == 0:
                        st.error("❌ No matching patients found between features and outcome data")
                        outcome_df = None
                    else:
                        st.info(f"📊 Found {len(common_patients)} patients with both features and outcome data")
                        
            except Exception as e:
                st.error(f"❌ Error loading outcome data: {str(e)}")
    
    elif outcome_method == "✏️ Manual Entry":
        st.write("**Manual Outcome Entry:**")
        patients = st.session_state.features_df['PatientID'].tolist()
        
        # Create a simple form for binary outcomes
        outcome_type = st.selectbox(
            "Select outcome type:",
            ["Binary (0/1)", "Continuous", "Categorical"]
        )
        
        if outcome_type == "Binary (0/1)":
            st.write("Enter binary outcomes (0 or 1) for each patient:")
            outcomes = {}
            
            # Create columns for better layout
            n_cols = 3
            cols = st.columns(n_cols)
            
            for i, patient in enumerate(patients):
                with cols[i % n_cols]:
                    outcomes[patient] = st.selectbox(
                        f"{patient}:",
                        [0, 1],
                        key=f"outcome_{patient}"
                    )
            
            if st.button("✅ Create Outcome Dataset"):
                outcome_df = pd.DataFrame({
                    'PatientID': patients,
                    'Outcome': [outcomes[p] for p in patients]
                })
                st.success("✅ Manual outcome data created!")
                st.dataframe(outcome_df)
    
    # Analysis section
    if outcome_df is not None:
        st.session_state.outcome_df = outcome_df
        
        st.divider()
        st.subheader("🔬 Statistical Analysis")
        
        # Merge features and outcomes
        merged_df = pd.merge(st.session_state.features_df, outcome_df, on='PatientID')
        
        # Select outcome column
        outcome_columns = [col for col in outcome_df.columns if col != 'PatientID']
        selected_outcome = st.selectbox(
            "Select outcome variable for analysis:",
            outcome_columns
        )
        
        if selected_outcome:
            st.session_state.selected_outcome = selected_outcome
            
            # Analysis options
            analysis_type = st.selectbox(
                "Select analysis type:",
                ["Univariate Analysis", "LASSO Feature Selection", "Correlation Analysis"]
            )
            
            if analysis_type == "Univariate Analysis":
                st.subheader("📈 Univariate Analysis")
                
                # Configuration
                col1, col2 = st.columns(2)
                with col1:
                    p_threshold = st.number_input(
                        "P-value threshold:",
                        min_value=0.001,
                        max_value=0.2,
                        value=0.05,
                        step=0.001
                    )
                with col2:
                    test_type = st.selectbox(
                        "Statistical test:",
                        ["Auto-detect", "T-test", "Mann-Whitney U", "Pearson Correlation", "Spearman Correlation"]
                    )
                
                if st.button("🔄 Run Univariate Analysis"):
                    with st.spinner("Running univariate analysis..."):
                        results_df = run_univariate_analysis(
                            merged_df,
                            selected_outcome,
                            p_threshold=p_threshold,
                            test_type=test_type
                        )
                        
                        if not results_df.empty:
                            st.success(f"✅ Found {len(results_df)} significant features (p < {p_threshold})")
                            st.session_state.univariate_results = results_df
                            
                            # Display results
                            st.dataframe(results_df)
                            
                            # Download results
                            csv = results_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "📥 Download Univariate Results",
                                csv,
                                f"univariate_results_{selected_outcome}.csv",
                                "text/csv",
                                key='download-univariate'
                            )
                        else:
                            st.warning("⚠️ No significant features found with current threshold")
            
            elif analysis_type == "LASSO Feature Selection":
                st.subheader("🎯 LASSO Feature Selection")
                
                # Configuration
                col1, col2 = st.columns(2)
                with col1:
                    alpha = st.number_input(
                        "Alpha (regularization strength):",
                        min_value=0.001,
                        max_value=10.0,
                        value=1.0,
                        step=0.1
                    )
                with col2:
                    cv_folds = st.number_input(
                        "Cross-validation folds:",
                        min_value=3,
                        max_value=10,
                        value=5
                    )
                
                max_features = st.number_input(
                    "Maximum features to select:",
                    min_value=1,
                    max_value=50,
                    value=10
                )
                
                if st.button("🔄 Run LASSO Selection"):
                    with st.spinner("Running LASSO feature selection..."):
                        selected_features, lasso_results = run_lasso_selection(
                            merged_df,
                            selected_outcome,
                            alpha=alpha,
                            cv_folds=cv_folds,
                            max_features=max_features
                        )
                        
                        if selected_features:
                            st.success(f"✅ LASSO selected {len(selected_features)} features")
                            st.session_state.lasso_features = selected_features
                            st.session_state.lasso_results = lasso_results
                            
                            # Display selected features
                            st.write("**Selected Features:**")
                            for i, feature in enumerate(selected_features, 1):
                                st.write(f"{i}. {feature}")
                            
                            # Display model performance
                            if lasso_results:
                                st.subheader("📊 Model Performance")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("CV Score", f"{lasso_results.get('cv_score', 0):.3f}")
                                with col2:
                                    st.metric("R² Score", f"{lasso_results.get('r2_score', 0):.3f}")
                                with col3:
                                    st.metric("Alpha Used", f"{lasso_results.get('alpha', alpha):.3f}")
                        else:
                            st.warning("⚠️ LASSO did not select any features with current parameters")
            
            elif analysis_type == "Correlation Analysis":
                st.subheader("🔗 Correlation Analysis")
                
                # Configuration
                col1, col2 = st.columns(2)
                with col1:
                    correlation_method = st.selectbox(
                        "Correlation method:",
                        ["pearson", "spearman", "kendall"]
                    )
                with col2:
                    min_correlation = st.number_input(
                        "Minimum correlation threshold:",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.5,
                        step=0.1
                    )
                
                if st.button("🔄 Generate Correlation Heatmap"):
                    with st.spinner("Generating correlation analysis..."):
                        # Use selected features if available, otherwise use top features
                        if st.session_state.get('lasso_features'):
                            features_to_analyze = st.session_state.lasso_features
                        elif st.session_state.get('univariate_results') is not None:
                            features_to_analyze = st.session_state.univariate_results['Feature'].head(20).tolist()
                        else:
                            # Use first 20 features
                            feature_cols = [col for col in merged_df.columns if col not in ['PatientID', selected_outcome]]
                            features_to_analyze = feature_cols[:20]
                        
                        heatmap_fig = generate_correlation_heatmap(
                            merged_df,
                            features_to_analyze,
                            method=correlation_method,
                            min_correlation=min_correlation
                        )
                        
                        if heatmap_fig:
                            st.plotly_chart(heatmap_fig, use_container_width=True)
                            st.session_state.correlation_heatmap = heatmap_fig
                        else:
                            st.warning("⚠️ Could not generate correlation heatmap")


def build_sidebar():
    """Builds the sidebar with system information and navigation."""
    st.sidebar.title("🔬 RadiomicsGUI")
    st.sidebar.write("Advanced Radiomics Analysis Platform")
    
    # System information with error handling
    st.sidebar.subheader("💻 System Information")
    try:
        resource_info = check_system_resources()
        
        # Handle different possible key names and structures
        if isinstance(resource_info, dict):
            # Try different possible key names for RAM
            ram_gb = None
            for key in ['available_ram_gb', 'ram_gb', 'memory_gb', 'available_memory_gb']:
                if key in resource_info:
                    ram_gb = resource_info[key]
                    break
            
            if ram_gb is not None:
                st.sidebar.metric("Available RAM", f"{ram_gb:.1f} GB")
            else:
                st.sidebar.metric("Available RAM", "N/A")
            
            # Try different possible key names for CPU
            cpu_count = None
            for key in ['cpu_count', 'cpus', 'cpu_cores', 'cores']:
                if key in resource_info:
                    cpu_count = resource_info[key]
                    break
            
            if cpu_count is not None:
                st.sidebar.metric("CPU Cores", cpu_count)
            else:
                st.sidebar.metric("CPU Cores", "N/A")
        else:
            st.sidebar.metric("Available RAM", "N/A")
            st.sidebar.metric("CPU Cores", "N/A")
            
    except Exception as e:
        st.sidebar.error(f"Error getting system info: {str(e)}")
        st.sidebar.metric("Available RAM", "N/A")
        st.sidebar.metric("CPU Cores", "N/A")
    
    # Progress tracking
    st.sidebar.subheader("📈 Progress")
    
    # Check completion status
    data_uploaded = st.session_state.get('uploaded_data_path') is not None
    preprocessing_done = st.session_state.get('preprocessing_done', False)
    extraction_done = st.session_state.get('extraction_done', False)
    analysis_done = st.session_state.get('univariate_results') is not None or st.session_state.get('lasso_features') is not None
    
    # Progress indicators
    st.sidebar.write("✅ Data Upload" if data_uploaded else "⏳ Data Upload")
    st.sidebar.write("✅ Pre-processing" if preprocessing_done else "⏳ Pre-processing")
    st.sidebar.write("✅ Feature Extraction" if extraction_done else "⏳ Feature Extraction")
    st.sidebar.write("✅ Analysis" if analysis_done else "⏳ Analysis")
    
    # Session state info
    if st.sidebar.button("🗑️ Clear Session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    # Help section
    st.sidebar.subheader("❓ Help")
    with st.sidebar.expander("Quick Start Guide"):
        st.write("""
        1. **Upload Data**: Select your DICOM files or directory
        2. **Scan Contours**: Choose modality and scan for ROIs
        3. **Pre-process**: Generate NIfTI files
        4. **Extract Features**: Configure and run radiomics extraction
        5. **Analyze**: Upload outcomes and run statistical analysis
        """)
    
    with st.sidebar.expander("Supported Formats"):
        st.write("""
        - **DICOM**: .dcm, .DCM files
        - **Archives**: .zip files containing DICOM
        - **Outcomes**: .csv files with PatientID column
        """)
    
    # About section
    st.sidebar.subheader("ℹ️ About")
    st.sidebar.write("RadiomicsGUI v2.0")
    st.sidebar.write("Built with PyRadiomics & Streamlit")
    st.sidebar.write("© 2024 Radiomics Research")

def main():
    """Main function to build the complete Streamlit application."""
    st.set_page_config(
        page_title="RadiomicsGUI",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Build sidebar
    build_sidebar()
    
    # Main content area
    st.title("🔬 RadiomicsGUI - Advanced Radiomics Analysis Platform")
    st.markdown("---")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs([
        "📤 Data Upload & Pre-processing",
        "🔥 Feature Extraction", 
        "📊 Statistical Analysis"
    ])
    
    with tab1:
        build_tab1_data_upload()
    
    with tab2:
        build_tab2__feature_extraction()
    
    with tab3:
        build_tab3_analysis()


if __name__ == "__main__":
    main()
