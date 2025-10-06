"""
Enhanced UI module with comprehensive multi-modality, NIfTI, and IBSI support.
Updated: 2025-09-15
Features:
- Original streamlit UI components preserved
- Enhanced NIfTI file support alongside DICOM
- Multi-modality support (CT, MRI, PET/CT)
- Longitudinal data handling
- IBSI feature extraction option
- Enhanced progress tracking with ETA
- Multi-series processing capabilities
- Comprehensive error handling and validation
"""

import streamlit as st
import traceback
import time
import pandas as pd
import numpy as np
import yaml

# Enhanced imports with error handling
try:
    from processing import (
        validate_directory_path,
        get_available_directories,
        process_selected_path,
        organize_dicom_files,
        scan_uploaded_data_for_contours,
        preprocess_uploaded_data,
        scan_nifti_data_for_analysis,
        scan_uploaded_data_for_contours_enhanced,
        preprocess_uploaded_data_enhanced,
        preprocess_nifti_data,
        enhanced_modality_detection,
        get_supported_modalities
    )

    from extraction import (
        generate_pyradiomics_params,
        run_extraction,
        validate_extraction_parameters,
        get_feature_extraction_info,
        generate_pyradiomics_params_enhanced,
        run_extraction_with_ibsi_enhanced,
        get_ibsi_feature_mapping,
        get_missing_ibsi_features_list,
        get_enhanced_extraction_info
    )

    from analysis import (
        run_univariate_analysis,
        run_lasso_selection,
        generate_correlation_heatmap
    )

    from utils import (
        check_system_resources,
        categorize_contours,
        validate_uploaded_files,
        ProgressTracker,
        detect_file_type,
        organize_nifti_files,
        categorize_contours_extended,
        get_available_modalities_extended
    )

except ImportError as e:
    st.header("💥 Application Error")
    st.error("A critical module failed to import. The application cannot start.")
    st.error(f"The specific error is: **{e}")
    st.code(traceback.format_exc(), language='text')
    st.stop()

# --- ENHANCED DATA INPUT SECTION ---
def enhanced_data_input_section():
    """Enhanced data input with comprehensive NIfTI and DICOM support"""
    st.header("Step 1.1: Select Your Data Source & Format")

    # Format and workflow selection
    col1, col2 = st.columns(2)

    with col1:
        input_format = st.radio(
            "Data Format:",
            ["DICOM", "NIfTI"],
            horizontal=True,
            help="Choose your data format. DICOM supports RT-STRUCT, NIfTI requires separate mask files."
        )
        st.session_state['input_format'] = input_format.lower()

    with col2:
        if input_format == "DICOM":
            multi_series = st.checkbox(
                "Multi-Series Mode",
                value=False,
                help="Enable to process multiple imaging series (longitudinal/multi-modality studies)"
            )
            st.session_state['multi_series_mode'] = multi_series
        else:
            st.session_state['multi_series_mode'] = False

    # Format-specific information
    if input_format == "DICOM":
        if st.session_state['multi_series_mode']:
            st.info("📋 Multi-Series DICOM mode: Process multiple timepoints/modalities with RT-STRUCT")
        else:
            st.info("📋 Single-Series DICOM mode: Process one modality with RT-STRUCT")
    else:
        st.info("🧠 NIfTI mode: Requires separate image and mask files (PatientID_image.nii.gz, PatientID_mask.nii.gz)")

    # Input method selection
    input_method = st.radio(
        "Choose how to provide your data:",
        ["📤 Upload Files", "📁 Select from Available Directories", "✏️ Manual Path Entry"],
        horizontal=True,
        help="Choose the most convenient method to specify your data location."
    )

    uploaded_files = None
    selected_path = None

    if input_method == "📤 Upload Files":
        if input_format == "DICOM":
            file_types = ['dcm', 'zip', 'DCM', 'ZIP', 'ima', 'dicom']
            help_text = "Upload DICOM files or ZIP archives containing DICOM data"
        else:
            file_types = ['nii', 'nii.gz', 'zip']
            help_text = "Upload NIfTI files (.nii, .nii.gz) or ZIP archives. Use naming: PatientID_image.nii.gz, PatientID_mask.nii.gz"

        uploaded_files = st.file_uploader(
            f"Choose your {input_format} files",
            accept_multiple_files=True,
            type=file_types,
            help=help_text
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
            f"Enter the full path to your {input_format} data directory:",
            placeholder=f"/path/to/your/{input_format.lower()}/data",
        )

    if selected_path:
        validation_issues = validate_directory_path(selected_path)
        if validation_issues:
            for issue in validation_issues:
                st.error(f"• {issue}")
            selected_path = None
        else:
            st.success(f"✅ Valid {input_format} directory selected: {selected_path}")

    return uploaded_files, selected_path

# --- ENHANCED TAB 1 - DATA UPLOAD ---
def build_tab1_data_upload():
    """Enhanced Tab 1 with comprehensive NIfTI and multi-modality support"""
    uploaded_files, selected_path = enhanced_data_input_section()

    input_format = st.session_state.get('input_format', 'dicom')

    # Process button
    process_button_label = ""
    if uploaded_files:
        process_button_label = f"🔄 Process Uploaded {input_format.upper()} Files"
    elif selected_path:
        process_button_label = f"🔄 Process Selected {input_format.upper()} Directory"

    if process_button_label and st.button(process_button_label, type="primary", key="process_data"):
        data_path = None
        with st.spinner(f"Processing {input_format} data source..."):
            if input_format == 'dicom':
                if uploaded_files:
                    data_path = organize_dicom_files(uploaded_files)
                elif selected_path:
                    data_path = process_selected_path(selected_path)
            else:  # nifti
                if uploaded_files:
                    data_path = organize_nifti_files(uploaded_files)
                elif selected_path:
                    data_path = selected_path

        if data_path:
            st.session_state['uploaded_data_path'] = data_path
            st.success(f"✅ {input_format.upper()} data processed. Ready for analysis: `{data_path}`")
        else:
            st.error(f"❌ {input_format.upper()} data processing failed. Please check your files or path.")

    # Continue with format-specific processing
    if st.session_state.get('uploaded_data_path'):
        st.divider()

        if input_format == 'dicom':
            build_dicom_analysis_section()
        else:
            build_nifti_analysis_section()
def build_preprocessing_results_display(result_df, processing_summary):
    """Display comprehensive preprocessing results"""
    st.subheader("📊 Enhanced Preprocessing Results")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Patients", processing_summary.get('total_patients', len(result_df)))
    with col2:
        st.metric("Successful", len(result_df))
    with col3:
        failed_count = processing_summary.get('total_patients', len(result_df)) - len(result_df)
        st.metric("Failed", failed_count)
    with col4:
        success_rate = (len(result_df) / processing_summary.get('total_patients', len(result_df))) * 100 if processing_summary.get('total_patients', 0) > 0 else 100
        st.metric("Success Rate", f"{success_rate:.1f}%")

    # Recovery statistics (if available)
    if 'recovery_statistics' in processing_summary:
        st.subheader("🎯 Ultimate Recovery Statistics")
        recovery_stats = processing_summary['recovery_statistics']

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("RT-Utils Standard", recovery_stats.get('robust_rt_utils', 0))
            st.metric("Enhanced Methods", recovery_stats.get('robust_sitk_skimage', 0) + recovery_stats.get('robust_enhanced_coord_transform', 0))
        with col2:
            st.metric("Morphology Enhanced", recovery_stats.get('robust_morphology_enhanced', 0))
            st.metric("Direct DICOM", recovery_stats.get('robust_direct_dicom', 0))
        with col3:
            st.metric("Alternative ROI", recovery_stats.get('alternative_roi', 0))
            st.metric("Conversion Rescued", recovery_stats.get('conversion_rescued', 0))

    # Results tabs
    results_tabs = st.tabs(["✅ Successful Patients", "📊 Dataset Overview", "⚠️ Issues"])

    with results_tabs[0]:
        st.dataframe(result_df, use_container_width=True)

        # Download options
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Dataset Summary",
            csv,
            f"dataset_summary_{st.session_state.get('selected_modality', 'processed')}.csv",
            "text/csv",
            key='download-preprocessing-csv'
        )

    with results_tabs[1]:
        # Dataset statistics
        if not result_df.empty:
            st.subheader("📈 Dataset Statistics")

            # Modality distribution
            if 'modality' in result_df.columns:
                modality_counts = result_df['modality'].value_counts()
                st.bar_chart(modality_counts)

            # ROI statistics
            if 'voxel_count' in result_df.columns:
                st.subheader("ROI Size Distribution")
                voxel_stats = result_df['voxel_count'].describe()
                st.dataframe(voxel_stats.to_frame().T)

    with results_tabs[2]:
        # Failed patients and issues
        failed_patients = processing_summary.get('failed_patients', {})
        if failed_patients:
            st.subheader("❌ Failed Patients")
            for patient_id, failure_info in failed_patients.items():
                with st.expander(f"Patient: {patient_id}"):
                    st.error(f"**Reason:** {failure_info.get('reason', 'Unknown')}")
                    if failure_info.get('details'):
                        st.code(failure_info['details'], language='text')
        else:
            st.success("🎉 All patients processed successfully!")

def build_preprocessing_error_display(processing_summary):
    """Display preprocessing error information"""
    st.subheader("❌ Preprocessing Issues")

    failed_patients = processing_summary.get('failed_patients', {})
    if not failed_patients:
        st.info("No errors recorded")
        return

    # Initialize error_types dictionary
    error_types = {}

    for failure_info in failed_patients.values():
        reason = failure_info.get('reason', 'Unknown error')

        # Categorize errors
        if 'No labels found' in reason:
            error_type = 'Mask alignment issues'
        elif 'validation failed' in reason.lower():
            error_type = 'Mask validation issues'
        elif 'missing files' in reason.lower():
            error_type = 'File access issues'
        elif 'not found' in reason.lower():
            error_type = 'Resource not found'
        elif 'empty' in reason.lower():
            error_type = 'Empty data'
        elif 'corrupt' in reason.lower():
            error_type = 'Corrupt data'
        elif 'format' in reason.lower():
            error_type = 'Format issues'
        else:
            error_type = 'Other preprocessing errors'

        # Increment error type count
        error_types[error_type] = error_types.get(error_type, 0) + 1

    # Display error analysis
    st.subheader("📈 Error Analysis")
    col1, col2 = st.columns(2)
    with col1:
        for error_type, count in error_types.items():
            st.write(f"• {error_type}: {count} patient(s)")

    with col2:
        total_errors = sum(error_types.values())
        st.metric("Total Errors", total_errors)

        if total_errors > 0:
            top_error = max(error_types.items(), key=lambda x: x[1])
            st.metric("Most Common Error", f"{top_error[0]} ({top_error[1]} cases)")

    # Display detailed errors
    st.subheader("🔍 Detailed Errors")
    for patient_id, failure_info in failed_patients.items():
        with st.expander(f"Patient: {patient_id}"):
            st.error(f"**Error:** {failure_info.get('reason', 'Unknown')}")

            # Display additional details if available
            details_container = st.container()
            with details_container:
                if 'details' in failure_info:
                    st.code(failure_info['details'])

                if 'mask_info' in failure_info:
                    st.write("**Mask Information:**")
                    st.json(failure_info['mask_info'])

                if 'suggestions' in failure_info:
                    st.write("**Suggested Fixes:**")
                    for suggestion in failure_info['suggestions']:
                        st.write(f"- {suggestion}")

    # Recovery suggestions
    st.subheader("💡 Recovery Suggestions")
    st.write("""
    - **Mask alignment issues**: Check if mask and image have matching dimensions and spatial alignment.
    - **Mask validation issues**: Verify mask contains valid binary values (0 and 1).
    - **File access issues**: Ensure all files exist and are accessible.
    - **Resource not found**: Check file paths and permissions.
    - **Empty data**: Verify that images and masks contain valid data.
    """)



def build_dicom_analysis_section():
    """Enhanced DICOM analysis with comprehensive multi-modality support"""
    st.header("Step 1.2: DICOM Analysis - Enhanced Modality Selection & ROI Detection")

    # Enhanced modality selection
    col1, col2 = st.columns(2)

    with col1:
        available_modalities = get_supported_modalities()

        if st.session_state.get('multi_series_mode', False):
            st.subheader("Multi-Modality Selection")
            selected_modalities = st.multiselect(
                "Select modalities to process:",
                options=available_modalities,
                default=['CT'],
                help="Choose multiple modalities for comprehensive analysis"
            )
        else:
            st.subheader("Single Modality Selection")
            selected_modalities = [st.selectbox(
                "Select imaging modality:",
                options=available_modalities,
                index=0,
                help="Choose the primary imaging modality to analyze"
            )]

        st.session_state['selected_modalities'] = selected_modalities

    with col2:
        st.subheader("Scan Configuration")
        enable_longitudinal = st.checkbox(
            "Enable Longitudinal Analysis",
            value=st.session_state.get('multi_series_mode', False),
            disabled=not st.session_state.get('multi_series_mode', False),
            help="Process multiple timepoints for the same patient"
        )

        # Enhanced scanning button
        scan_button_text = "🔍 Enhanced Multi-Modality Scan" if len(selected_modalities) > 1 else f"🔍 Scan for {selected_modalities[0]} Data"

        if st.button(scan_button_text, type="primary"):
            with st.spinner(f"Enhanced scanning for {', '.join(selected_modalities)} series and ROIs..."):
                all_contours, pat_data, pat_status, available_mods, longitudinal_data = scan_uploaded_data_for_contours_enhanced(
                    st.session_state.uploaded_data_path,
                    selected_modalities,
                    st.session_state.get('multi_series_mode', False)
                )

                # Store enhanced results
                st.session_state.all_contours = all_contours
                st.session_state.patient_contour_data = pat_data
                st.session_state.patient_status = pat_status
                st.session_state.available_modalities = available_mods
                st.session_state.longitudinal_data = longitudinal_data

                # Display comprehensive results
                if available_mods:
                    st.info(f"📊 Available modalities in your data: {', '.join(sorted(available_mods))}")

                    # Check modality availability
                    missing_modalities = [mod for mod in selected_modalities if mod not in available_mods]
                    if missing_modalities:
                        st.warning(f"⚠️ Requested modalities not found: {', '.join(missing_modalities)}")

    # Enhanced results display
    if st.session_state.get('all_contours'):
        st.success(f"🎯 Enhanced scan complete! Found {len(st.session_state.all_contours)} unique contours across {len(selected_modalities)} modalities.")

        # Comprehensive results summary
        with st.expander("📋 Detailed Scan Results", expanded=True):
            summary_tabs = st.tabs(["📊 Overview", "🏥 Patient Status", "📈 Longitudinal Data"])

            with summary_tabs[0]:
                # Overview metrics
                success_count = sum(1 for status in st.session_state.patient_status.values() if status['status'] == 'success')
                total_patients = len(st.session_state.patient_status)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Patients", total_patients)
                with col2:
                    st.metric("Successfully Processed", success_count)
                with col3:
                    st.metric("Available Modalities", len(st.session_state.available_modalities))
                with col4:
                    st.metric("Unique Contours", len(st.session_state.all_contours))

            with summary_tabs[1]:
                # Patient-by-patient status
                for patient_id, status_info in st.session_state.patient_status.items():
                    if status_info['status'] == 'success':
                        st.success(f"✅ {patient_id}: {len(status_info.get('contours', []))} contours found")
                    else:
                        st.error(f"❌ {patient_id}: {', '.join(status_info.get('issues', ['Unknown error']))}")

            with summary_tabs[2]:
                # Longitudinal data overview
                if st.session_state.get('longitudinal_data'):
                    for patient_id, patient_longitudinal in st.session_state.longitudinal_data.items():
                        with st.container():
                            st.subheader(f"Patient: {patient_id}")
                            compatible_pairs = patient_longitudinal.get('compatible_pairs', [])
                            if compatible_pairs:
                                pairs_df = pd.DataFrame([{
                                    'Modality': pair['modality'],
                                    'Timepoint': pair['timepoint'],
                                    'Series Description': pair['series_description'],
                                    'Study Date': pair['study_date'],
                                    'Slice Count': pair['slice_count'],
                                    'Contours': len(pair['contours'])
                                } for pair in compatible_pairs])
                                st.dataframe(pairs_df, use_container_width=True)
                            else:
                                st.warning(f"No compatible pairs for {patient_id}")

        # Enhanced ROI selection
        build_enhanced_roi_selection()

def build_nifti_analysis_section():
    """NIfTI-specific analysis section"""
    st.header("Step 1.2: NIfTI Analysis - Image/Mask Pair Detection")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("NIfTI Scan Configuration")
        auto_pair = st.checkbox(
            "Auto-pair images and masks",
            value=True,
            help="Automatically match image files with corresponding mask files based on filename similarity"
        )

        strict_validation = st.checkbox(
            "Strict validation",
            value=True,
            help="Enable comprehensive validation of image/mask pairs (dimensions, spacing, etc.)"
        )

    with col2:
        if st.button("🔍 Scan NIfTI Data", type="primary"):
            with st.spinner("Scanning NIfTI data for image/mask pairs..."):
                patient_data, processing_summary = scan_nifti_data_for_analysis(
                    st.session_state.uploaded_data_path
                )

                # Store results
                st.session_state.nifti_patient_data = patient_data
                st.session_state.nifti_processing_summary = processing_summary

                # Display results
                if processing_summary['valid_pairs'] > 0:
                    st.success(f"✅ Found {processing_summary['valid_pairs']} valid image/mask pairs!")
                else:
                    st.warning("⚠️ No valid image/mask pairs found.")

    # Display NIfTI results
    if st.session_state.get('nifti_patient_data'):
        build_nifti_results_display()

def build_nifti_results_display():
    """Display NIfTI scan results"""
    st.subheader("📊 NIfTI Scan Results")

    patient_data = st.session_state.nifti_patient_data
    processing_summary = st.session_state.nifti_processing_summary

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Patients", processing_summary['total_patients'])
    with col2:
        st.metric("Valid Pairs", processing_summary['valid_pairs'])
    with col3:
        success_rate = (processing_summary['valid_pairs'] / max(processing_summary['total_patients'], 1)) * 100
        st.metric("Success Rate", f"{success_rate:.1f}%")

    # Patient details
    with st.expander("📋 Patient Details", expanded=True):
        for patient_id, patient_info in patient_data.items():
            if patient_info['status'] == 'success':
                st.success(f"✅ {patient_id}: {len(patient_info['pairs'])} valid pair(s)")
                for i, pair in enumerate(patient_info['pairs']):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"Image: {pair['image']['filename']}")
                    with col2:
                        st.write(f"Mask: {pair['mask']['filename']}")
                    with col3:
                        st.write(f"Modality: {pair['modality']}")
            else:
                st.error(f"❌ {patient_id}: {patient_info.get('error', 'Unknown error')}")

    # NIfTI preprocessing section
    build_nifti_preprocessing_section()

def build_nifti_preprocessing_section():
    """NIfTI preprocessing section"""
    st.divider()
    st.header("Step 1.3: NIfTI Preprocessing")

    patient_data = st.session_state.nifti_patient_data

    # Collect all valid pairs for processing
    valid_pairs = []
    for patient_id, patient_info in patient_data.items():
        if patient_info['status'] == 'success':
            for pair in patient_info['pairs']:
                valid_pairs.append({
                    'patient_id': patient_id,
                    'image_path': pair['image']['path'],
                    'mask_path': pair['mask']['path'],
                    'modality': pair['modality'],
                    'roi_name': 'ROI'  # Default ROI name for NIfTI
                })

    if valid_pairs:
        st.info(f"Ready to process {len(valid_pairs)} image/mask pairs")

        if st.button("🚀 Start NIfTI Preprocessing", type="primary"):
            # Setup progress UI
            progress_container = st.container()
            status_container = st.container()
            with progress_container:
                progress_bar = st.progress(0)
                progress_text = st.empty()
            with status_container:
                status_placeholder = st.empty()

            # Store UI elements in session state
            st.session_state['ui_progress_bar'] = progress_bar
            st.session_state['ui_progress_text'] = progress_text
            st.session_state['ui_status_placeholder'] = status_placeholder

            # Process NIfTI data
            result_df, processing_summary = preprocess_nifti_data(
                st.session_state.uploaded_data_path,
                valid_pairs
            )

            if not result_df.empty:
                st.success(f"✅ Successfully processed {len(result_df)} NIfTI pairs!")
                st.session_state.dataset_df = result_df
                st.session_state.preprocessing_done = True

                # Display results
                st.subheader("📊 Preprocessing Results")
                st.dataframe(result_df)

                # Download option
                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download NIfTI Dataset Summary",
                    csv,
                    "nifti_dataset_summary.csv",
                    "text/csv",
                    key='download-nifti-csv'
                )
            else:
                st.error("❌ NIfTI preprocessing failed. Please check your data.")
def build_enhanced_roi_selection():
    """Enhanced ROI selection with categorization and search"""
    st.divider()
    st.header("Step 1.3: Enhanced ROI Selection & Preprocessing")

    if not st.session_state.get('all_contours'):
        st.warning("⚠️ Please scan for contours first.")
        return

    # Enhanced ROI categorization and selection
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ROI Categories")
        targets, oars, other = categorize_contours_extended(st.session_state.all_contours)

        # Display categorized ROIs
        roi_category = st.selectbox(
            "Browse by category:",
            ["All ROIs", "Target Volumes", "Organs at Risk", "Other Structures"],
            help="ROIs are automatically categorized based on common naming conventions"
        )

        if roi_category == "Target Volumes":
            available_rois = targets
        elif roi_category == "Organs at Risk":
            available_rois = oars
        elif roi_category == "Other Structures":
            available_rois = other
        else:
            available_rois = sorted(st.session_state.all_contours)

        # ROI search and selection
        search_roi = st.text_input(
            "🔍 Search ROIs:",
            placeholder="Type to search...",
            help="Filter ROIs by name (case-insensitive)"
        )

        if search_roi:
            filtered_rois = [roi for roi in available_rois if search_roi.lower() in roi.lower()]
        else:
            filtered_rois = available_rois

        selected_roi = st.selectbox(
            "Select target ROI for extraction:",
            options=[""] + filtered_rois,
            help="Choose the ROI you want to extract features from"
        )

    with col2:
        st.subheader("ROI Statistics")

        # Display ROI availability statistics
        if selected_roi:
            st.info(f"Selected ROI: **{selected_roi}")

            # Calculate availability across patients
            patients_with_roi = []
            for patient_id, contours in st.session_state.patient_contour_data.items():
                if any(selected_roi.lower() in contour.lower() for contour in contours):
                    patients_with_roi.append(patient_id)

            col2_1, col2_2 = st.columns(2)
            with col2_1:
                st.metric("Patients with ROI", len(patients_with_roi))
            with col2_2:
                total_patients = len(st.session_state.patient_contour_data)
                availability_pct = (len(patients_with_roi) / max(total_patients, 1)) * 100
                st.metric("Availability", f"{availability_pct:.1f}%")

            if patients_with_roi:
                with st.expander("📋 Patients with this ROI"):
                    st.write(", ".join(patients_with_roi))

        # Category statistics
        st.subheader("ROI Categories")
        col2_1, col2_2, col2_3 = st.columns(3)
        with col2_1:
            st.metric("Targets", len(targets))
        with col2_2:
            st.metric("OARs", len(oars))
        with col2_3:
            st.metric("Other", len(other))

    # Preprocessing section
    if selected_roi:
        st.divider()
        st.subheader("🚀 Start Preprocessing")

        # Multi-series options
        if st.session_state.get('multi_series_mode', False):
            st.info("🔄 Multi-series mode enabled - will process all selected modalities")

            # Show selected series for processing
            if st.session_state.get('longitudinal_data'):
                series_count = 0
                for patient_longitudinal in st.session_state.longitudinal_data.values():
                    series_count += len(patient_longitudinal.get('compatible_pairs', []))
                st.info(f"📊 Ready to process {series_count} series across {len(st.session_state.longitudinal_data)} patients")

        preprocessing_col1, preprocessing_col2 = st.columns(2)

        with preprocessing_col1:
            # Modality-specific settings
            selected_modalities = st.session_state.get('selected_modalities', ['CT'])
            primary_modality = selected_modalities[0] if selected_modalities else 'CT'

            st.write(f"**Primary Modality:** {primary_modality}")
            st.write(f"**Target ROI:** {selected_roi}")

            if len(selected_modalities) > 1:
                st.write(f"**Additional Modalities:** {', '.join(selected_modalities[1:])}")

        with preprocessing_col2:
            # Enhanced preprocessing options
            robust_processing = st.checkbox(
                "Enable Robust Processing",
                value=True,
                help="Use multiple mask generation methods for maximum success rate"
            )

            validate_results = st.checkbox(
                "Validate Results",
                value=True,
                help="Perform comprehensive validation of generated masks and images"
            )

        # Start preprocessing
        if st.button("🚀 Start Enhanced Preprocessing", type="primary"):
            # Setup progress UI
            progress_container = st.container()
            status_container = st.container()
            with progress_container:
                progress_bar = st.progress(0)
                progress_text = st.empty()
            with status_container:
                status_placeholder = st.empty()

            # Store UI elements in session state
            st.session_state['ui_progress_bar'] = progress_bar
            st.session_state['ui_progress_text'] = progress_text
            st.session_state['ui_status_placeholder'] = status_placeholder

            # Run enhanced preprocessing
            if st.session_state.get('multi_series_mode', False):
                # Multi-series preprocessing
                result_df, processing_summary = preprocess_uploaded_data_enhanced(
                    st.session_state.uploaded_data_path,
                    selected_roi,
                    selected_modalities,
                    multi_series_mode=True,
                    selected_series=[]  # This would be populated from longitudinal_data
                )
            else:
                # Single-series preprocessing
                result_df, processing_summary = preprocess_uploaded_data(
                    st.session_state.uploaded_data_path,
                    selected_roi,
                    primary_modality
                )

            # Store processing summary
            st.session_state['processing_summary'] = processing_summary

            if not result_df.empty:
                st.success(f"✅ Successfully preprocessed {len(result_df)} patients!")
                st.session_state.dataset_df = result_df
                st.session_state.preprocessing_done = True

                # Enhanced results display
                build_preprocessing_results_display(result_df, processing_summary)
            else:
                st.error("❌ Preprocessing failed. Please check your data and try again.")
                build_preprocessing_error_display(processing_summary)

# --- ENHANCED TAB 2 - FEATURE EXTRACTION ---
def build_tab2_feature_extraction():
    """Enhanced Tab 2 with IBSI support and multi-modality optimization"""
    if not st.session_state.get('preprocessing_done', False):
        st.warning("⚠️ Please complete the data upload and preprocessing steps first.")
        return

    st.header("Step 2: Enhanced Radiomics Feature Extraction")

    # Enhanced configuration section
    build_enhanced_extraction_configuration()

    # Extraction execution section
    build_extraction_execution_section()

def build_enhanced_extraction_configuration():
    """Enhanced extraction configuration with IBSI and modality support"""
    st.subheader("🔧 Enhanced PyRadiomics Configuration")

    # Feature extraction tabs
    config_tabs = st.tabs(["⚙️ Basic Settings", "🎯 IBSI Features", "🔬 Advanced Settings"])

    with config_tabs[0]:
        # Basic feature class selection
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Feature Classes to Extract:**")
            feature_classes = {}
            feature_classes['firstorder'] = st.checkbox("First Order Statistics", value=True, help="Statistical features like mean, variance, skewness")
            feature_classes['shape'] = st.checkbox("Shape Features", value=True, help="Morphological features like volume, surface area")
            feature_classes['glcm'] = st.checkbox("GLCM Features", value=True, help="Grey Level Co-occurrence Matrix features")
            feature_classes['glrlm'] = st.checkbox("GLRLM Features", value=True, help="Grey Level Run Length Matrix features")

        with col2:
            st.write("**Additional Feature Classes:**")
            feature_classes['glszm'] = st.checkbox("GLSZM Features", value=True, help="Grey Level Size Zone Matrix features")
            feature_classes['ngtdm'] = st.checkbox("NGTDM Features", value=True, help="Neighbouring Grey Tone Difference Matrix features")
            feature_classes['gldm'] = st.checkbox("GLDM Features", value=True, help="Grey Level Dependence Matrix features")

        # Store feature classes in session state
        st.session_state['feature_classes'] = feature_classes

    with config_tabs[1]:
        # IBSI-specific settings
        st.write("**IBSI Compliance Settings:**")

        col1, col2 = st.columns(2)

        with col1:
            enable_ibsi_features = st.checkbox(
                "Enable Additional IBSI Features",
                value=False,
                help="Extract features from IBSI standard that are not included in PyRadiomics"
            )

            use_ibsi_nomenclature = st.checkbox(
                "Use IBSI Nomenclature",
                value=True,
                help="Convert feature names to IBSI standard nomenclature"
            )

            st.session_state['ibsi_features_enabled'] = enable_ibsi_features
            st.session_state['use_ibsi_nomenclature'] = use_ibsi_nomenclature

        with col2:
            if enable_ibsi_features:
                st.info("🎯 Additional IBSI features will be calculated")
                ibsi_info = get_missing_ibsi_features_list()
                with st.expander("📋 Additional IBSI Features"):
                    for feature_id, description in ibsi_info.items():
                        st.write(f"• **{feature_id}**: {description}")

            if use_ibsi_nomenclature:
                st.info("📝 Feature names will use IBSI standard nomenclature")
                with st.expander("🔗 PyRadiomics → IBSI Mapping Example"):
                    mapping_sample = {
                        'shape_VoxelVolume': 'morph_Volume_voxel',
                        'firstorder_Mean': 'stat_Mean',
                        'glcm_Contrast': 'glcm_Contrast'
                    }
                    for pyrad, ibsi in mapping_sample.items():
                        st.write(f"• `{pyrad}` → `{ibsi}`")

    with config_tabs[2]:
        # Advanced settings with modality optimization
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Preprocessing Settings:**")
            normalize_image = st.checkbox("Normalize Image", value=True, help="Normalize image intensities")
            resample_pixel_spacing = st.checkbox("Resample Pixel Spacing", value=False, help="Resample to isotropic voxels")

            # Modality-specific bin width
            dataset_df = st.session_state.get('dataset_df', pd.DataFrame())
            if not dataset_df.empty and 'modality' in dataset_df.columns:
                primary_modality = dataset_df['modality'].iloc[0]
            else:
                primary_modality = st.session_state.get('selected_modality', 'CT')

            # Default bin width based on modality
            if primary_modality.startswith('MR'):
                default_bin_width = 5
            elif primary_modality.startswith('PT'):
                default_bin_width = 0.1
            else:
                default_bin_width = 25

            bin_width = st.number_input(
                f"Bin Width (optimized for {primary_modality})",
                min_value=0.1,
                max_value=100.0,
                value=float(default_bin_width),
                help="Discretization bin width - automatically optimized based on modality"
            )

            if resample_pixel_spacing:
                pixel_spacing = st.number_input("Pixel Spacing (mm)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
            else:
                pixel_spacing = None

        with col2:
            st.write("**Advanced Parameters:**")
            interpolator = st.selectbox(
                "Interpolator",
                ["sitkBSpline", "sitkLinear", "sitkNearestNeighbor"],
                index=0,
                help="Interpolation method for resampling"
            )
            pad_distance = st.number_input("Pad Distance", min_value=0, max_value=20, value=5, help="Padding around ROI")
            geometryTolerance = st.number_input(
                "Geometry Tolerance",
                min_value=0.0,
                max_value=1.0,
                value=0.0001,
                format="%.6f",
                help="Tolerance for geometry calculations"
            )

        # Store advanced settings
        st.session_state['extraction_settings'] = {
            'normalize_image': normalize_image,
            'resample_pixel_spacing': resample_pixel_spacing,
            'pixel_spacing': pixel_spacing,
            'bin_width': bin_width,
            'interpolator': interpolator,
            'pad_distance': pad_distance,
            'geometryTolerance': geometryTolerance,
            'modality': primary_modality
        }

    # Generate parameters button
    if st.button("🔄 Generate Enhanced PyRadiomics Parameters", type="secondary"):
        with st.spinner("Generating optimized parameter configuration..."):
            try:
                settings = st.session_state.get('extraction_settings', {})
                feature_classes = st.session_state.get('feature_classes', {})

                params = generate_pyradiomics_params_enhanced(
                    feature_classes=feature_classes,
                    normalize_image=settings.get('normalize_image', True),
                    resample_pixel_spacing=settings.get('resample_pixel_spacing', False),
                    pixel_spacing=settings.get('pixel_spacing'),
                    bin_width=settings.get('bin_width', 25),
                    interpolator=settings.get('interpolator', 'sitkBSpline'),
                    pad_distance=settings.get('pad_distance', 5),
                    geometryTolerance=settings.get('geometryTolerance', 0.0001),
                    modality=settings.get('modality', 'CT')
                )

                st.session_state.pyradiomics_params = params
                st.success("✅ Enhanced PyRadiomics parameters generated successfully!")

                with st.expander("📋 Generated Parameters"):
                    st.code(yaml.dump(params, default_flow_style=False), language='yaml')

            except Exception as e:
                st.error(f"Error generating parameters: {str(e)}")
                # Fallback to basic parameter generation
                params = generate_pyradiomics_params(
                    feature_classes=feature_classes,
                    normalize_image=settings.get('normalize_image', True),
                    resample_pixel_spacing=settings.get('resample_pixel_spacing', False),
                    pixel_spacing=settings.get('pixel_spacing'),
                    bin_width=settings.get('bin_width', 25),
                    interpolator=settings.get('interpolator', 'sitkBSpline'),
                    pad_distance=settings.get('pad_distance', 5),
                    geometryTolerance=settings.get('geometryTolerance', 0.0001)
                )
                st.session_state.pyradiomics_params = params
                st.success("✅ Basic PyRadiomics parameters generated successfully!")

def build_extraction_execution_section():
    """Enhanced extraction execution with comprehensive progress tracking"""
    if not st.session_state.get('pyradiomics_params'):
        st.warning("⚠️ Please generate PyRadiomics parameters first.")
        return

    st.divider()
    st.subheader("🚀 Feature Extraction")

    # System resources and settings
    col1, col2 = st.columns(2)

    with col1:
        try:
            resource_info = check_system_resources()
            st.subheader("💻 System Resources")
            ram_gb = resource_info.get('available_ram_gb', 0)
            cpu_count = resource_info.get('cpu_count', 1)

            col1_1, col1_2 = st.columns(2)
            with col1_1:
                st.metric("Available RAM", f"{ram_gb:.1f} GB")
            with col1_2:
                st.metric("CPU Cores", cpu_count)
        except Exception as e:
            st.warning(f"Could not get system resources: {str(e)}")
            cpu_count = 1

    with col2:
        st.subheader("⚙️ Processing Settings")
        dataset_df = st.session_state.get('dataset_df', pd.DataFrame())
        total_patients = len(dataset_df)
        st.metric("Patients to Process", total_patients)

        use_parallel = st.checkbox(
            "Enable Parallel Processing",
            value=False,
            help="Use multiple CPU cores (experimental)"
        )

        if use_parallel and cpu_count > 1:
            n_jobs = st.slider(
                "Number of parallel jobs:",
                min_value=1,
                max_value=min(cpu_count, total_patients),
                value=min(2, cpu_count)
            )
        else:
            n_jobs = 1

    # IBSI compliance settings
    st.subheader("🏅 IBSI Compliance Settings")
    col1, col2 = st.columns(2)
    with col1:
        enable_ibsi_compliance = st.checkbox(
            "Enable IBSI Compliance",
            value=True,
            help="Apply IBSI standard feature naming and additional features"
        )
    with col2:
        st.info("IBSI compliance ensures standardized feature naming and additional features")

    # Start extraction
    if st.button("🔥 Start Feature Extraction", type="primary"):
        # Setup progress UI
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Store IBSI settings in session state
            st.session_state['ibsi_compliance_enabled'] = enable_ibsi_compliance

            # Call the extraction function with the correct parameters
            features_df = run_extraction_with_ibsi_enhanced(
                dataset_df=st.session_state.get('dataset_df'),
                params=st.session_state.get('pyradiomics_params'),
                n_jobs=n_jobs,
                enable_ibsi_compliance=enable_ibsi_compliance
            )

            if features_df is not None and not features_df.empty:
                st.success(f"✅ Successfully extracted features from {len(features_df)} patients!")
                st.session_state.features_df = features_df
                st.session_state.extraction_done = True

                # Show basic results
                st.subheader("📊 Extraction Results")
                st.write(f"Total features extracted: {features_df.shape[1]-1}")
                st.write(f"Patients processed: {len(features_df)}")

                # Show sample data
                st.dataframe(features_df.head())

                # Download option
                csv = features_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Features",
                    csv,
                    "radiomics_features.csv",
                    "text/csv"
                )
            else:
                st.error("❌ Feature extraction failed. Please check your data and parameters.")

        except Exception as e:
            st.error(f"❌ Extraction error: {str(e)}")
            with st.expander("Error Details"):
                st.code(traceback.format_exc())
        finally:
            progress_bar.empty()
            status_text.empty()


# --- ENHANCED TAB 3 - ANALYSIS ---
def build_tab3_analysis():
    """Enhanced Tab 3 with comprehensive statistical analysis"""
    if not st.session_state.get('extraction_done', False):
        st.warning("⚠️ Please complete the feature extraction step first.")
        return

    st.header("Step 3: Enhanced Statistical Analysis & Feature Selection")

    # Enhanced outcome data section
    build_enhanced_outcome_section()

    # Statistical analysis section
    if st.session_state.get('outcome_df') is not None:
        build_enhanced_statistical_analysis()

def build_enhanced_outcome_section():
    """Enhanced outcome data handling"""
    st.subheader("📊 Outcome Data Management")

    outcome_method = st.radio(
        "How would you like to provide outcome data?",
        ["📤 Upload CSV File", "✏️ Manual Entry", "🔗 Use Existing Data"],
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

                # Validate outcome data
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

                        # Display outcome summary
                        with st.expander("📋 Outcome Data Summary"):
                            st.dataframe(outcome_df.head())

                            # Outcome variable analysis
                            outcome_cols = [col for col in outcome_df.columns if col != 'PatientID']
                            for col in outcome_cols:
                                if outcome_df[col].dtype in ['int64', 'float64']:
                                    st.write(f"**{col}**: Numeric variable (mean: {outcome_df[col].mean():.2f})")
                                else:
                                    unique_vals = outcome_df[col].nunique()
                                    st.write(f"**{col}**: Categorical variable ({unique_vals} unique values)")

            except Exception as e:
                st.error(f"❌ Error loading outcome data: {str(e)}")

    elif outcome_method == "✏️ Manual Entry":
        st.write("**Manual Outcome Entry:**")

        patients = st.session_state.features_df['PatientID'].tolist()

        # Outcome type selection
        col1, col2 = st.columns(2)

        with col1:
            outcome_type = st.selectbox(
                "Select outcome type:",
                ["Binary (0/1)", "Continuous", "Categorical", "Survival Time"]
            )

        with col2:
            outcome_name = st.text_input(
                "Outcome variable name:",
                value="Outcome",
                help="Name for your outcome variable"
            )

        if outcome_type == "Binary (0/1)":
            st.write("Enter binary outcomes (0 or 1) for each patient:")
            outcomes = {}

            # Organized layout for many patients
            n_cols = 3
            cols = st.columns(n_cols)

            for i, patient in enumerate(patients):
                with cols[i % n_cols]:
                    outcomes[patient] = st.selectbox(
                        f"{patient}:",
                        [0, 1],
                        key=f"outcome_{patient}"
                    )

        elif outcome_type == "Continuous":
            st.write("Enter continuous values for each patient:")
            outcomes = {}

            for patient in patients:
                outcomes[patient] = st.number_input(
                    f"{patient}:",
                    value=0.0,
                    key=f"outcome_{patient}"
                )

        elif outcome_type == "Categorical":
            # Define categories first
            categories = st.text_input(
                "Enter categories (comma-separated):",
                value="Low,Medium,High",
                help="E.g., Low,Medium,High or Grade1,Grade2,Grade3"
            ).split(',')
            categories = [cat.strip() for cat in categories if cat.strip()]

            if categories:
                outcomes = {}
                for patient in patients:
                    outcomes[patient] = st.selectbox(
                        f"{patient}:",
                        categories,
                        key=f"outcome_{patient}"
                    )

        elif outcome_type == "Survival Time":
            st.write("Enter survival time and event status:")
            outcomes = {}
            events = {}

            for patient in patients:
                col1, col2 = st.columns(2)
                with col1:
                    outcomes[patient] = st.number_input(
                        f"{patient} - Time:",
                        min_value=0.0,
                        value=12.0,
                        key=f"time_{patient}"
                    )
                with col2:
                    events[patient] = st.selectbox(
                        f"{patient} - Event:",
                        [0, 1],
                        key=f"event_{patient}",
                        help="0=Censored, 1=Event occurred"
                    )

        # Create outcome dataset
        if st.button("✅ Create Outcome Dataset"):
            if outcome_type == "Survival Time":
                outcome_df = pd.DataFrame({
                    'PatientID': patients,
                    f'{outcome_name}_Time': [outcomes[p] for p in patients],
                    f'{outcome_name}_Event': [events[p] for p in patients]
                })
            else:
                outcome_df = pd.DataFrame({
                    'PatientID': patients,
                    outcome_name: [outcomes[p] for p in patients]
                })

            st.success("✅ Outcome data created!")
            st.dataframe(outcome_df)

    elif outcome_method == "🔗 Use Existing Data":
        # Check if outcome data already exists in features
        if 'clinical_df' in st.session_state and st.session_state.clinical_df is not None:
            outcome_df = st.session_state.clinical_df
            st.success("✅ Using existing clinical data")
            st.dataframe(outcome_df.head())
        else:
            st.info("ℹ️ No existing outcome data found. Please use another method.")

    if outcome_df is not None:
        st.session_state.outcome_df = outcome_df

def build_enhanced_statistical_analysis():
    """Enhanced statistical analysis with multiple methods"""
    st.divider()
    st.subheader("🔬 Enhanced Statistical Analysis")

    # Merge features with outcomes
    merged_df = pd.merge(st.session_state.features_df, st.session_state.outcome_df, on='PatientID')
    st.session_state.merged_df = merged_df

    # Outcome variable selection
    outcome_columns = [col for col in st.session_state.outcome_df.columns if col != 'PatientID']
    selected_outcome = st.selectbox(
        "Select outcome variable for analysis:",
        outcome_columns,
        help="Choose the outcome variable you want to analyze"
    )

    if selected_outcome:
        st.session_state.selected_outcome = selected_outcome

        # Analysis method selection
        analysis_tabs = st.tabs([
            "📈 Univariate Analysis",
            "🎯 LASSO Selection",
            "🔗 Correlation Analysis",
            "🧪 Advanced Analysis"
        ])

        with analysis_tabs[0]:
            build_univariate_analysis_section(merged_df, selected_outcome)

        with analysis_tabs[1]:
            build_lasso_analysis_section(merged_df, selected_outcome)

        with analysis_tabs[2]:
            build_correlation_analysis_section(merged_df, selected_outcome)

        with analysis_tabs[3]:
            build_advanced_analysis_section(merged_df, selected_outcome)

def build_univariate_analysis_section(merged_df, selected_outcome):
    """Enhanced univariate analysis with proper data type handling"""
    st.subheader("📈 Univariate Analysis")

    # Check if the selected outcome is numeric
    if not pd.api.types.is_numeric_dtype(merged_df[selected_outcome]):
        st.warning(f"⚠️ The selected outcome '{selected_outcome}' is not numeric. "
                   f"Attempting to convert categorical values to numeric...")

        # Try to convert categorical outcome to numeric
        try:
            unique_values = merged_df[selected_outcome].unique()
            if len(unique_values) <= 10:  # Likely categorical
                merged_df = merged_df.copy()
                merged_df[selected_outcome] = pd.factorize(merged_df[selected_outcome])[0]
                st.success(f"✅ Successfully converted '{selected_outcome}' to numeric values.")
            else:
                st.error(f"❌ Cannot convert '{selected_outcome}' to numeric. Please select a numeric outcome.")
                return
        except Exception as e:
            st.error(f"❌ Error converting '{selected_outcome}' to numeric: {str(e)}")
            return

    col1, col2 = st.columns(2)
    with col1:
        p_threshold = st.number_input(
            "P-value threshold:",
            min_value=0.001,
            max_value=0.2,
            value=0.05,
            step=0.001,
            help="Features with p-values below this threshold are considered significant"
        )

    with col2:
        top_n_features = st.number_input(
            "Number of top features to display:",
            min_value=5,
            max_value=50,
            value=15,
            help="Number of most significant features to show in results"
        )

    if st.button("🔄 Run Univariate Analysis"):
        with st.spinner("Running univariate analysis..."):
            try:
                # Get feature columns (exclude metadata)
                feature_cols = [col for col in merged_df.columns
                              if col not in ['PatientID', selected_outcome, 'modality', 'timepoint',
                                            'series_uid', 'roi_voxel_count', 'roi_percentage', 'extraction_timestamp']]

                # Filter to only numeric features
                numeric_feature_cols = []
                for col in feature_cols:
                    if col in merged_df.columns and pd.api.types.is_numeric_dtype(merged_df[col]):
                        numeric_feature_cols.append(col)

                if not numeric_feature_cols:
                    st.error("❌ No numeric features found for analysis. Please check your data.")
                    return

                # Call the analysis function
                top_features_df, fig = run_univariate_analysis(
                    merged_df,
                    numeric_feature_cols,
                    selected_outcome
                )

                # Display the results
                st.subheader("📊 Top Correlated Features")
                st.dataframe(top_features_df)

                # Display the figure
                st.pyplot(fig)

                # Store results in session state
                st.session_state.univariate_results = top_features_df

                # Download option
                csv = top_features_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Univariate Results",
                    csv,
                    f"univariate_results_{selected_outcome}.csv",
                    "text/csv",
                    key='download-univariate'
                )

            except Exception as e:
                st.error(f"❌ Univariate analysis failed: {str(e)}")
                with st.expander("🔍 Error Details"):
                    st.code(traceback.format_exc())


def build_lasso_analysis_section(merged_df, selected_outcome):
    """Enhanced LASSO feature selection with proper data validation"""
    st.subheader("🎯 LASSO Feature Selection")

    # Check if the selected outcome is numeric
    if not pd.api.types.is_numeric_dtype(merged_df[selected_outcome]):
        st.warning(f"⚠️ The selected outcome '{selected_outcome}' is not numeric. "
                   f"Attempting to convert categorical values to numeric...")

        # Try to convert categorical outcome to numeric
        try:
            unique_values = merged_df[selected_outcome].unique()
            if len(unique_values) <= 10:  # Likely categorical
                merged_df = merged_df.copy()
                merged_df[selected_outcome] = pd.factorize(merged_df[selected_outcome])[0]
                st.success(f"✅ Successfully converted '{selected_outcome}' to numeric values.")
            else:
                st.error(f"❌ Cannot convert '{selected_outcome}' to numeric. Please select a numeric outcome.")
                return
        except Exception as e:
            st.error(f"❌ Error converting '{selected_outcome}' to numeric: {str(e)}")
            return

    col1, col2 = st.columns(2)
    with col1:
        alpha_mode = st.selectbox(
            "Alpha selection mode:",
            ["Auto (Cross-validation)", "Manual"],
            help="Choose how to set the regularization strength"
        )

        cv_folds = st.number_input(
            "Cross-validation folds:",
            min_value=3,
            max_value=10,
            value=5,
            help="Number of folds for cross-validation"
        )

    with col2:
        max_features = st.number_input(
            "Maximum features to select:",
            min_value=1,
            max_value=50,
            value=10,
            help="Maximum number of features to select"
        )

    if st.button("🔄 Run LASSO Selection"):
        with st.spinner("Running LASSO feature selection..."):
            try:
                # Get feature columns (exclude metadata)
                feature_cols = [col for col in merged_df.columns
                              if col not in ['PatientID', selected_outcome, 'modality', 'timepoint',
                                            'series_uid', 'roi_voxel_count', 'roi_percentage', 'extraction_timestamp']]

                # Filter to only numeric features
                numeric_feature_cols = []
                for col in feature_cols:
                    if col in merged_df.columns and pd.api.types.is_numeric_dtype(merged_df[col]):
                        numeric_feature_cols.append(col)

                if not numeric_feature_cols:
                    st.error("❌ No numeric features found for analysis. Please check your data.")
                    return

                st.info(f"📊 Using {len(numeric_feature_cols)} numeric features for LASSO selection")

                # Call the analysis function
                selected_features, fig = run_lasso_selection(
                    merged_df,
                    numeric_feature_cols,
                    selected_outcome
                )

                if selected_features:
                    st.success(f"✅ LASSO selected {len(selected_features)} features")
                    st.session_state.lasso_features = selected_features

                    # Display selected features
                    st.subheader("🎯 Selected Features")

                    # Create features dataframe with coefficients
                    features_df = pd.DataFrame([
                        {'Feature': feature, 'Coefficient': coef}
                        for feature, coef in selected_features.items()
                    ]).sort_values('Coefficient', key=abs, ascending=False)

                    st.dataframe(features_df, use_container_width=True)

                    # Display the figure
                    if fig:
                        st.pyplot(fig)

                    # Download option
                    csv = features_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download LASSO Results",
                        csv,
                        f"lasso_features_{selected_outcome}.csv",
                        "text/csv",
                        key='download-lasso'
                    )
                else:
                    st.warning("⚠️ LASSO did not select any features with current parameters")
                    st.info("💡 Try adjusting the parameters or check your data")

            except Exception as e:
                st.error(f"❌ LASSO selection failed: {str(e)}")
                with st.expander("🔍 Error Details"):
                    st.code(traceback.format_exc())

def build_correlation_analysis_section(merged_df, selected_outcome):
    """Enhanced correlation analysis with proper handling of return values"""
    st.subheader("🔗 Correlation Analysis")

    col1, col2 = st.columns(2)
    with col1:
        correlation_method = st.selectbox(
            "Correlation method:",
            ["pearson", "spearman", "kendall"],
            help="Pearson: linear relationships, Spearman: monotonic relationships, Kendall: rank-based"
        )

    with col2:
        feature_selection_for_heatmap = st.selectbox(
            "Features for heatmap:",
            ["Top LASSO features", "Top Univariate features", "All features", "Custom selection"],
            help="Choose which features to include in correlation heatmap"
        )

    # Feature selection for correlation
    if feature_selection_for_heatmap == "Top LASSO features":
        if st.session_state.get('lasso_features'):
            features_for_analysis = list(st.session_state.lasso_features.keys())
        else:
            st.warning("⚠️ No LASSO features available. Please run LASSO selection first.")
            features_for_analysis = []

    elif feature_selection_for_heatmap == "Top Univariate features":
        if st.session_state.get('univariate_results') is not None:
            features_for_analysis = st.session_state.univariate_results['Feature'].tolist()
        else:
            st.warning("⚠️ No univariate results available. Please run univariate analysis first.")
            features_for_analysis = []

    elif feature_selection_for_heatmap == "All features":
        feature_cols = [col for col in merged_df.columns
                       if col not in ['PatientID', selected_outcome, 'modality', 'timepoint',
                                     'series_uid', 'roi_voxel_count', 'roi_percentage', 'extraction_timestamp']]
        features_for_analysis = feature_cols

    else:  # Custom selection
        feature_cols = [col for col in merged_df.columns
                       if col not in ['PatientID', selected_outcome, 'modality', 'timepoint',
                                     'series_uid', 'roi_voxel_count', 'roi_percentage', 'extraction_timestamp']]
        features_for_analysis = st.multiselect(
            "Select features for correlation analysis:",
            feature_cols,
            help="Choose specific features to include in correlation analysis"
        )

    if features_for_analysis and st.button("🔄 Generate Correlation Heatmap"):
        with st.spinner("Generating correlation heatmap..."):
            try:
                # Include outcome variable in correlation
                columns_for_correlation = features_for_analysis + [selected_outcome]

                # Generate correlation heatmap
                heatmap_fig = generate_correlation_heatmap(
                    merged_df,
                    columns_for_correlation
                )

                if heatmap_fig:
                    st.subheader("🔗 Feature Correlation Heatmap")
                    st.pyplot(heatmap_fig)
                    st.session_state.correlation_heatmap = heatmap_fig

            except Exception as e:
                st.error(f"❌ Correlation analysis failed: {str(e)}")
                with st.expander("🔍 Error Details"):
                    st.code(traceback.format_exc())

def build_advanced_analysis_section(merged_df, selected_outcome):
    """Advanced analysis methods"""
    st.subheader("🧪 Advanced Analysis Methods")

    st.info("🚧 Advanced analysis methods coming soon! This will include:")

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Machine Learning Methods:**")
        st.write("• Random Forest feature importance")
        st.write("• Support Vector Machine classification")
        st.write("• Gradient Boosting models")
        st.write("• Principal Component Analysis")

    with col2:
        st.write("**Statistical Methods:**")
        st.write("• Multiple testing correction")
        st.write("• Survival analysis (Cox regression)")
        st.write("• Clustering analysis")
        st.write("• Feature stability assessment")

    # Placeholder for future advanced methods
    advanced_method = st.selectbox(
        "Select advanced method:",
        ["Coming Soon - Random Forest", "Coming Soon - SVM", "Coming Soon - PCA", "Coming Soon - Survival Analysis"],
        disabled=True
    )

    st.button("🔄 Run Advanced Analysis", disabled=True, help="Advanced methods will be available in future updates")

# --- ENHANCED SIDEBAR ---
def build_sidebar():
    """Enhanced sidebar with comprehensive information and controls"""
    st.sidebar.title("🔬 Enhanced RadiomicsGUI")
    st.sidebar.write("Advanced Multi-Modal Radiomics Analysis Platform")
    st.sidebar.write("*With IBSI Compliance & NIfTI Support*")

    st.sidebar.divider()

    # Enhanced system information
    st.sidebar.subheader("💻 System Information")
    try:
        resource_info = check_system_resources()
        if isinstance(resource_info, dict):
            # Memory information
            ram_gb = resource_info.get('available_ram_gb', 0)
            total_ram_gb = resource_info.get('total_ram_gb', 0)
            ram_percent = resource_info.get('ram_percent', 0)

            st.sidebar.metric("Available RAM", f"{ram_gb:.1f} GB")
            st.sidebar.metric("Total RAM", f"{total_ram_gb:.1f} GB")
            st.sidebar.progress(ram_percent / 100)

            # CPU information
            cpu_count = resource_info.get('cpu_count', 1)
            st.sidebar.metric("CPU Cores", cpu_count)

            # Disk space
            disk_gb = resource_info.get('available_disk_gb', 0)
            if disk_gb > 0:
                st.sidebar.metric("Available Disk", f"{disk_gb:.1f} GB")
        else:
            st.sidebar.error("⚠️ System info unavailable")
    except Exception as e:
        st.sidebar.error(f"⚠️ System error: {str(e)}")

    st.sidebar.divider()

    # Enhanced progress tracking
    st.sidebar.subheader("📈 Enhanced Progress")

    # Main workflow progress
    data_uploaded = st.session_state.get('uploaded_data_path') is not None
    preprocessing_done = st.session_state.get('preprocessing_done', False)
    extraction_done = st.session_state.get('extraction_done', False)
    analysis_done = (st.session_state.get('univariate_results') is not None or
                    st.session_state.get('lasso_features') is not None)

    progress_items = [
        ("Data Upload", data_uploaded),
        ("Pre-processing", preprocessing_done),
        ("Feature Extraction", extraction_done),
        ("Statistical Analysis", analysis_done)
    ]

    for item_name, completed in progress_items:
        status_icon = "✅" if completed else "⏳"
        st.sidebar.write(f"{status_icon} {item_name}")

    # Calculate overall progress
    completed_steps = sum([data_uploaded, preprocessing_done, extraction_done, analysis_done])
    overall_progress = completed_steps / 4
    st.sidebar.progress(overall_progress)
    st.sidebar.write(f"Overall Progress: {overall_progress*100:.0f}%")

    st.sidebar.divider()

    # Session information
    st.sidebar.subheader("📊 Session Information")

    if st.session_state.get('dataset_df') is not None:
        dataset_size = len(st.session_state.dataset_df)
        st.sidebar.metric("Processed Patients", dataset_size)

    if st.session_state.get('features_df') is not None:
        feature_count = st.session_state.features_df.shape[1] - 1  # Exclude PatientID
        st.sidebar.metric("Extracted Features", feature_count)

    # Input format
    input_format = st.session_state.get('input_format', 'Unknown')
    st.sidebar.write(f"**Data Format:** {input_format.upper()}")

    # Selected modalities
    selected_modalities = st.session_state.get('selected_modalities', [])
    if selected_modalities:
        st.sidebar.write(f"**Modalities:** {', '.join(selected_modalities)}")

    # IBSI status
    ibsi_enabled = st.session_state.get('ibsi_features_enabled', False)
    ibsi_status = "Enabled" if ibsi_enabled else "Disabled"
    st.sidebar.write(f"**IBSI Features:** {ibsi_status}")

    st.sidebar.divider()

    # Enhanced controls
    st.sidebar.subheader("🔧 Enhanced Controls")

    # Session management
    if st.sidebar.button("🗑️ Clear Session", help="Clear all session data and restart"):
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    # Export session
    if st.sidebar.button("📁 Export Session", help="Export current session data"):
        session_data = {
            'preprocessing_done': st.session_state.get('preprocessing_done', False),
            'extraction_done': st.session_state.get('extraction_done', False),
            'input_format': st.session_state.get('input_format', 'unknown'),
            'selected_modalities': st.session_state.get('selected_modalities', []),
            'ibsi_features_enabled': st.session_state.get('ibsi_features_enabled', False)
        }

        st.sidebar.json(session_data)

    # Advanced settings
    with st.sidebar.expander("⚙️ Advanced Settings"):
        # ETA toggle
        eta_enabled = st.checkbox(
            "Enable ETA calculations",
            value=st.session_state.get('eta_enabled', True),
            help="Show estimated time remaining for long processes"
        )
        st.session_state['eta_enabled'] = eta_enabled

        # Debug mode
        debug_mode = st.checkbox(
            "Debug mode",
            value=False,
            help="Show detailed debug information"
        )
        st.session_state['debug_mode'] = debug_mode

    st.sidebar.divider()

    # Enhanced help section
    st.sidebar.subheader("❓ Enhanced Help")

    with st.sidebar.expander("🚀 Quick Start Guide"):
        st.write("""
        **DICOM Workflow:**
        1. Upload DICOM files or select directory
        2. Choose modality (CT/MR/PT)
        3. Scan for ROIs and contours
        4. Select target ROI
        5. Run preprocessing with robust methods
        6. Configure feature extraction (+ IBSI)
        7. Extract features
        8. Upload outcomes and analyze

        **NIfTI Workflow:**
        1. Upload NIfTI image/mask pairs
        2. Scan for valid pairs
        3. Run NIfTI preprocessing
        4. Configure feature extraction
        5. Extract features
        6. Upload outcomes and analyze
        """)

    with st.sidebar.expander("📁 Supported Formats"):
        st.write("""
        **DICOM Support:**
        - Individual .dcm files
        - ZIP archives with DICOM
        - RT-STRUCT files for ROIs
        - Multiple modalities (CT/MR/PT)
        - Longitudinal data support

        **NIfTI Support:**
        - .nii and .nii.gz files
        - Separate image/mask pairs
        - Auto-pairing by filename
        - Multi-label mask support

        **Outcomes:**
        - CSV files with PatientID
        - Binary, continuous, categorical
        - Survival data support
        """)

    with st.sidebar.expander("🎯 IBSI Compliance"):
        st.write("""
        **IBSI Features:**
        - Standard PyRadiomics → IBSI mapping
        - Additional IBSI-specific features
        - Morphological enhancements
        - Statistical completeness
        - Phase 2 compliance

        **Quality Assurance:**
        - Comprehensive mask validation
        - Robust processing methods
        - Error recovery systems
        - Progress tracking with ETA
        """)

    with st.sidebar.expander("🔧 Technical Details"):
        st.write("""
        **Processing Methods:**
        - 5 robust mask generation methods
        - Multiple file format support
        - Enhanced coordinate transformation
        - Morphological enhancement
        - Alternative format fallbacks

        **Analysis Methods:**
        - Enhanced univariate analysis
        - LASSO feature selection
        - Correlation analysis
        - Advanced ML methods (coming)
        """)

    st.sidebar.divider()

    # Version and credits
    st.sidebar.subheader("ℹ️ About")
    st.sidebar.write("**Enhanced RadiomicsGUI v3.0**")
    st.sidebar.write("*Built with PyRadiomics & Streamlit*")
    st.sidebar.write("*IBSI Phase 2 Compliant*")
    st.sidebar.write("*Multi-Modal & NIfTI Support*")
    st.sidebar.write("© 2025 Advanced Radiomics Research")

    # Feature extraction info
    try:
        extraction_info = get_feature_extraction_info()
        if 'radiomics_version' in extraction_info:
            st.sidebar.write(f"PyRadiomics: v{extraction_info['radiomics_version']}")
        if 'ibsi_features' in extraction_info:
            ibsi_info = extraction_info['ibsi_features']
            st.sidebar.write(f"IBSI Features: {ibsi_info.get('total_ibsi_compliance', 'N/A')}")
    except:
        pass

# --- MAIN APPLICATION ---
def main():
    """Main application entry point with proper tab structure"""
    st.set_page_config(
        page_title="Enhanced RadiomicsGUI",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state
    try:
        from utils import initialize_session_state, register_cleanup
        initialize_session_state()
        register_cleanup()
    except ImportError:
        st.error("Failed to initialize session state")
        return

    # Build sidebar
    build_sidebar()

    # Main content
    st.title("🔬 Enhanced RadiomicsGUI - Advanced Multi-Modal Analysis Platform")
    st.markdown("*Comprehensive DICOM & NIfTI support with IBSI compliance and robust processing*")
    st.divider()

    # Create the main tabs
    tab1, tab2, tab3 = st.tabs([
        "📤 **Step 1: Data Upload & Pre-processing**",
        "🔥 **Step 2: Enhanced Feature Extraction**",
        "📊 **Step 3: Statistical Analysis & Insights**"
    ])

    # Execute tab content
    with tab1:
        build_tab1_data_upload()

    with tab2:
        build_tab2_feature_extraction()

    with tab3:
        build_tab3_analysis()

if __name__ == "__main__":
    main()
