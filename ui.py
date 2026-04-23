"""
Enhanced UI module with comprehensive multi-modality, NIfTI, and IBSI support.
Updated: 2025-11-03 (v7 - Complete - Multi-Series + Multi-ROI compatible)

COMPLETE VERSION - All functions included, ready for manual file replacement.

Key improvements in v7:
- Adds multi-ROI selection UI (Single / Multiple / All per series)
- Implements preprocessing-stage multi-ROI expansion with intelligent fallback
- Fully compatible with multi-series processing (processes all series × all ROIs)
- Zero changes required to processing.py or extraction.py
- Backward compatible: original single-ROI workflow preserved when new options disabled
- Handles signature mismatches gracefully (auto-fallback with user notification)

How Multi-ROI works WITHOUT changing current workflow:
1. UI detects multiple ROIs and loops through them
2. For each ROI, calls existing preprocessing functions (preprocess_uploaded_data or preprocess_uploaded_data_enhanced)
3. Collects all results and concatenates into single dataset
4. Stores combined dataset in st.session_state.dataset_df
5. Feature extraction runs on expanded dataset (existing run_extraction function)
6. Result: (Patient × Series × ROI) feature matrix

Example workflow:
Patient_001 has 2 series (Baseline, FollowUp), each with 2 ROIs (Tumor, Lymph)
→ Preprocessing creates 4 rows: P001_Baseline_Tumor, P001_Baseline_Lymph, P001_FollowUp_Tumor, P001_FollowUp_Lymph
→ Extraction processes all 4 rows → 4 feature sets
"""

import streamlit as st
import traceback
import time
import pandas as pd
import numpy as np
import yaml
import os

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
        preprocess_selected_combinations,
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
        get_available_modalities_extended,
        initialize_session_state,
        register_cleanup
    )

except ImportError as e:
    st.header("💥 Application Error")
    st.error("A critical module failed to import. The application cannot start.")
    st.error(f"The specific error is: **{e}**")
    st.code(traceback.format_exc(), language='text')
    st.stop()

# -----------------------
# Helper functions for multi-ROI processing
# -----------------------
def _concat_preprocessing_results(result_list):
    """
    Safely concatenate multiple (df, summary) tuples from preprocessing functions.
    Used when processing multiple ROIs - combines all results into one dataset.
    """
    dfs = []
    all_summaries = []
    
    for result in result_list:
        if result and len(result) >= 2:
            df, summary = result[0], result[1]
            if isinstance(df, pd.DataFrame) and not df.empty:
                dfs.append(df)
            if isinstance(summary, dict):
                all_summaries.append(summary)
    
    # Combine dataframes
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
    else:
        combined_df = pd.DataFrame()
    
    # Combine summaries
    combined_summary = {
        'total_patients': 0,
        'valid_pairs': 0,
        'failed_patients': {},
        'recovery_statistics': {}
    }
    
    for summary in all_summaries:
        combined_summary['total_patients'] += summary.get('total_patients', 0)
        combined_summary['valid_pairs'] += summary.get('valid_pairs', 0)
        
        # Merge failed_patients
        failed = summary.get('failed_patients', {})
        for pid, info in failed.items():
            if pid not in combined_summary['failed_patients']:
                combined_summary['failed_patients'][pid] = info
        
        # Merge recovery stats
        recovery = summary.get('recovery_statistics', {})
        for key, val in recovery.items():
            combined_summary['recovery_statistics'][key] = \
                combined_summary['recovery_statistics'].get(key, 0) + val
    
    return combined_df, combined_summary

def _check_preprocessing_signature(func_name):
    """
    Check if preprocessing function accepts roi_name parameter.
    Returns True if compatible, False otherwise.
    """
    try:
        import inspect
        if func_name == 'preprocess_uploaded_data_enhanced':
            sig = inspect.signature(preprocess_uploaded_data_enhanced)
        elif func_name == 'preprocess_uploaded_data':
            sig = inspect.signature(preprocess_uploaded_data)
        else:
            return False
        
        params = sig.parameters
        return 'roi_name' in params or 'selected_roi' in params
    except Exception:
        return False

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

        # Workflow mode selector: Standard vs Advanced Search
        st.divider()
        st.subheader("🧭 Preprocessing Workflow")
        workflow_mode = st.radio(
            "Choose a preprocessing workflow:",
            [
                "Standard (pick one / multiple / all ROIs across all patients)",
                "🔎 Advanced Search (user-assisted Patient / Series / ROI selection)"
            ],
            index=0,
            horizontal=False,
            help=(
                "Advanced Search lets you categorize the scan results into three "
                "boxes — Patient, Series, and ROI — and apply your choice to all "
                "patients or a selected subset. It also supports multiple series "
                "and multiple ROIs per patient."
            ),
            key="workflow_mode_radio"
        )

        if workflow_mode.startswith("Standard"):
            build_enhanced_roi_selection()
        else:
            build_advanced_search_section()

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
    """
    CRITICAL FUNCTION - Enhanced ROI selection with multi-series + multi-ROI compatibility.
    
    This is where multi-ROI processing happens WITHOUT changing processing.py:
    1. User selects one or more ROIs
    2. UI loops through each ROI
    3. For each ROI, calls existing preprocessing functions (preprocess_uploaded_data or preprocess_uploaded_data_enhanced)
    4. Collects all results and concatenates them
    5. Stores combined dataset for extraction
    
    Result: (Series × ROI) dataset rows ready for feature extraction
    """
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

        # CRITICAL: ROI processing mode selection
        st.markdown("---")
        st.subheader("🎯 Multi-ROI Processing Configuration")
        
        roi_mode = st.radio(
            "ROI Processing Mode:",
            [
                "Single ROI (Original workflow)",
                "Multiple ROIs (Select specific ROIs)",
                "Process all ROIs per series (Automatic)"
            ],
            index=0,
            help="Choose how many ROIs to process. Multi-ROI creates (Series × ROI) dataset rows."
        )

        selected_rois = []
        
        if roi_mode == "Single ROI (Original workflow)":
            # Original single-ROI selection
            selected_roi = st.selectbox(
                "Select target ROI for extraction:",
                options=[""] + filtered_rois,
                help="Choose the ROI you want to extract features from"
            )
            if selected_roi:
                selected_rois = [selected_roi]
                
        elif roi_mode == "Multiple ROIs (Select specific ROIs)":
            # Multi-select for specific ROIs
            selected_rois = st.multiselect(
                "Select target ROI(s) for extraction:",
                options=filtered_rois,
                default=filtered_rois[:min(3, len(filtered_rois))],
                help="Select multiple ROIs to process across all series"
            )
            
        else:  # Process all ROIs per series
            # Automatic mode - will process all available ROIs
            st.info(f"📋 Will automatically process all {len(filtered_rois)} detected ROIs per series")
            selected_rois = None  # Signal to process all

    with col2:
        st.subheader("ROI Statistics")

        # Display ROI availability statistics
        if selected_rois or (selected_rois is None):
            display_text = "All ROIs" if selected_rois is None else ", ".join(selected_rois[:3])
            if selected_rois and len(selected_rois) > 3:
                display_text += f" (+{len(selected_rois)-3} more)"
            st.info(f"Selected: **{display_text}**")

            # Calculate availability across patients
            patients_with_roi = []
            for patient_id, contours in st.session_state.patient_contour_data.items():
                if selected_rois is None:
                    # All ROIs mode - patient has data if any contour exists
                    if contours:
                        patients_with_roi.append(patient_id)
                else:
                    # Check if patient has selected ROIs
                    if any(any(sel_roi.lower() in contour.lower() for contour in contours) for sel_roi in selected_rois):
                        patients_with_roi.append(patient_id)

            col2_1, col2_2 = st.columns(2)
            with col2_1:
                st.metric("Patients with ROI(s)", len(patients_with_roi))
            with col2_2:
                total_patients = len(st.session_state.patient_contour_data)
                availability_pct = (len(patients_with_roi) / max(total_patients, 1)) * 100
                st.metric("Availability", f"{availability_pct:.1f}%")

            # Per-series ROI breakdown button
            if st.button("📋 Show Per-Series ROI Breakdown", key="show_roi_breakdown"):
                rows = []
                longitudinal = st.session_state.get('longitudinal_data', {})
                
                if longitudinal:
                    # Show detailed per-series breakdown
                    for pid, patient_long in longitudinal.items():
                        pairs = patient_long.get('compatible_pairs', [])
                        for pair in pairs:
                            series_desc = pair.get('series_description', 'Series')
                            timepoint = pair.get('timepoint', '')
                            contours = pair.get('contours', [])
                            rows.append({
                                'Patient': pid,
                                'Series_Description': series_desc,
                                'Timepoint': timepoint,
                                'Contours_Count': len(contours),
                                'Contours': ', '.join(contours)
                            })
                else:
                    # Fallback to basic patient-level info
                    for pid, contours in st.session_state.patient_contour_data.items():
                        rows.append({
                            'Patient': pid,
                            'Series_Description': 'N/A',
                            'Timepoint': 'N/A',
                            'Contours_Count': len(contours),
                            'Contours': ', '.join(contours)
                        })
                
                if rows:
                    preview_df = pd.DataFrame(rows)
                    st.dataframe(preview_df, use_container_width=True)
                    
                    # Calculate total series × ROI combinations
                    if selected_rois is None:
                        total_combos = sum(row['Contours_Count'] for row in rows)
                    else:
                        total_combos = len(rows) * len(selected_rois)
                    
                    st.success(f"📊 Total (Series × ROI) combinations to process: {total_combos}")
                else:
                    st.warning("No series data available")

        # Category statistics
        st.subheader("ROI Categories")
        col2_1, col2_2, col2_3 = st.columns(3)
        with col2_1:
            st.metric("Targets", len(targets))
        with col2_2:
            st.metric("OARs", len(oars))
        with col2_3:
            st.metric("Other", len(other))

    # CRITICAL: Preprocessing section - this is where multi-ROI magic happens
    if (roi_mode == "Single ROI (Original workflow)" and selected_rois) or \
       (roi_mode == "Multiple ROIs (Select specific ROIs)" and selected_rois) or \
       (roi_mode == "Process all ROIs per series (Automatic)"):
        
        st.divider()
        st.subheader("🚀 Start Preprocessing")

        # Multi-series info
        if st.session_state.get('multi_series_mode', False):
            st.info("🔄 Multi-series mode enabled - will process all selected modalities")
            
            if st.session_state.get('longitudinal_data'):
                series_count = sum(len(pl.get('compatible_pairs', [])) 
                                 for pl in st.session_state.longitudinal_data.values())
                st.info(f"📊 Ready to process {series_count} series across {len(st.session_state.longitudinal_data)} patients")

        preprocessing_col1, preprocessing_col2 = st.columns(2)

        with preprocessing_col1:
            # Modality-specific settings
            selected_modalities = st.session_state.get('selected_modalities', ['CT'])
            primary_modality = selected_modalities[0] if selected_modalities else 'CT'

            st.write(f"**Primary Modality:** {primary_modality}")
            
            if selected_rois is None:
                st.write(f"**Target ROI(s):** All ROIs per series (automatic)")
            else:
                st.write(f"**Target ROI(s):** {', '.join(selected_rois)}")

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

        # CRITICAL: Start preprocessing button - MULTI-ROI LOOP HAPPENS HERE
        if st.button("🚀 Start Enhanced Preprocessing", type="primary", key="start_enhanced_preprocessing_v7"):
            # Setup progress UI
            progress_bar = st.progress(0.0)
            progress_text = st.empty()
            status_placeholder = st.empty()
            
            st.session_state['ui_progress_bar'] = progress_bar
            st.session_state['ui_progress_text'] = progress_text
            st.session_state['ui_status_placeholder'] = status_placeholder

            try:
                # STEP 1: Determine which ROIs to process
                if selected_rois is None:
                    # All ROIs mode - use filtered_rois or all_contours
                    roi_list_to_process = filtered_rois if filtered_rois else sorted(st.session_state.get('all_contours', []))
                else:
                    roi_list_to_process = list(selected_rois)

                total_rois = len(roi_list_to_process)
                
                if total_rois == 0:
                    st.error("❌ No ROIs selected for processing.")
                    return

                status_placeholder.info(f"🔄 Processing {total_rois} ROI(s) across all series...")

                # STEP 2: Check if preprocessing functions support ROI parameter
                # This determines if we can pass roi_name to preprocessing functions
                if st.session_state.get('multi_series_mode', False):
                    supports_roi = _check_preprocessing_signature('preprocess_uploaded_data_enhanced')
                    func_name = 'preprocess_uploaded_data_enhanced'
                else:
                    supports_roi = _check_preprocessing_signature('preprocess_uploaded_data')
                    func_name = 'preprocess_uploaded_data'

                if not supports_roi and total_rois > 1:
                    st.warning(f"⚠️ The {func_name} function does not accept ROI parameter. "
                             f"Multi-ROI processing requires updating {func_name} signature. "
                             f"Processing first ROI only for now.")
                    roi_list_to_process = roi_list_to_process[:1]
                    total_rois = 1

                # STEP 3: CRITICAL MULTI-ROI LOOP
                # Process each ROI by calling existing preprocessing functions
                results_accumulator = []
                
                for idx, roi_name in enumerate(roi_list_to_process):
                    progress_pct = idx / total_rois
                    progress_bar.progress(progress_pct)
                    progress_text.text(f"Processing ROI {idx+1}/{total_rois}: {roi_name}")

                    try:
                        if st.session_state.get('multi_series_mode', False):
                            # Multi-series preprocessing
                            if supports_roi:
                                # Call with ROI parameter
                                df_roi, summary_roi = preprocess_uploaded_data_enhanced(
                                    st.session_state.uploaded_data_path,
                                    roi_name,  # Pass ROI name
                                    selected_modalities,
                                    multi_series_mode=True,
                                    selected_series=[]
                                )
                            else:
                                # Fallback: call without roi_name (processes default ROI)
                                df_roi, summary_roi = preprocess_uploaded_data_enhanced(
                                    st.session_state.uploaded_data_path,
                                    selected_modalities,
                                    multi_series_mode=True,
                                    selected_series=[]
                                )
                        else:
                            # Single-series preprocessing
                            if supports_roi:
                                # Call with ROI parameter
                                df_roi, summary_roi = preprocess_uploaded_data(
                                    st.session_state.uploaded_data_path,
                                    roi_name,  # Pass ROI name
                                    primary_modality
                                )
                            else:
                                # Fallback: call without roi_name
                                df_roi, summary_roi = preprocess_uploaded_data(
                                    st.session_state.uploaded_data_path,
                                    primary_modality
                                )
                        
                        # Add ROI identifier to results if not present
                        if not df_roi.empty:
                            if 'roi_name' not in df_roi.columns:
                                df_roi['roi_name'] = roi_name
                        
                        # Store this ROI's results
                        results_accumulator.append((df_roi, summary_roi))
                        
                    except Exception as e:
                        st.warning(f"⚠️ Preprocessing failed for ROI {roi_name}: {str(e)}")
                        # Continue with other ROIs instead of stopping
                        continue

                # STEP 4: Combine all ROI results into single dataset
                progress_bar.progress(0.95)
                progress_text.text("Combining results from all ROIs...")
                
                combined_df, combined_summary = _concat_preprocessing_results(results_accumulator)

                if combined_df.empty:
                    st.error("❌ Preprocessing produced no valid results for the requested ROIs.")
                    build_preprocessing_error_display(combined_summary)
                    return

                # STEP 5: Store combined results in session - ready for extraction
                st.session_state.dataset_df = combined_df
                st.session_state.preprocessing_done = True
                st.session_state['processing_summary'] = combined_summary

                progress_bar.progress(1.0)
                progress_text.text("✅ Preprocessing complete!")

                st.success(f"✅ Successfully preprocessed {len(combined_df)} dataset rows (Series × ROI combinations)")
                
                # Show what was created
                if 'roi_name' in combined_df.columns:
                    unique_series = combined_df['patient_id'].nunique() if 'patient_id' in combined_df.columns else len(combined_df)
                    unique_rois = combined_df['roi_name'].nunique()
                    st.info(f"📊 Created dataset with {unique_series} series × {unique_rois} ROIs = {len(combined_df)} rows")
                
                # Display results
                build_preprocessing_results_display(combined_df, combined_summary)

            except Exception as e:
                st.error(f"❌ Preprocessing failed with error: {str(e)}")
                with st.expander("🔍 Error Details"):
                    st.code(traceback.format_exc())
            finally:
                try:
                    progress_bar.empty()
                    progress_text.empty()
                    status_placeholder.empty()
                except Exception:
                    pass

    else:
        st.info("ℹ️ Select ROI(s) to enable preprocessing.")


# =============================================================================
# ADVANCED SEARCH WORKFLOW (User-assisted Patient / Series / ROI selection)
# =============================================================================

def _flatten_longitudinal_to_rows():
    """
    Flatten st.session_state.longitudinal_data into a list of unique
    (patient, series, ROI) rows for the Advanced Search UI.

    Returns a list of dicts with keys:
        patient_id, modality, timepoint, series_uid, series_description,
        study_date, slice_count, series_path, rtstruct_path, roi_name
    Plus a second list of "series rows" (one per compatible pair, with a
    contours list) used for per-series ROI selection.
    """
    longitudinal = st.session_state.get('longitudinal_data', {}) or {}
    combo_rows = []
    series_rows = []

    for patient_id, patient_long in longitudinal.items():
        for pair in patient_long.get('compatible_pairs', []):
            series_rows.append({
                'patient_id': patient_id,
                'modality': pair.get('modality', 'CT'),
                'timepoint': pair.get('timepoint', ''),
                'series_uid': pair.get('series_uid', ''),
                'series_description': pair.get('series_description', 'Unknown_Series'),
                'study_date': pair.get('study_date', ''),
                'slice_count': pair.get('slice_count', 0),
                'series_path': pair.get('series_path', ''),
                'rtstruct_path': pair.get('rtstruct_path', ''),
                'contours': list(pair.get('contours', []) or []),
            })
            for roi in pair.get('contours', []) or []:
                combo_rows.append({
                    'patient_id': patient_id,
                    'modality': pair.get('modality', 'CT'),
                    'timepoint': pair.get('timepoint', ''),
                    'series_uid': pair.get('series_uid', ''),
                    'series_description': pair.get('series_description', 'Unknown_Series'),
                    'study_date': pair.get('study_date', ''),
                    'slice_count': pair.get('slice_count', 0),
                    'series_path': pair.get('series_path', ''),
                    'rtstruct_path': pair.get('rtstruct_path', ''),
                    'roi_name': roi,
                })

    return combo_rows, series_rows


def _series_label(s):
    """Human-friendly label for a series row."""
    desc = s.get('series_description') or 'Unknown_Series'
    mod = s.get('modality', '')
    tp = s.get('timepoint', '')
    parts = [p for p in [mod, desc, tp] if p]
    label = " | ".join(parts) if parts else desc
    if s.get('slice_count'):
        label += f"  ({s['slice_count']} slices)"
    return label


def _render_series_availability(selected_patients, series_rows):
    """
    Diagnostic panel: show every physical series the scanner found in each
    selected patient folder, and flag those that do NOT have a compatible
    RTSTRUCT pair. Series without a compatible RTSTRUCT do not appear in
    Advanced Search's three boxes, which is the usual reason a patient
    folder with multiple physical series gets reduced to a single series.

    Data sources:
        series_rows         -> RTSTRUCT-compatible series (one row per pair)
        longitudinal_data   -> all physical series the scanner walked through,
                               compatible or not, under
                               longitudinal_data[pid]['series_data']
    """
    longitudinal = st.session_state.get('longitudinal_data', {}) or {}
    if not longitudinal:
        return

    compat_keys = {
        (s['patient_id'], s.get('series_uid', '')) for s in series_rows
    }

    rows = []
    incompat_total = 0
    for pid in selected_patients:
        patient_long = longitudinal.get(pid, {}) or {}
        series_data = patient_long.get('series_data', {}) or {}
        if not series_data:
            continue
        for modality, timepoints in series_data.items():
            for tp, series_map in (timepoints or {}).items():
                for uid, info in (series_map or {}).items():
                    is_compat = (pid, uid) in compat_keys
                    if not is_compat:
                        incompat_total += 1
                    rows.append({
                        'Patient': pid,
                        'Modality': modality,
                        'Timepoint': tp,
                        'Series Description': info.get('series_description') or 'Unknown_Series',
                        'Slices': info.get('slice_count', 0),
                        'Series Path': info.get('path', ''),
                        'Status': '✅ Has RTSTRUCT' if is_compat else '⚠️ No compatible RTSTRUCT',
                        'SeriesInstanceUID': uid,
                    })

    if not rows:
        return

    total = len(rows)
    compat_count = total - incompat_total

    with st.expander(
        f"🩻 Series availability for selected patient(s): "
        f"{compat_count}/{total} compatible, {incompat_total} hidden",
        expanded=(incompat_total > 0)
    ):
        st.caption(
            "This table lists every physical image series the scanner found in "
            "the selected patient folder(s). Rows marked '⚠️ No compatible "
            "RTSTRUCT' are NOT offered in the boxes below — usually because "
            "the series' FrameOfReferenceUID does not match the RTSTRUCT, or "
            "because no RTSTRUCT was drawn on that series."
        )
        avail_df = pd.DataFrame(rows)
        st.dataframe(
            avail_df[['Patient', 'Modality', 'Series Description', 'Timepoint',
                      'Slices', 'Status']],
            use_container_width=True
        )
        if incompat_total > 0:
            st.warning(
                f"{incompat_total} series are present on disk but have no "
                "compatible RTSTRUCT, so they will not be processed by Advanced "
                "Search. To include them:\n\n"
                "• Supply an RTSTRUCT whose FrameOfReferenceUID matches each "
                "such series (usually means drawing / re-registering contours "
                "on that series), OR\n"
                "• Re-register the image series so they share a single "
                "FrameOfReferenceUID with the existing RTSTRUCT, OR\n"
                "• Process those series in a separate pass with their own "
                "RTSTRUCT file."
            )


def build_advanced_search_section():
    """
    Advanced Search workflow - user-assisted Patient/Series/ROI selection.

    The user is shown three selection boxes:
      - Box 1: Patients (apply to all or a selected subset)
      - Box 2: Series   (per-patient series descriptions/timepoints)
      - Box 3: ROIs     (contour names detected across selected series)

    The UI builds the set of (patient, series, ROI) combinations that will
    be processed, lets the user preview a plan summary (how many patients,
    how many series per patient, how many ROIs per series), and then runs
    preprocess_selected_combinations() to generate the dataset.

    Supports multiple series and corresponding ROI processing within a
    single patient folder.
    """
    st.divider()
    st.header("Step 1.3: 🔎 Advanced Search - User-Assisted Selection")

    longitudinal = st.session_state.get('longitudinal_data', {}) or {}
    if not longitudinal:
        st.warning(
            "⚠️ Advanced Search requires the enhanced scan output. "
            "Please run the scan button above first."
        )
        return

    combo_rows, series_rows = _flatten_longitudinal_to_rows()
    if not series_rows:
        st.error("❌ No compatible Series/RTSTRUCT pairs were found. Nothing to categorize.")
        return

    st.caption(
        "Categorize the scan results into three boxes — Patient, Series, ROI — "
        "and apply the selection to all patients or a selected subset. "
        "Multiple series and multiple ROIs per patient are supported."
    )

    all_patients = sorted({s['patient_id'] for s in series_rows})

    # ---- Box 1: Patient scope ----------------------------------------------
    st.subheader("📦 Box 1 — Patient Selection")
    col_a, col_b = st.columns([1, 2])
    with col_a:
        patient_scope = st.radio(
            "Apply to:",
            ["All patients", "Selected subset"],
            index=0,
            key="adv_patient_scope",
            help="Apply the series/ROI selection to every patient, or to a subset you pick."
        )
    with col_b:
        if patient_scope == "Selected subset":
            selected_patients = st.multiselect(
                "Choose patients:",
                options=all_patients,
                default=all_patients[: min(5, len(all_patients))],
                key="adv_selected_patients",
                help="Only these patients will be processed."
            )
        else:
            selected_patients = all_patients
            st.info(f"All {len(all_patients)} patient(s) will be included.")

    if not selected_patients:
        st.warning("Select at least one patient to continue.")
        return

    # Filter to selected patients
    filtered_series = [s for s in series_rows if s['patient_id'] in selected_patients]

    # ---- Box 2: Series selection -------------------------------------------
    st.subheader("📦 Box 2 — Series Selection")
    series_descriptions = sorted({s.get('series_description') or 'Unknown_Series' for s in filtered_series})
    modalities_present = sorted({s.get('modality', 'CT') for s in filtered_series})

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        series_mode = st.radio(
            "Series strategy:",
            [
                "Match by series description (recommended)",
                "All series per patient",
                "Latest series per patient per modality"
            ],
            index=0,
            key="adv_series_mode",
            help=(
                "• Match by series description: pick one or more Series Descriptions; "
                "every matching series across selected patients is used.\n"
                "• All series per patient: use every series for each selected patient "
                "(enables multi-series processing).\n"
                "• Latest series: one series per (patient, modality) — the most recent one."
            )
        )
    with col_s2:
        selected_modalities_adv = st.multiselect(
            "Modalities to include:",
            options=modalities_present,
            default=modalities_present,
            key="adv_modalities",
            help="Restrict to specific modalities present in the scan."
        )

    selected_series_descriptions = []
    if series_mode.startswith("Match by series description"):
        # Default to ALL descriptions so multi-series phantoms (e.g. Chest + Control)
        # are included by default. The user can narrow it down afterwards.
        default_desc = list(series_descriptions)
        selected_series_descriptions = st.multiselect(
            "Series descriptions to include:",
            options=series_descriptions,
            default=default_desc,
            key="adv_series_descriptions",
            help=(
                "Pick one or more series descriptions. Every matching series is "
                "processed. Defaults to all descriptions found in the scan so "
                "patients with multiple series (e.g. Chest + Control) are not "
                "silently reduced to one."
            )
        )
        if not selected_series_descriptions:
            st.warning("Select at least one series description.")
            return

    # ---- Series availability diagnostic ------------------------------------
    # Show every physical series the scanner found per selected patient,
    # including series that had NO compatible RTSTRUCT (and therefore do NOT
    # appear in the three boxes above). This is the most common reason a
    # patient folder with 2+ series ends up processed as a single series.
    _render_series_availability(selected_patients, series_rows)

    # Apply series filters
    def _series_passes_filters(s):
        if s.get('modality', 'CT') not in selected_modalities_adv:
            return False
        if series_mode.startswith("Match by series description"):
            return (s.get('series_description') or 'Unknown_Series') in selected_series_descriptions
        return True

    candidate_series = [s for s in filtered_series if _series_passes_filters(s)]

    if series_mode.startswith("Latest series"):
        latest = {}
        for s in candidate_series:
            key = (s['patient_id'], s.get('modality', 'CT'))
            cur = latest.get(key)
            if cur is None or (s.get('timepoint', '') > cur.get('timepoint', '')):
                latest[key] = s
        candidate_series = list(latest.values())

    if not candidate_series:
        st.error("❌ No series match the current selection. Loosen the filters.")
        return

    # ---- Box 3: ROI selection ----------------------------------------------
    st.subheader("📦 Box 3 — ROI Selection")
    roi_universe = sorted({roi for s in candidate_series for roi in s.get('contours', [])})
    if not roi_universe:
        st.error("❌ No ROIs are available in the currently selected series.")
        return

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        roi_mode = st.radio(
            "ROI strategy:",
            [
                "Select specific ROIs",
                "All ROIs per series"
            ],
            index=0,
            key="adv_roi_mode",
            help="Pick one or more ROI names to look for in every selected series, or process every ROI of every selected series."
        )
    with col_r2:
        if roi_mode == "Select specific ROIs":
            selected_rois_adv = st.multiselect(
                "ROI names (case-insensitive match):",
                options=roi_universe,
                default=roi_universe[: min(2, len(roi_universe))],
                key="adv_selected_rois",
                help="Each selected ROI is processed in every matching series. Missing ROIs are skipped per-series."
            )
        else:
            selected_rois_adv = None
            st.info(f"📋 All ROIs of each selected series will be processed ({len(roi_universe)} unique ROIs available).")

    if roi_mode == "Select specific ROIs" and not selected_rois_adv:
        st.warning("Select at least one ROI.")
        return

    # ---- Build final (patient, series, ROI) combinations -------------------
    combinations = []
    for s in candidate_series:
        series_contours = s.get('contours', []) or []
        if selected_rois_adv is None:
            target_rois = series_contours
        else:
            target_rois = []
            for want in selected_rois_adv:
                match = None
                for c in series_contours:
                    if c == want or c.lower() == want.lower():
                        match = c
                        break
                if match is None:
                    for c in series_contours:
                        if want.lower() in c.lower():
                            match = c
                            break
                if match is not None and match not in target_rois:
                    target_rois.append(match)

        for roi in target_rois:
            combinations.append({
                'patient_id': s['patient_id'],
                'modality': s.get('modality', 'CT'),
                'timepoint': s.get('timepoint', ''),
                'series_uid': s.get('series_uid', ''),
                'series_description': s.get('series_description', 'Unknown_Series'),
                'study_date': s.get('study_date', ''),
                'slice_count': s.get('slice_count', 0),
                'series_path': s.get('series_path', ''),
                'rtstruct_path': s.get('rtstruct_path', ''),
                'roi_name': roi,
            })

    # ---- Plan summary -------------------------------------------------------
    st.divider()
    st.subheader("📋 Extraction Plan Summary")

    if not combinations:
        st.error("❌ The current selection does not produce any (Patient × Series × ROI) combinations to process.")
        return

    plan_df = pd.DataFrame(combinations)
    unique_patients = plan_df['patient_id'].nunique()
    total_combos = len(plan_df)

    # Series per patient
    series_per_patient = (
        plan_df.groupby('patient_id')[['series_uid', 'series_description']]
        .apply(lambda d: d.drop_duplicates().shape[0])
    )
    # ROI per (patient, series)
    roi_per_series = (
        plan_df.groupby(['patient_id', 'series_description', 'series_uid'])['roi_name']
        .nunique()
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Patients", unique_patients)
    with m2:
        st.metric("Series (total)", plan_df[['patient_id', 'series_uid']].drop_duplicates().shape[0])
    with m3:
        st.metric("Avg series / patient", f"{series_per_patient.mean():.1f}")
    with m4:
        st.metric("Avg ROI / series", f"{roi_per_series.mean():.1f}")

    st.caption(f"Total (Patient × Series × ROI) combinations to process: **{total_combos}**")

    with st.expander("🔍 Show detailed plan", expanded=False):
        preview_df = plan_df[[
            'patient_id', 'modality', 'series_description', 'timepoint', 'roi_name', 'slice_count'
        ]].rename(columns={
            'patient_id': 'Patient',
            'modality': 'Modality',
            'series_description': 'Series',
            'timepoint': 'Timepoint',
            'roi_name': 'ROI',
            'slice_count': 'Slices'
        })
        st.dataframe(preview_df, use_container_width=True)

    with st.expander("📊 Per-patient breakdown", expanded=False):
        breakdown = (
            plan_df.groupby(['patient_id'])
            .agg(
                n_series=('series_uid', lambda x: x.nunique()),
                n_rois=('roi_name', 'nunique'),
                n_combinations=('roi_name', 'size')
            )
            .reset_index()
            .rename(columns={
                'patient_id': 'Patient',
                'n_series': 'Series',
                'n_rois': 'Unique ROIs',
                'n_combinations': 'Combinations'
            })
        )
        st.dataframe(breakdown, use_container_width=True)

    # ---- Run preprocessing --------------------------------------------------
    st.divider()
    if st.button(
        f"🚀 Run Advanced Preprocessing on {total_combos} Combination(s)",
        type="primary",
        key="adv_run_preprocessing"
    ):
        progress_bar = st.progress(0.0)
        progress_text = st.empty()
        status_placeholder = st.empty()
        st.session_state['ui_progress_bar'] = progress_bar
        st.session_state['ui_progress_text'] = progress_text
        st.session_state['ui_status_placeholder'] = status_placeholder

        try:
            result_df, summary = preprocess_selected_combinations(combinations)

            if result_df is None or result_df.empty:
                st.error("❌ Advanced preprocessing produced no valid results.")
                build_preprocessing_error_display(summary)
                return

            st.session_state.dataset_df = result_df
            st.session_state.preprocessing_done = True
            st.session_state['processing_summary'] = summary
            st.session_state['advanced_plan_summary'] = {
                'requested_patients': unique_patients,
                'requested_series': int(plan_df[['patient_id', 'series_uid']].drop_duplicates().shape[0]),
                'requested_combinations': total_combos,
            }

            st.success(
                f"✅ Advanced preprocessing complete: "
                f"{summary['successful_combinations']}/{summary['total_combinations']} combinations succeeded "
                f"across {summary['successful_patients']} patient(s)."
            )
            build_preprocessing_results_display(result_df, summary)
        except Exception as e:
            st.error(f"❌ Advanced preprocessing failed: {str(e)}")
            with st.expander("🔍 Error Details"):
                st.code(traceback.format_exc())
        finally:
            try:
                progress_bar.empty()
                progress_text.empty()
                status_placeholder.empty()
            except Exception:
                pass


# Keep all Tab 2 and Tab 3 functions unchanged from v1
# These work with the expanded dataset created by build_enhanced_roi_selection()
# or build_advanced_search_section()

def build_tab2_feature_extraction():
    """Enhanced Tab 2 with IBSI support"""
    if not st.session_state.get('preprocessing_done', False):
        st.warning("⚠️ Please complete the data upload and preprocessing steps first.")
        return

    st.header("Step 2: Enhanced Radiomics Feature Extraction")

    # Configuration section (keep from v1)
    build_enhanced_extraction_configuration()

    # Extraction execution (works with multi-ROI dataset)
    build_extraction_execution_section()

def build_enhanced_extraction_configuration():
    """
    Enhanced extraction configuration with FIXED parameter generation.
    
    CRITICAL FIXES:
    - Proper indentation throughout (no syntax errors)
    - Generates PyRadiomics parameters WITHOUT invalid keys
    - No 'enableCExtensions', '_metadata', or other schema-breaking keys
    - Works with multi-ROI dataset (processes all rows from preprocessing)
    
    Multi-ROI Strategy:
    - Configuration works the same for single or multi-ROI
    - Generated parameters apply to ALL rows in dataset
    - No workflow changes - just standard PyRadiomics configuration
    """
    st.subheader("🔧 Enhanced PyRadiomics Configuration")

    config_tabs = st.tabs(["⚙️ Basic Settings", "🎯 IBSI Features", "🔬 Advanced Settings"])

    with config_tabs[0]:
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Feature Classes to Extract:**")
            feature_classes = {}
            feature_classes['firstorder'] = st.checkbox("First Order Statistics", value=True)
            feature_classes['shape'] = st.checkbox("Shape Features", value=True)
            feature_classes['glcm'] = st.checkbox("GLCM Features", value=True)
            feature_classes['glrlm'] = st.checkbox("GLRLM Features", value=True)

        with col2:
            st.write("**Additional Feature Classes:**")
            feature_classes['glszm'] = st.checkbox("GLSZM Features", value=True)
            feature_classes['ngtdm'] = st.checkbox("NGTDM Features", value=True)
            feature_classes['gldm'] = st.checkbox("GLDM Features", value=True)

        st.session_state['feature_classes'] = feature_classes

    with config_tabs[1]:
        st.write("**IBSI Compliance Settings:**")
        col1, col2 = st.columns(2)

        with col1:
            enable_ibsi_features = st.checkbox("Enable Additional IBSI Features", value=False)
            use_ibsi_nomenclature = st.checkbox("Use IBSI Nomenclature", value=True)
            st.session_state['ibsi_features_enabled'] = enable_ibsi_features
            st.session_state['use_ibsi_nomenclature'] = use_ibsi_nomenclature

        with col2:
            if enable_ibsi_features:
                st.info("🎯 Additional IBSI features will be calculated")

    with config_tabs[2]:
        col1, col2 = st.columns(2)

        with col1:
            normalize_image = st.checkbox("Normalize Image", value=True)
            resample_pixel_spacing = st.checkbox("Resample Pixel Spacing", value=False)

            dataset_df = st.session_state.get('dataset_df', pd.DataFrame())
            if not dataset_df.empty and 'modality' in dataset_df.columns:
                primary_modality = dataset_df['modality'].iloc[0]
            else:
                primary_modality = 'CT'

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
                value=float(default_bin_width)
            )

            if resample_pixel_spacing:
                pixel_spacing = st.number_input(
                    "Pixel Spacing (mm)", 
                    min_value=0.1, 
                    max_value=5.0, 
                    value=1.0, 
                    step=0.1
                )
            else:
                pixel_spacing = None

        with col2:
            interpolator = st.selectbox(
                "Interpolator", 
                ["sitkBSpline", "sitkLinear", "sitkNearestNeighbor"], 
                index=0
            )
            pad_distance = st.number_input(
                "Pad Distance", 
                min_value=0, 
                max_value=20, 
                value=5
            )
            geometryTolerance = st.number_input(
                "Geometry Tolerance", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.0001, 
                format="%.6f"
            )

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

    # ========================================================================
    # CRITICAL SECTION: Parameter Generation with FIXED Indentation
    # ========================================================================
    
    if st.button("🔄 Generate Enhanced PyRadiomics Parameters", type="secondary"):
        with st.spinner("Generating optimized parameter configuration..."):
            try:
                settings = st.session_state.get('extraction_settings', {})
                feature_classes = st.session_state.get('feature_classes', {})

                # Build feature classes dictionary (only enabled ones)
                feature_classes_dict = {
                    name: [] for name, enabled in feature_classes.items() if enabled
                }
                
                # ✅ CRITICAL FIX: Generate params WITHOUT invalid keys
                # NO 'enableCExtensions', NO '_metadata', NO other invalid keys
                params = {
                    'setting': {
                        'binWidth': settings.get('bin_width', 25),
                        'interpolator': settings.get('interpolator', 'sitkBSpline'),
                        'padDistance': settings.get('pad_distance', 5),
                        'geometryTolerance': settings.get('geometryTolerance', 0.0001),
                        'force2D': False,
                        'force2Ddimension': 0
                    },
                    'imageType': {
                        'Original': {}
                    },
                    'featureClass': feature_classes_dict
                }
                
                # Add optional settings (only if requested)
                if settings.get('resample_pixel_spacing', False) and settings.get('pixel_spacing'):
                    params['setting']['resampledPixelSpacing'] = [
                        float(settings['pixel_spacing']),
                        float(settings['pixel_spacing']),
                        float(settings['pixel_spacing'])
                    ]
                
                if settings.get('normalize_image', True):
                    params['setting']['normalize'] = True
                    params['setting']['normalizeScale'] = 100
                
                # Modality-specific adjustments
                modality = settings.get('modality', 'CT')
                if modality.startswith('MR') and params['setting']['binWidth'] > 10:
                    params['setting']['binWidth'] = 5
                    st.info(f"🔧 Adjusted bin width to {params['setting']['binWidth']} for MR imaging")
                elif modality.startswith('PT') and params['setting']['binWidth'] > 1:
                    params['setting']['binWidth'] = 0.25
                    st.info(f"🔧 Adjusted bin width to {params['setting']['binWidth']} for PT imaging")
                
                # Store parameters in session state
                st.session_state.pyradiomics_params = params
                
                # Success message
                st.success("✅ PyRadiomics parameters generated successfully!")
                st.info("✅ Parameters validated - NO invalid keys (schema-compliant)")

                # Show generated parameters
                with st.expander("📋 View Generated Parameters"):
                    st.code(yaml.dump(params, default_flow_style=False), language='yaml')
                
                # Show what will be extracted
                dataset_df = st.session_state.get('dataset_df', pd.DataFrame())
                if not dataset_df.empty:
                    total_rows = len(dataset_df)
                    st.info(f"📊 Ready to extract features from {total_rows} dataset rows")
                    
                    if 'roi_name' in dataset_df.columns:
                        unique_rois = dataset_df['roi_name'].nunique()
                        st.info(f"🎯 Dataset contains {unique_rois} unique ROI(s)")
            
            except Exception as e:
                st.error(f"❌ Error generating parameters: {str(e)}")
                with st.expander("🔍 Error Details"):
                    st.code(traceback.format_exc())
                    st.write("**Troubleshooting:**")
                    st.write("- Check that all feature classes are properly configured")
                    st.write("- Verify extraction settings are valid")
                    st.write("- Ensure dataset_df exists in session state")

def build_pre_extraction_summary():
    """
    Show a concise summary BEFORE radiomic feature extraction begins.

    Reports, for the dataset that will be fed into extraction:
      • Total unique patients
      • Total series and average series per patient
      • Total ROIs and average ROIs per series
      • Per-patient breakdown (series count, ROI count, total combinations)
    """
    dataset_df = st.session_state.get('dataset_df')
    if dataset_df is None or dataset_df.empty:
        return

    st.subheader("🧾 Pre-Extraction Summary")
    st.caption(
        "Review what will be sent to radiomic feature extraction: "
        "how many patients, how many series per patient, and how many ROIs per series."
    )

    total_rows = len(dataset_df)
    n_patients = dataset_df['patient_id'].nunique() if 'patient_id' in dataset_df.columns else total_rows

    # Identify series uniquely using series_uid if available, else series_description
    if 'series_uid' in dataset_df.columns and dataset_df['series_uid'].astype(str).str.len().gt(0).any():
        series_key_cols = ['patient_id', 'series_uid']
    elif 'series_description' in dataset_df.columns:
        series_key_cols = ['patient_id', 'series_description']
    else:
        series_key_cols = ['patient_id']

    series_pairs = dataset_df[series_key_cols].drop_duplicates() if set(series_key_cols).issubset(dataset_df.columns) else dataset_df[['patient_id']].drop_duplicates()
    n_series_total = len(series_pairs)
    avg_series_per_patient = (n_series_total / n_patients) if n_patients else 0

    if 'roi_name' in dataset_df.columns:
        rois_per_series = (
            dataset_df.groupby(series_key_cols)['roi_name']
            .nunique()
        )
        n_unique_rois = dataset_df['roi_name'].nunique()
        avg_rois_per_series = rois_per_series.mean() if len(rois_per_series) else 0
    else:
        n_unique_rois = 1
        avg_rois_per_series = 1.0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Patients", n_patients)
    with c2:
        st.metric("Series (total)", n_series_total)
    with c3:
        st.metric("Avg series / patient", f"{avg_series_per_patient:.2f}")
    with c4:
        st.metric("Avg ROIs / series", f"{avg_rois_per_series:.2f}")

    c5, c6 = st.columns(2)
    with c5:
        st.metric("Unique ROI names", n_unique_rois)
    with c6:
        st.metric("Total extraction tasks", total_rows)

    with st.expander("📊 Per-patient breakdown", expanded=False):
        agg_args = {}
        if 'series_uid' in dataset_df.columns:
            agg_args['Series'] = ('series_uid', lambda x: x.nunique())
        elif 'series_description' in dataset_df.columns:
            agg_args['Series'] = ('series_description', lambda x: x.nunique())
        else:
            agg_args['Series'] = ('patient_id', 'size')

        if 'roi_name' in dataset_df.columns:
            agg_args['Unique_ROIs'] = ('roi_name', 'nunique')

        agg_args['Tasks'] = ('patient_id', 'size')

        try:
            breakdown = (
                dataset_df.groupby('patient_id')
                .agg(**agg_args)
                .reset_index()
                .rename(columns={'patient_id': 'Patient'})
            )
            st.dataframe(breakdown, use_container_width=True)
        except Exception:
            st.dataframe(dataset_df.head(50), use_container_width=True)


def build_extraction_execution_section():
    """Extraction execution - works with multi-ROI dataset from preprocessing"""
    if not st.session_state.get('pyradiomics_params'):
        st.warning("⚠️ Please generate PyRadiomics parameters first.")
        return

    st.divider()
    st.subheader("🚀 Feature Extraction")

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
            cpu_count = 1

    with col2:
        st.subheader("⚙️ Processing Settings")
        dataset_df = st.session_state.get('dataset_df', pd.DataFrame())
        total_rows = len(dataset_df)
        
        if 'roi_name' in dataset_df.columns:
            unique_rois = dataset_df['roi_name'].nunique()
            st.metric("Dataset Rows (Series × ROI)", total_rows)
            st.info(f"📊 Processing {unique_rois} unique ROI(s) across all series")
        else:
            st.metric("Patients to Process", total_rows)

        use_parallel = st.checkbox("Enable Parallel Processing", value=False)
        if use_parallel and cpu_count > 1:
            n_jobs = st.slider("Number of parallel jobs:", min_value=1, max_value=min(cpu_count, total_rows) if total_rows > 0 else cpu_count, value=min(2, cpu_count))
        else:
            n_jobs = 1

    st.subheader("🏅 IBSI Compliance Settings")
    col1, col2 = st.columns(2)
    with col1:
        enable_ibsi_compliance = st.checkbox("Enable IBSI Compliance", value=True)
    with col2:
        st.info("IBSI compliance ensures standardized feature naming")

    st.markdown("---")

    # ------------------------------------------------------------------
    # Pre-extraction summary: patients, series/patient, ROI/series
    # ------------------------------------------------------------------
    build_pre_extraction_summary()

    if st.button("🔥 Start Feature Extraction", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            st.session_state['ibsi_compliance_enabled'] = enable_ibsi_compliance
            
            df_to_extract = st.session_state.get('dataset_df')
            
            if df_to_extract is None or df_to_extract.empty:
                st.error("❌ No dataset prepared. Complete preprocessing first.")
                return

            required_cols = {'patient_id', 'image_path', 'mask_path'}
            if not required_cols.issubset(set(df_to_extract.columns)):
                st.error(f"❌ Dataset missing required columns.")
                return

            st.info(f"🔄 Starting extraction on {len(df_to_extract)} dataset rows...")

            # Call extraction. Pass the FULL dataset (not just the 3 required
            # columns) so run_extraction() can keep roi_name, series_*, modality,
            # voxel_count, etc. aligned 1:1 with each feature row.
            # The previous code stripped these columns and then re-merged on
            # patient_id alone, which produced an N×N Cartesian product and
            # replicated one (series, ROI)'s features across every ROI.
            features_df = run_extraction(
                dataset_df=df_to_extract.reset_index(drop=True),
                params=st.session_state.get('pyradiomics_params'),
                n_jobs=n_jobs
            )

            if features_df is None or features_df.empty:
                st.error("❌ Feature extraction failed.")
                return

            # Save results
            out_dir = st.session_state.get('output_directory', 'output')
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "radiomic_features_multi_roi.csv")
            features_df.to_csv(out_path, index=False)

            st.success(f"✅ Extraction complete! Saved: {out_path}")
            st.session_state.features_df = features_df
            st.session_state.extraction_done = True

            # Show results
            st.subheader("📊 Extraction Results")
            st.dataframe(features_df.head(200), use_container_width=True)

            csv = features_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Features", csv, "radiomic_features.csv", "text/csv")

            progress_bar.progress(1.0)

        except Exception as e:
            st.error(f"❌ Extraction error: {str(e)}")
            with st.expander("Error Details"):
                st.code(traceback.format_exc())
        finally:
            try:
                progress_bar.empty()
                status_text.empty()
            except Exception:
                pass

# Keep Tab 3 and sidebar unchanged from v1
# Add this complete Tab 3 implementation before main() function (around line 1350)

def build_tab3_analysis():
    """Enhanced Tab 3 with comprehensive statistical analysis - works with multi-ROI data"""
    if not st.session_state.get('extraction_done', False):
        st.warning("⚠️ Please complete the feature extraction step first.")
        return

    st.header("Step 3: Enhanced Statistical Analysis & Feature Selection")
    
    # Check if we have multi-ROI data
    features_df = st.session_state.get('features_df')
    if features_df is not None and not features_df.empty:
        if 'roi_name' in features_df.columns:
            unique_rois = features_df['roi_name'].nunique()
            unique_patients = features_df['patient_id'].nunique() if 'patient_id' in features_df.columns else len(features_df)
            st.info(f"📊 Analyzing {len(features_df)} feature sets from {unique_patients} patients × {unique_rois} ROIs")
        else:
            st.info(f"📊 Analyzing {len(features_df)} patients")

    # Enhanced outcome data section
    build_enhanced_outcome_section()

    # Statistical analysis section
    if st.session_state.get('outcome_df') is not None:
        build_enhanced_statistical_analysis()

def build_enhanced_outcome_section():
    """Enhanced outcome data handling - works with multi-ROI data"""
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
            help="CSV should contain 'PatientID' (or 'patient_id') and outcome columns"
        )

        if uploaded_outcome:
            try:
                outcome_df = pd.read_csv(uploaded_outcome)
                
                # Try to find patient ID column (flexible naming)
                id_col = None
                for col in ['PatientID', 'patient_id', 'Patient_ID', 'ID']:
                    if col in outcome_df.columns:
                        id_col = col
                        break
                
                if id_col is None:
                    st.error("❌ Outcome data must contain a patient ID column (PatientID, patient_id, etc.)")
                    outcome_df = None
                else:
                    # Standardize column name
                    if id_col != 'PatientID':
                        outcome_df = outcome_df.rename(columns={id_col: 'PatientID'})
                    
                    st.success(f"✅ Loaded outcome data with {len(outcome_df)} patients")

                    # Check for matching patients
                    feature_patients = set(st.session_state.features_df['patient_id'].values)
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
        
        # Get unique patients from features
        features_df = st.session_state.features_df
        if 'patient_id' in features_df.columns:
            patients = features_df['patient_id'].unique().tolist()
        else:
            patients = features_df['PatientID'].unique().tolist() if 'PatientID' in features_df.columns else []

        if not patients:
            st.error("Cannot find patient IDs in feature data")
            return

        # Outcome type selection
        col1, col2 = st.columns(2)

        with col1:
            outcome_type = st.selectbox(
                "Select outcome type:",
                ["Binary (0/1)", "Continuous", "Categorical"]
            )

        with col2:
            outcome_name = st.text_input(
                "Outcome variable name:",
                value="Outcome",
                help="Name for your outcome variable"
            )

        st.info(f"📋 Enter outcomes for {len(patients)} unique patients")

        if outcome_type == "Binary (0/1)":
            st.write("Enter binary outcomes (0 or 1) for each patient:")
            outcomes = {}
            
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

        # Create outcome dataset
        if st.button("✅ Create Outcome Dataset"):
            outcome_df = pd.DataFrame({
                'PatientID': patients,
                outcome_name: [outcomes[p] for p in patients]
            })

            st.success("✅ Outcome data created!")
            st.dataframe(outcome_df)

    elif outcome_method == "🔗 Use Existing Data":
        if 'clinical_df' in st.session_state and st.session_state.clinical_df is not None:
            outcome_df = st.session_state.clinical_df
            st.success("✅ Using existing clinical data")
            st.dataframe(outcome_df.head())
        else:
            st.info("ℹ️ No existing outcome data found. Please use another method.")

    if outcome_df is not None:
        st.session_state.outcome_df = outcome_df

    """Enhanced statistical analysis with multiple methods - works with multi-ROI data"""
    st.divider()
    st.subheader("🔬 Enhanced Statistical Analysis")

    # Pull feature & outcome data from session
    features_df = st.session_state.get('features_df')
    outcome_df = st.session_state.get('outcome_df')

    if features_df is None or features_df.empty:
        st.error("No features available. Run feature extraction first.")
        return
    if outcome_df is None or outcome_df.empty:
        st.error("No outcome data available. Provide outcome data in Step 3.")
        return

    # Normalize patient ID column names for robust merging
    # Find patient id column in outcome_df
    possible_id_cols = ['PatientID', 'patient_id', 'Patient_Id', 'Patient_ID', 'ID', 'id']
    outcome_id_col = None
    for c in possible_id_cols:
        if c in outcome_df.columns:
            outcome_id_col = c
            break
    if outcome_id_col is None:
        # fallback: try first column as id if it looks like IDs
        outcome_id_col = outcome_df.columns[0]
        st.warning(f"No obvious patient ID column in outcome data; using '{outcome_id_col}' as ID. If this is wrong, please rename your column to 'PatientID' or 'patient_id'.")

    # Ensure features_df has a patient id column
    feature_id_col = None
    for c in ['patient_id', 'PatientID', 'Patient_Id', 'id', 'ID']:
        if c in features_df.columns:
            feature_id_col = c
            break
    if feature_id_col is None:
        st.error("Features data does not contain a patient identifier column (e.g., 'patient_id' or 'PatientID'). Extraction must output a patient id column.")
        return

    # Standardize column names for merging (create copies to avoid mutating session df)
    features_copy = features_df.copy()
    outcome_copy = outcome_df.copy()

    # Rename columns to standardized names for merge
    features_copy = features_copy.rename(columns={feature_id_col: 'patient_id'})
    if outcome_id_col != 'patient_id':
        outcome_copy = outcome_copy.rename(columns={outcome_id_col: 'patient_id'})

    # Merge features with outcomes on patient_id
    merged_df = pd.merge(features_copy, outcome_copy, on='patient_id', how='inner')
    if merged_df is None or merged_df.empty:
        st.error("No matching patients found between features and outcome data after merge. Check patient IDs.")
        with st.expander("Hints"):
            st.write("• Ensure IDs match exactly between feature and outcome files (case and formatting).")
            st.write(f"• Feature ID column found: '{feature_id_col}'. Outcome ID column used: '{outcome_id_col}'.")
            st.write("• Example mismatch: leading/trailing spaces, different separators, or missing prefixes.")
        return

    st.session_state.merged_df = merged_df

    # Outcome variable selection: only show columns originating from outcome_copy (exclude patient_id)
    outcome_columns = [c for c in outcome_copy.columns if c != 'patient_id']
    if not outcome_columns:
        st.error("Outcome data contains no outcome columns (only patient id detected).")
        return

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
            "🔗 Correlation Analysis"
        ])

        with analysis_tabs[0]:
            build_univariate_analysis_section(merged_df, selected_outcome)

        with analysis_tabs[1]:
            build_lasso_analysis_section(merged_df, selected_outcome)

        with analysis_tabs[2]:
            build_correlation_analysis_section(merged_df, selected_outcome)


def build_univariate_analysis_section(merged_df, selected_outcome):
    """Enhanced univariate analysis with robust checks"""
    st.subheader("📈 Univariate Analysis")

    # Defensive check: ensure selected_outcome column exists in merged_df
    if selected_outcome not in merged_df.columns:
        st.error(f"Selected outcome column '{selected_outcome}' not found in merged dataset.")
        with st.expander("Available columns in merged dataset"):
            st.dataframe(pd.DataFrame({'columns': merged_df.columns.tolist()}))
        st.info("You may need to re-check the outcome data column names or the patient ID merge step.")
        return

    # Ensure outcome is numeric or convert if categorical
    if not pd.api.types.is_numeric_dtype(merged_df[selected_outcome]):
        st.warning(f"Outcome '{selected_outcome}' is not numeric; attempting to factorize/cast for analysis.")
        try:
            merged_df = merged_df.copy()
            merged_df[selected_outcome] = pd.factorize(merged_df[selected_outcome])[0]
            st.success("✅ Converted outcome to numeric codes.")
        except Exception as e:
            st.error(f"Failed to convert outcome to numeric: {e}")
            return

    col1, col2 = st.columns(2)
    with col1:
        p_threshold = st.number_input("P-value threshold:", min_value=0.001, max_value=0.2, value=0.05, step=0.001)
    with col2:
        top_n_features = st.number_input("Number of top features:", min_value=5, max_value=200, value=15)

    # Determine feature columns: exclude merge keys and known metadata
    exclude_cols = {'patient_id', 'PatientID', selected_outcome, 'modality', 'timepoint', 'series_uid', 'roi_name', 'image_path', 'mask_path', 'original_image_path', 'original_mask_path'}
    feature_cols = [col for col in merged_df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(merged_df[col])]
    if not feature_cols:
        st.error("No numeric features found for univariate analysis.")
        return

    if st.button("🔄 Run Univariate Analysis"):
        with st.spinner("Running univariate analysis..."):
            try:
                # Call analysis function (assumes run_univariate_analysis returns (top_df, fig))
                top_features_df, fig = run_univariate_analysis(merged_df, feature_cols, selected_outcome, p_threshold=p_threshold, top_n=top_n_features)

                st.subheader("📊 Top Correlated Features")
                st.dataframe(top_features_df)
                if fig:
                    st.pyplot(fig)

                st.session_state.univariate_results = top_features_df

                csv = top_features_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Results", csv, f"univariate_results_{selected_outcome}.csv", "text/csv")

            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                with st.expander("🔍 Error Details"):
                    st.code(traceback.format_exc())

def build_lasso_analysis_section(merged_df, selected_outcome):
    """Enhanced LASSO feature selection"""
    st.subheader("🎯 LASSO Feature Selection")

    if not pd.api.types.is_numeric_dtype(merged_df[selected_outcome]):
        st.warning(f"⚠️ Converting outcome to numeric...")
        try:
            merged_df = merged_df.copy()
            merged_df[selected_outcome] = pd.factorize(merged_df[selected_outcome])[0]
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            return

    col1, col2 = st.columns(2)
    with col1:
        cv_folds = st.number_input("Cross-validation folds:", min_value=3, max_value=10, value=5)
    with col2:
        max_features = st.number_input("Maximum features:", min_value=1, max_value=50, value=10)

    if st.button("🔄 Run LASSO Selection"):
        with st.spinner("Running LASSO..."):
            try:
                feature_cols = [col for col in merged_df.columns
                              if col not in ['PatientID', 'patient_id', selected_outcome, 'modality',
                                           'timepoint', 'series_uid', 'roi_voxel_count', 'roi_percentage',
                                           'extraction_timestamp', 'roi_name']]

                numeric_feature_cols = [col for col in feature_cols
                                      if pd.api.types.is_numeric_dtype(merged_df[col])]

                if not numeric_feature_cols:
                    st.error("❌ No numeric features found")
                    return

                selected_features, fig = run_lasso_selection(merged_df, numeric_feature_cols, selected_outcome)

                if selected_features:
                    st.success(f"✅ LASSO selected {len(selected_features)} features")
                    st.session_state.lasso_features = selected_features

                    features_df = pd.DataFrame([
                        {'Feature': feature, 'Coefficient': coef}
                        for feature, coef in selected_features.items()
                    ]).sort_values('Coefficient', key=abs, ascending=False)

                    st.dataframe(features_df, use_container_width=True)

                    if fig:
                        st.pyplot(fig)

                    csv = features_df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Results", csv, f"lasso_features_{selected_outcome}.csv", "text/csv")
                else:
                    st.warning("⚠️ LASSO did not select any features")

            except Exception as e:
                st.error(f"❌ LASSO failed: {str(e)}")
                with st.expander("🔍 Error Details"):
                    st.code(traceback.format_exc())

def build_correlation_analysis_section(merged_df, selected_outcome):
    """Enhanced correlation analysis"""
    st.subheader("🔗 Correlation Analysis")

    col1, col2 = st.columns(2)
    with col1:
        correlation_method = st.selectbox("Correlation method:", ["pearson", "spearman", "kendall"])
    with col2:
        feature_selection = st.selectbox("Features for heatmap:", 
                                        ["Top LASSO features", "Top Univariate features", "All features"])

    if feature_selection == "Top LASSO features":
        if st.session_state.get('lasso_features'):
            features_for_analysis = list(st.session_state.lasso_features.keys())
        else:
            st.warning("⚠️ No LASSO features available")
            features_for_analysis = []
    elif feature_selection == "Top Univariate features":
        if st.session_state.get('univariate_results') is not None:
            features_for_analysis = st.session_state.univariate_results['Feature'].tolist()
        else:
            st.warning("⚠️ No univariate results available")
            features_for_analysis = []
    else:
        feature_cols = [col for col in merged_df.columns
                       if col not in ['PatientID', 'patient_id', selected_outcome, 'modality',
                                    'timepoint', 'series_uid', 'roi_voxel_count', 'roi_percentage',
                                    'extraction_timestamp', 'roi_name']]
        features_for_analysis = feature_cols

    if features_for_analysis and st.button("🔄 Generate Correlation Heatmap"):
        with st.spinner("Generating heatmap..."):
            try:
                columns_for_correlation = features_for_analysis + [selected_outcome]
                heatmap_fig = generate_correlation_heatmap(merged_df, columns_for_correlation)

                if heatmap_fig:
                    st.subheader("🔗 Feature Correlation Heatmap")
                    st.pyplot(heatmap_fig)
                    st.session_state.correlation_heatmap = heatmap_fig

            except Exception as e:
                st.error(f"❌ Correlation failed: {str(e)}")
                with st.expander("🔍 Error Details"):
                    st.code(traceback.format_exc())

def build_sidebar():
    """Enhanced sidebar - keep v1 code"""
    st.sidebar.title("🔬 Enhanced RadiomicsGUI")
    st.sidebar.write("Advanced Multi-Modal Radiomics Analysis Platform")
    st.sidebar.write("*With Multi-ROI & IBSI Support*")
    
    st.sidebar.divider()
    
    try:
        resource_info = check_system_resources()
        if isinstance(resource_info, dict):
            ram_gb = resource_info.get('available_ram_gb', 0)
            st.sidebar.metric("Available RAM", f"{ram_gb:.1f} GB")
    except Exception:
        pass
    
    st.sidebar.divider()
    
    st.sidebar.subheader("📈 Progress")
    data_uploaded = st.session_state.get('uploaded_data_path') is not None
    preprocessing_done = st.session_state.get('preprocessing_done', False)
    extraction_done = st.session_state.get('extraction_done', False)
    
    for item_name, completed in [("Data Upload", data_uploaded), ("Pre-processing", preprocessing_done), ("Feature Extraction", extraction_done)]:
        status_icon = "✅" if completed else "⏳"
        st.sidebar.write(f"{status_icon} {item_name}")

def main():
    """Main application entry point"""
    st.set_page_config(page_title="Enhanced RadiomicsGUI", page_icon="🔬", layout="wide")

    try:
        initialize_session_state()
        register_cleanup()
    except:
        pass

    build_sidebar()

    st.title("🔬 Enhanced RadiomicsGUI - Multi-ROI Analysis Platform")
    st.markdown("*Process multiple series × multiple ROIs without workflow changes*")
    st.divider()

    tab1,
# Continue from line 1404 in ui.py

def main():
    """Main application entry point"""
    st.set_page_config(
        page_title="Enhanced RadiomicsGUI",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    try:
        initialize_session_state()
        register_cleanup()
    except ImportError:
        st.error("Failed to initialize session state")
        return

    build_sidebar()

    st.title("🔬 Enhanced RadiomicsGUI - Multi-Series + Multi-ROI Analysis Platform")
    st.markdown("*Comprehensive DICOM & NIfTI support with IBSI compliance and multi-ROI processing*")
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
