# Tab 1 Redesign - V2 Features

## Overview
This document describes the v2 redesign of Tab 1 in the Radiomics Pipeline application, implementing enhanced workflow management and directory analysis capabilities.

## New Features

### 1. Simplified Tab 1 Structure
- **Separate workflow tracks**: Users can now choose between single patient and multiple patients analysis
- **Clear workflow separation**: Intuitive interface design with dedicated sections for each workflow type
- **Improved navigation**: Step-by-step guidance through the data processing pipeline

### 2. Enhanced Directory Analysis
When users provide a directory path or zip file, the system now automatically analyzes and displays:
- **Patient count**: Total number of patients in the directory
- **Imaging series breakdown**: Associated imaging series for each patient (CT, PET/CT, MRI, CBCT)
- **RTSTRUCT availability**: Number of RTSTRUCT files available for radiomic feature extraction
- **Compatibility checking**: Automatic validation between imaging series and RTSTRUCT files

### 3. Interactive Processing Control
- **Detailed information toggle**: Checkbox to show/hide further processing details
- **Patient-specific breakdown**: Eligible pairs (Imaging series and RTSTRUCT file) for each patient
- **Processing readiness indicators**: Clear indication of which patients are ready for analysis

### 4. Improved User Experience
- **Workflow-specific interfaces**: Tailored UI for single vs multiple patient workflows
- **Progress indicators**: Visual feedback during directory analysis
- **Comprehensive reporting**: Detailed patient-by-patient breakdown of available data
- **Error handling**: Clear error messages and guidance for data issues

## Technical Implementation

### Modified Files

#### `processing.py`
- `analyze_directory_structure()`: Comprehensive directory analysis with patient/series breakdown
- `get_single_patient_info()`: Single patient data analysis and compatibility checking
- `check_imaging_rtstruct_compatibility()`: Validates imaging series and RTSTRUCT compatibility

#### `ui.py`
- `build_tab1_data_upload()`: Main redesigned interface with workflow selection
- `build_single_patient_workflow()`: Single patient workflow management
- `build_multiple_patients_workflow()`: Multiple patients workflow management
- `display_single_patient_analysis()`: Single patient results visualization
- `display_directory_analysis()`: Directory analysis results display
- `continue_single_patient_processing()`: Single patient processing continuation
- `continue_multiple_patients_processing()`: Multiple patients processing continuation

#### `utils.py`
- `format_patient_summary()`: Patient information formatting helper
- `calculate_processing_readiness()`: Processing readiness statistics calculation
- Enhanced session state management with new workflow-specific keys

## Usage Guide

### Single Patient Workflow
1. Select "👤 Single Patient Analysis" from the workflow options
2. Upload files or select directory path
3. Click "🔄 Process Single Patient Data"
4. Review patient analysis results
5. Check "Show detailed processing information" for comprehensive breakdown
6. Select imaging/RTSTRUCT pair and target contour
7. Generate NIfTI files for processing

### Multiple Patients Workflow
1. Select "👥 Multiple Patients Analysis" from the workflow options
2. Upload files or select directory path
3. Click "🔄 Process Multiple Patients Data"
4. Review directory analysis results including patient count and compatibility
5. Check "Show detailed processing information for each patient" for full breakdown
6. Select imaging modality and target contour for batch processing
7. Generate NIfTI files for all eligible patients

## Key Benefits

### For Single Patient Analysis
- **Focused interface**: Streamlined workflow for individual patient processing
- **Detailed compatibility info**: Clear indication of available imaging/RTSTRUCT pairs
- **Flexible selection**: Choose specific pairs and contours for analysis

### For Multiple Patients Analysis
- **Comprehensive overview**: Full directory structure analysis before processing
- **Batch processing readiness**: Clear indication of which patients are ready
- **Modality-specific processing**: Process all patients with same modality/contour combination
- **Processing statistics**: Detailed breakdown of data availability across patients

## Backward Compatibility
- All existing functionality is preserved
- Previous workflows continue to work as before
- New features are additive and don't break existing usage patterns
- Session state management enhanced without breaking existing keys

## Error Handling
- **Directory validation**: Comprehensive path and content validation
- **Compatibility checking**: Automatic validation of imaging/RTSTRUCT pairs
- **Clear error messages**: Descriptive error reporting with actionable guidance
- **Graceful degradation**: System continues to function even with partial data issues