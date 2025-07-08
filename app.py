# app.py
import streamlit as st
import sys
import os

# Add current directory to Python path for Streamlit Cloud
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from the ui.py file (not ui package)
from ui import (
    build_tab1_data_upload,
    build_tab2_feature_extraction,
    build_tab3_analysis,
    build_sidebar
)
from utils import initialize_session_state, register_cleanup

def main():
    st.set_page_config(layout="wide", page_title="Radiomics Pipeline", page_icon="🏥")
    initialize_session_state()
    register_cleanup()
    
    st.title("🏥 End-to-End Radiomics Pipeline")
    st.markdown("An integrated tool for DICOM data processing, feature extraction, and clinical analysis.")
    st.divider()
    
    build_sidebar()
    
    tab1, tab2, tab3 = st.tabs([
        "📤 **Step 1: Data & Pre-processing**",
        "⚙️ **Step 2: Feature Extraction**",
        "📊 **Step 3: Analysis & Correlation**"
    ])
    
    with tab1: 
        build_tab1_data_upload()
    with tab2: 
        build_tab2_feature_extraction()
    with tab3: 
        build_tab3_analysis()

if __name__ == "__main__":
    main()
