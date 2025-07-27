"""
Medical Report Explainer - Main Streamlit Application
A comprehensive tool for explaining medical reports using RAG and LLM
"""

import streamlit as st
import uuid
from pathlib import Path
import pandas as pd

# Configure page
st.set_page_config(
    page_title="🏥 Medical Report Explainer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import our modules
from config.config import (
    PAGE_TITLE, DISCLAIMER, READING_LEVELS, GEMINI_API_KEY
)
from src.document_processor import DocumentProcessor, create_sample_medical_text
from src.embeddings import initialize_embeddings_system
from src.llm_integration import MedicalExplainer, create_medical_disclaimer
from src.data_ingestion import create_medical_corpus
from utils.feedback import render_feedback_widget, display_feedback_dashboard


def initialize_session_state():
    """Initialize session state variables"""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if "medical_text" not in st.session_state:
        st.session_state.medical_text = ""
    
    if "extracted_sections" not in st.session_state:
        st.session_state.extracted_sections = {}
    
    if "complex_terms" not in st.session_state:
        st.session_state.complex_terms = []
    
    if "embeddings_initialized" not in st.session_state:
        st.session_state.embeddings_initialized = False


def render_sidebar():
    """Render sidebar with settings and information"""
    with st.sidebar:
        st.title("🏥 Medical Report Explainer")
        
        # API Key status
        if GEMINI_API_KEY:
            st.success("✅ Gemini API configured")
        else:
            st.error("❌ Gemini API key not set")
            st.info("Set GEMINI_API_KEY environment variable")
        
        st.markdown("---")
        
        # Reading level selection
        st.subheader("⚙️ Settings")
        reading_level = st.selectbox(
            "Select Reading Level:",
            options=list(READING_LEVELS.keys()),
            index=1,  # Default to "8th-grade"
            help="Choose how complex the explanations should be"
        )
        
        # Enable tooltips
        enable_tooltips = st.checkbox(
            "Enable term tooltips",
            value=True,
            help="Show simple explanations when hovering over complex terms"
        )
        
        # Features toggle
        st.subheader("🔧 Features")
        show_sections = st.checkbox("Show report sections", value=True)
        show_complex_terms = st.checkbox("Highlight complex terms", value=True)
        
        st.markdown("---")
        
        # System status
        st.subheader("📊 System Status")
        
        # Check embeddings system
        if not st.session_state.embeddings_initialized:
            with st.spinner("Checking embeddings system..."):
                success, message = initialize_embeddings_system()
                st.session_state.embeddings_initialized = success
                if success:
                    st.success("✅ Vector store ready")
                else:
                    st.warning("⚠️ Vector store not ready")
                    st.info(message)
        else:
            st.success("✅ Vector store ready")
        
        # Data management
        st.subheader("🗃️ Data Management")
        if st.button("Download Medical Corpus"):
            with st.spinner("Downloading medical corpus..."):
                try:
                    corpus = create_medical_corpus()
                    st.success(f"Downloaded {corpus['total_articles']} articles")
                    # Reinitialize embeddings
                    st.session_state.embeddings_initialized = False
                except Exception as e:
                    st.error(f"Error downloading corpus: {e}")
        
        # Information
        st.markdown("---")
        st.subheader("ℹ️ About")
        st.info("""
        This tool helps explain medical reports in simple language.
        
        **Features:**
        - Upload PDF/TXT/DOCX files
        - Get explanations at different reading levels
        - Ask questions about medical terms
        - Generate summaries
        
        **Data Sources:**
        - PubMed medical literature
        - Mayo Clinic articles
        """)
        
        return reading_level, enable_tooltips, show_sections, show_complex_terms


def render_file_upload():
    """Render file upload section"""
    st.header("📄 Upload Medical Report")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a medical report file",
            type=["pdf", "txt", "docx"],
            help="Upload a PDF, TXT, or DOCX file containing a medical report"
        )
        
        if uploaded_file is not None:
            # Process the uploaded file
            processor = DocumentProcessor()
            
            with st.spinner("Extracting text from file..."):
                result = processor.extract_text(uploaded_file)
            
            if result["success"]:
                st.success(f"✅ Text extracted successfully!")
                
                # Show file info - handle metadata safely
                metadata = result.get("metadata", {})
                if metadata and 'name' in metadata and 'size' in metadata:
                    st.info(f"📁 {metadata['name']} ({metadata['size']:,} bytes)")
                else:
                    st.info(f"📁 {uploaded_file.name} ({getattr(uploaded_file, 'size', 0):,} bytes)")
                
                # Store extracted text
                extracted_text = result.get("text", "")
                if extracted_text:
                    st.session_state.medical_text = processor.preprocess_medical_text(extracted_text)
                    
                    # Extract sections
                    st.session_state.extracted_sections = processor.extract_medical_sections(
                        st.session_state.medical_text
                    )
                    
                    # Preview text
                    with st.expander("📖 Preview extracted text"):
                        preview_text = st.session_state.medical_text
                        if len(preview_text) > 1000:
                            preview_text = preview_text[:1000] + "..."
                        
                        st.text_area(
                            "Extracted text:",
                            value=preview_text,
                            height=200,
                            disabled=True
                        )
                else:
                    st.warning("⚠️ No text was extracted from the file. Please check if the file contains readable text.")
                    st.session_state.medical_text = ""
                    st.session_state.extracted_sections = {}
            else:
                st.error("❌ Failed to extract text from file")
                errors = result.get("errors", ["Unknown error occurred"])
                for error in errors:
                    st.error(f"Error: {error}")
    
    with col2:
        st.subheader("🧪 Try Sample Report")
        if st.button("Load Sample Medical Report"):
            st.session_state.medical_text = create_sample_medical_text()
            processor = DocumentProcessor()
            st.session_state.extracted_sections = processor.extract_medical_sections(
                st.session_state.medical_text
            )
            st.success("✅ Sample report loaded!")
            st.rerun()


def render_medical_sections(show_sections: bool):
    """Render extracted medical sections"""
    if not show_sections or not st.session_state.extracted_sections:
        return
    
    st.header("📋 Report Sections")
    
    sections = st.session_state.extracted_sections
    
    # Create tabs for different sections
    section_names = [name.replace("_", " ").title() for name in sections.keys() if sections[name].strip()]
    
    if section_names:
        tabs = st.tabs(section_names)
        
        section_keys = [key for key in sections.keys() if sections[key].strip()]
        
        for tab, section_key in zip(tabs, section_keys):
            with tab:
                st.text_area(
                    f"{section_key.replace('_', ' ').title()} Content:",
                    value=sections[section_key],
                    height=150,
                    disabled=True
                )


def render_complex_terms(show_complex_terms: bool, enable_tooltips: bool):
    """Render complex terms identification"""
    if not show_complex_terms or not st.session_state.medical_text:
        return
    
    st.header("🔍 Complex Medical Terms")
    
    # Initialize medical explainer
    explainer = MedicalExplainer()
    
    # Identify complex terms
    complex_terms = explainer.identify_complex_terms(st.session_state.medical_text)
    st.session_state.complex_terms = complex_terms
    
    if complex_terms:
        st.subheader("Found complex terms in your report:")
        
        for i, term_info in enumerate(complex_terms):
            with st.expander(f"🏷️ {term_info['term'].title()}", expanded=False):
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.write(f"**Term:** {term_info['term']}")
                    st.write(f"**Simple explanation:** {term_info['simple_explanation']}")
                
                with col2:
                    if st.button(f"Get detailed explanation", key=f"explain_{i}"):
                        reading_level = st.session_state.get("reading_level", "8th-grade")
                        with st.spinner("Getting detailed explanation..."):
                            explanation = explainer.explain_medical_term(
                                term_info['term'],
                                reading_level=reading_level,
                                context=st.session_state.medical_text[:500]
                            )
                            st.write("**Detailed Explanation:**")
                            st.write(explanation)
                            
                            # Add feedback widget
                            render_feedback_widget(
                                feedback_type="explanation",
                                query=term_info['term'],
                                response=explanation
                            )
    else:
        st.info("No complex terms detected in the uploaded report.")


def render_summary_generation(reading_level: str):
    """Render summary generation section"""
    if not st.session_state.medical_text:
        return
    
    st.header("📝 Generate Summary")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if st.button("Generate Summary", type="primary"):
            explainer = MedicalExplainer()
            
            with st.spinner("Generating summary..."):
                summary = explainer.generate_summary(
                    st.session_state.medical_text,
                    reading_level=reading_level
                )
                
                # Store in session state
                st.session_state.generated_summary = summary
    
    # Display summary if available
    if hasattr(st.session_state, 'generated_summary'):
        with col1:
            st.subheader(f"Summary ({reading_level} level)")
            st.write(st.session_state.generated_summary)
            
            # Add feedback widget
            render_feedback_widget(
                feedback_type="summary",
                query=f"Summary at {reading_level} level",
                response=st.session_state.generated_summary
            )


def render_qa_section(reading_level: str):
    """Render Q&A section"""
    st.header("❓ Ask Questions")
    
    # Initialize session state for selected question
    if "selected_question" not in st.session_state:
        st.session_state.selected_question = ""
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_question = st.text_input(
            "Ask a question about the medical report:",
            value=st.session_state.selected_question,
            placeholder="e.g., What does hypertension mean? What are the next steps?",
            help="Ask about medical terms, conditions, or treatments mentioned in the report",
            key="question_input"
        )
        
        # Update session state when user types
        if user_question != st.session_state.selected_question:
            st.session_state.selected_question = user_question
    
    with col2:
        ask_button = st.button("Ask Question", type="primary")
    
    if ask_button and user_question:
        explainer = MedicalExplainer()
        
        with st.spinner("Finding answer..."):
            answer = explainer.answer_medical_question(
                question=user_question,
                medical_text=st.session_state.medical_text,
                reading_level=reading_level
            )
            
            st.subheader("💡 Answer")
            st.write(answer)
            
            # Add feedback widget
            render_feedback_widget(
                feedback_type="question",
                query=user_question,
                response=answer
            )
    
    # Show example questions
    st.subheader("💭 Example Questions")
    st.caption("Click on any question below to automatically fill the input box:")
    
    example_questions = [
        "What does this diagnosis mean?",
        "What are the treatment options mentioned?", 
        "Should I be worried about these findings?",
        "What do these test results indicate?",
        "What should I ask my doctor about this report?"
    ]
    
    # Display example questions in a more compact layout
    cols = st.columns(2)  # Use 2 columns instead of 5 for better mobile view
    
    for i, question in enumerate(example_questions):
        col_idx = i % 2  # Alternate between columns
        with cols[col_idx]:
            if st.button(f"💬 {question}", key=f"example_q_{i}", use_container_width=True):
                st.session_state.selected_question = question
                st.rerun()


def render_admin_panel():
    """Render admin panel for feedback and system management"""
    st.header("🔧 Admin Panel")
    
    tabs = st.tabs(["Feedback Dashboard", "System Status", "Data Management"])
    
    with tabs[0]:
        display_feedback_dashboard()
    
    with tabs[1]:
        st.subheader("System Health")
        
        # Check various system components
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("API Status", "✅ Active" if GEMINI_API_KEY else "❌ Inactive")
        
        with col2:
            st.metric("Vector Store", "✅ Ready" if st.session_state.embeddings_initialized else "❌ Not Ready")
        
        with col3:
            # Calculate storage usage
            data_dir = Path("data")
            if data_dir.exists():
                total_size = sum(f.stat().st_size for f in data_dir.rglob('*') if f.is_file())
                st.metric("Storage Used", f"{total_size / (1024*1024):.1f} MB")
            else:
                st.metric("Storage Used", "0 MB")
    
    with tabs[2]:
        st.subheader("Data Operations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Rebuild Vector Store"):
                with st.spinner("Rebuilding vector store..."):
                    success, message = initialize_embeddings_system()
                    if success:
                        st.success("Vector store rebuilt successfully!")
                        st.session_state.embeddings_initialized = True
                    else:
                        st.error(f"Failed to rebuild: {message}")
        
        with col2:
            if st.button("Clear Feedback Data"):
                if st.checkbox("Confirm deletion"):
                    # Clear feedback file
                    from config.config import FEEDBACK_FILE
                    if FEEDBACK_FILE.exists():
                        FEEDBACK_FILE.unlink()
                        st.success("Feedback data cleared!")


def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar and get settings
    reading_level, enable_tooltips, show_sections, show_complex_terms = render_sidebar()
    st.session_state.reading_level = reading_level
    
    # Main content area
    st.title("🏥 Medical Report Explainer")
    
    # Medical disclaimer
    with st.expander("⚠️ Important Medical Disclaimer - Please Read", expanded=False):
        st.markdown(create_medical_disclaimer())
    
    # Check if we have medical text
    if not st.session_state.medical_text:
        render_file_upload()
    else:
        # Show current report info
        st.success("📄 Medical report loaded and ready for analysis!")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.info(f"📊 Report length: {len(st.session_state.medical_text)} characters")
        with col2:
            if st.button("Load Different Report"):
                # Clear current report
                st.session_state.medical_text = ""
                st.session_state.extracted_sections = {}
                st.session_state.complex_terms = []
                st.rerun()
        with col3:
            if st.button("Download Report Text"):
                st.download_button(
                    label="Download as TXT",
                    data=st.session_state.medical_text,
                    file_name="medical_report.txt",
                    mime="text/plain"
                )
        
        # Main features
        st.markdown("---")
        
        # Render different sections based on settings
        render_medical_sections(show_sections)
        render_complex_terms(show_complex_terms, enable_tooltips)
        render_summary_generation(reading_level)
        render_qa_section(reading_level)
    
    # Admin panel (hidden by default)
    with st.expander("🔧 Admin Panel", expanded=False):
        render_admin_panel()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
    🏥 Medical Report Explainer | Built with Streamlit, LangChain, and Google Gemini<br>
    For educational purposes only. Always consult healthcare professionals for medical advice.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main() 
