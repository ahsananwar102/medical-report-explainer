# 🏥 Medical Report Explainer

A comprehensive web application that helps users understand medical reports through AI-powered explanations, summaries, and Q&A functionality.

## ✨ Features

- **📄 Document Upload**: Support for PDF, TXT, and DOCX medical reports
- **🔍 Text Extraction**: Advanced extraction with fallback methods for complex documents  
- **📚 RAG System**: Retrieval-Augmented Generation using medical corpus (PubMed + Mayo Clinic)
- **🧠 AI Explanations**: Google Gemini-powered explanations at different reading levels
- **📝 Smart Summaries**: Generate patient-friendly summaries
- **❓ Interactive Q&A**: Ask questions about medical terms and conditions
- **🏷️ Term Highlighting**: Identify and explain complex medical terminology
- **👍 Feedback System**: User feedback collection and analytics
- **⚠️ Medical Disclaimer**: Prominent medical disclaimers and safety warnings

## 🛠️ Tech Stack

- **Frontend**: Streamlit (deployed on Hugging Face Spaces)
- **Backend**: Python 3.10+
- **Vector Store**: FAISS for similarity search
- **Embeddings**: Sentence-BERT (all-MiniLM-L6-v2)
- **LLM**: Google Gemini API (free tier)
- **Document Processing**: PyMuPDF, pdfplumber, python-docx
- **Data Sources**: PubMed (via Entrez API), Mayo Clinic articles
- **Optional NER**: spaCy with medical models

## 📦 Installation

### Local Development

1. **Clone the repository**
```bash
git clone <repository-url>
cd medreport_explainer
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
# Create .env file or set environment variables
export GEMINI_API_KEY="your_gemini_api_key_here"
```

5. **Download medical corpus (optional)**
```bash
python src/data_ingestion.py
```

6. **Run the application**
```bash
streamlit run streamlit_app.py
```

### Getting Google Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Create a new API key
4. Copy the API key and set it as an environment variable

## 🚀 Deployment

### Hugging Face Spaces

This application is configured for deployment on Hugging Face Spaces:

1. **Create a new Space**
   - Go to [Hugging Face Spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Choose "Streamlit" as the SDK
   - Set visibility (public/private)

2. **Upload files**
   - Upload all project files to your Space
   - Ensure `app.py` is the entry point

3. **Set environment variables**
   - In your Space settings, add:
     - `GEMINI_API_KEY`: Your Google Gemini API key

4. **Configure Space**
   - The Space will automatically use `requirements.txt`
   - Build time may take 5-10 minutes due to dependencies

### Alternative Deployment Options

#### Streamlit Cloud
```bash
# Push to GitHub and connect to Streamlit Cloud
# Add secrets in Streamlit Cloud dashboard:
# GEMINI_API_KEY = "your_api_key"
```

#### Docker
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 📁 Project Structure

```
medreport_explainer/
├── streamlit_app.py           # Main Streamlit application
├── app.py                     # Entry point for Hugging Face Spaces
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── config/
│   ├── __init__.py
│   └── config.py             # Configuration settings
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py     # PubMed & Mayo Clinic data download
│   ├── document_processor.py # PDF/TXT/DOCX text extraction
│   ├── embeddings.py         # FAISS vector store & embeddings
│   └── llm_integration.py    # Google Gemini integration
├── utils/
│   ├── __init__.py
│   └── feedback.py           # User feedback management
└── data/                     # Data storage (created automatically)
    ├── corpus/               # Medical corpus storage
    ├── embeddings/           # FAISS index files
    └── feedback.json         # User feedback data
```

## 🔧 Configuration

Key configuration options in `config/config.py`:

```python
# API Settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-1.5-flash"  # Free model

# Embedding Settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# File Processing
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
SUPPORTED_FORMATS = [".pdf", ".txt", ".docx"]
```

## 📚 Usage

### Basic Workflow

1. **Upload a medical report** (PDF, TXT, or DOCX)
2. **Choose reading level** (12-year-old or 8th-grade)
3. **Review extracted sections** and highlighted complex terms
4. **Generate summary** for easy understanding
5. **Ask questions** about specific terms or conditions
6. **Provide feedback** to improve the system

### Example Queries

- "What does hypertension mean?"
- "Explain this diagnosis in simple terms"
- "What are the treatment options mentioned?"
- "Should I be worried about these findings?"

### Reading Levels

- **12-year-old**: Very simple language, short sentences, no medical jargon
- **8th-grade**: Clear language with medical terms explained in parentheses

## 🛡️ Safety & Disclaimers

This application includes comprehensive medical disclaimers and safety warnings:

- ⚠️ **Educational purposes only** - Not medical advice
- 👨‍⚕️ **Always consult healthcare professionals** for medical decisions
- 🚨 **Emergency situations** - Contact emergency services immediately
- 📋 **Information limitations** - May not apply to specific situations

## 🔍 Data Sources

### Medical Corpus
- **PubMed**: Medical literature abstracts via Entrez API
- **Mayo Clinic**: Disease and condition articles
- **Processing**: Chunked and embedded for RAG retrieval

### Privacy
- No patient data is stored permanently
- Feedback is stored locally and anonymized
- API calls to Gemini follow Google's privacy policies

## 📊 Monitoring & Feedback

### Built-in Analytics
- User feedback collection (👍/👎)
- Feature usage statistics
- System performance monitoring
- Export capabilities for analysis

### Feedback Dashboard
- Overall satisfaction rates
- Feature-specific feedback
- Recent user comments
- CSV export functionality

## 🐛 Troubleshooting

### Common Issues

1. **API Key Error**
   ```
   Error: GEMINI_API_KEY environment variable not set
   ```
   **Solution**: Set your Gemini API key in environment variables

2. **Vector Store Not Ready**
   ```
   Warning: Vector store not ready
   ```
   **Solution**: Download medical corpus first or rebuild vector store

3. **File Upload Issues**
   ```
   Error: Unsupported format
   ```
   **Solution**: Use PDF, TXT, or DOCX files under 10MB

4. **Memory Issues**
   ```
   StreamlitAPIException: Memory limit exceeded
   ```
   **Solution**: Reduce file size or chunk size in configuration

### Debug Mode

Enable debug logging by setting:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include unit tests for new features
- Update README for new functionality

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Google Gemini** for AI capabilities
- **Hugging Face** for hosting and transformers
- **PubMed/NCBI** for medical literature access
- **Mayo Clinic** for patient education resources
- **Streamlit** for the web framework
- **LangChain** for RAG architecture

## 📞 Support

For issues and questions:
- 🐛 **Bug Reports**: Open GitHub issues
- 💡 **Feature Requests**: Create enhancement issues  
- 📧 **Contact**: [Your contact information]
- 📖 **Documentation**: Check this README and inline docstrings

---

**⚠️ Important**: This tool is for educational purposes only. Always consult qualified healthcare professionals for medical advice, diagnosis, and treatment decisions. 