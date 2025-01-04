# RAG with PDF Highlighting

## Demo

<video width="600" controls autoplay>
    <source src="demo-v1.mp4" type="video/mp4">
    Your browser does not support the video tag.
</video>

## Prerequisites
- Python 3.8+
- Git
- pip

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://bitbucket.org/v4c_work/gen_ai_hack4ce.git
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Create .env file
Paste these commands in the `.env` file and enter the openai api key and langchain api key.
```bash
OPENAI_API_KEY=<openai-api-key>
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY=<langchain-api-key>
LANGCHAIN_PROJECT=<project-name>
```

### 4. Optional: Upload PDFs
Place your PDF documents in the `pdf-docs` folder.

### 5. Create Vector Database
```bash
python create_vectordb.py
```
This command processes the PDFs and creates a vector database for semantic search.

### 6. Start PDF File Server
```bash
python -m http.server 8003
```
Opens a local server to display PDF files.

### 7. Run Streamlit Application
```bash
streamlit run app.py
```
Launch the interactive question-answering interface.

## Usage
After following the setup steps, navigate to the Streamlit app in your browser and start asking questions about your uploaded PDFs.

## Troubleshooting
- Ensure all dependencies are installed correctly
- Check that PDFs are in the correct folder
- Verify Python version compatibility