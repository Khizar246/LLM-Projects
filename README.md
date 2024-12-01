# News Research Bot 📈

A powerful Streamlit-based web application that processes news articles, generates embeddings, and answers user queries using a robust pipeline of NLP tools.

## Features
- **Dynamic URL Input**: Users can input multiple article URLs to process.
- **Content Extraction**: Fetches and parses article content using `newspaper3k`.
- **Text Chunking**: Splits large texts into manageable chunks for processing.
- **Vector Store Creation**: Leverages FAISS to create a vector store of embeddings for document retrieval.
- **Question Answering**: Provides precise answers to user queries using a pretrained QA model (`deepset/roberta-base-squad2`).
- **Source Highlighting**: Displays relevant paragraphs and source URLs for better traceability.

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Khizar246/LLM-Projects.git
cd LLM-Projects

### 2. Set Up Virtual Environment (Optional)
Create and activate a virtual environment to isolate dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
Install all required Python libraries:
```bash
pip install -r requirements.txt
```

# Streamlit for creating the user interface
streamlit==1.25.0  # Latest stable version for compatibility

# Newspaper3k for extracting article content
newspaper3k==0.2.8  # Ensure compatibility with the code

# LangChain dependencies for embeddings and vector stores
langchain-community==0.0.5  # Include the latest compatible version for LangChain community modules
langchain-huggingface==0.0.5  # For HuggingFace embedding models

# FAISS for vector stores
faiss-cpu==1.7.4  # CPU version of FAISS (change to faiss-gpu if GPU acceleration is needed)

# Transformers for question-answering models
transformers==4.35.0  # Latest stable version

# PyTorch for deep learning backend
torch==2.0.1+cu118  # Compatible with modern CUDA versions (adjust for specific GPU setup if needed)

# NLTK for text processing
nltk==3.10  # Include for handling text tokenization and splitting

# Other necessary packages
numpy==1.24.4  # FAISS and transformers rely on NumPy
scipy==1.11.2  # Required for FAISS

---

## Usage

### 1. Run the Application
Start the Streamlit server:
```bash
streamlit run app.py
```

### 2. Input URLs
- Enter the number of URLs to process in the sidebar.
- Paste article URLs into the input fields.

### 3. Process Articles
- Click the "Process URLs" button to fetch and process the content.
- The application generates embeddings and creates a FAISS vector store for document retrieval.

### 4. Ask Questions
- Type your question in the main input box.
- The application retrieves the most relevant documents, processes your query, and provides:
  - The best possible answer.
  - A summarized paragraph for context.
  - Source URLs.

---

## Requirements
- Python 3.8 or later
- Dependencies listed in `requirements.txt`

---

## How It Works

1. **Article Parsing**:
   - Extracts text from news articles using `newspaper3k`.

2. **Text Processing**:
   - Splits text into chunks using `RecursiveCharacterTextSplitter` from LangChain.

3. **Embedding Generation**:
   - Embeddings are generated using `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace).

4. **Vector Store**:
   - Creates and stores a FAISS index for efficient document retrieval.

5. **Question Answering**:
   - Uses a pretrained QA pipeline (`deepset/roberta-base-squad2`) from HuggingFace.

6. **Display Results**:
   - Presents answers, summaries, and relevant sources in a user-friendly Streamlit interface.

---

## Folder Structure
```
.
├── app.py              # Main application file
├── requirements.txt    # Dependencies
├── README.md           # Project documentation
└── faiss_store_langchain # Folder for FAISS vector store (auto-generated)
```

---

## Troubleshooting

### Common Issues
1. **NLTK Data Not Found**:
   - If the `punkt` tokenizer isn't downloaded automatically, run:
     ```python
     import nltk
     nltk.download('punkt')
     ```

2. **GPU Issues**:
   - Ensure PyTorch and FAISS are configured for GPU use if needed.

3. **Error Fetching Content**:
   - Verify that the URLs are correct and publicly accessible.
