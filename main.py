import streamlit as st
from newspaper import Article
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
import nltk
import os
import logging

# Suppress unnecessary logs from libraries
logging.getLogger("newspaper").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("faiss").setLevel(logging.WARNING)

# Download NLTK data files if not already downloaded
nltk_data_path = os.path.join(os.path.expanduser('~'), 'nltk_data')
if not os.path.exists(nltk_data_path):
    nltk.download('punkt', download_dir=nltk_data_path)
else:
    nltk.data.path.append(nltk_data_path)

# Streamlit UI
st.title("News Research Bot 📈")
st.sidebar.title("News Article URLs")

# Dynamic URL input fields
num_urls = st.sidebar.number_input("Number of URLs", min_value=1, max_value=10, value=3)
urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(num_urls)]

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_langchain"

main_placeholder = st.empty()

# Function to fetch article content from a URL
def fetch_article_content(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception:
        return None  # Handle cases where content can't be fetched

# Function to split text into chunks
def split_into_chunks(text, chunk_size=512, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    return text_splitter.split_text(text)

# Function to extract the paragraph containing the answer
def get_paragraph(text, start_idx, end_idx):
    # Find the start of the paragraph
    paragraph_start = text.rfind('\n', 0, start_idx)
    if paragraph_start == -1:
        paragraph_start = 0
    else:
        paragraph_start += 1  # Move past the newline character

    # Find the end of the paragraph
    paragraph_end = text.find('\n', end_idx)
    if paragraph_end == -1:
        paragraph_end = len(text)

    # Extract the paragraph
    paragraph = text[paragraph_start:paragraph_end]
    return paragraph.strip()

# Initialize embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Check for GPU availability and set device accordingly
device = 0 if torch.cuda.is_available() else -1

# Initialize the QA pipeline
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2", device=device)

docs = []  # Ensure docs is defined at the top-level scope

if process_url_clicked:
    if not any(urls):
        st.error("Please enter at least one URL.")
        st.stop()
    try:
        # Fetch article content
        main_placeholder.text("Fetching Article Content... ✅✅✅")
        for url in urls:
            if url:
                content = fetch_article_content(url)
                if content:
                    chunks = split_into_chunks(content)
                    for chunk in chunks:
                        # Use Document objects with metadata for source tracking
                        docs.append(Document(page_content=chunk, metadata={"source": url}))
                else:
                    st.warning(f"Could not fetch content from: {url}")
        if not docs:
            st.error("No valid content fetched. Please check the URLs and try again.")
            st.stop()

        # Create FAISS vector store from documents
        vector_store = FAISS.from_documents(docs, embedding_model)

        # Save the FAISS store locally
        vector_store.save_local(file_path)

        st.success("URLs processed successfully!")
    except Exception as e:
        st.error(f"Error occurred during processing: {e}")

# Question Answering Section
query = main_placeholder.text_input("Question: ")
if query:
    try:
        # Load FAISS vector store safely
        st.info("Loading FAISS vector store...")
        if os.path.exists(file_path):
            vector_store = FAISS.load_local(file_path, embeddings=embedding_model, allow_dangerous_deserialization=True)
        else:
            st.error("FAISS vector store not found. Please process URLs first.")
            st.stop()

        # Retrieve relevant documents
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # Retrieve top 3 documents
        source_docs = retriever.get_relevant_documents(query)

        if not source_docs:
            st.warning("No relevant documents found for your query.")
            st.stop()

        # Combine the content of the documents into a context
        context = ' '.join([doc.page_content for doc in source_docs])

        # Use the QA pipeline
        inputs = {'question': query, 'context': context}
        result = qa_pipeline(inputs)
        answer = result['answer']
        answer_start = result['start']
        answer_end = result['end']

        # Extract the paragraph containing the answer
        summary = get_paragraph(context, answer_start, answer_end)

        # Find the relevant URLs from the source documents
        relevant_urls = set()
        for doc in source_docs:
            relevant_urls.add(doc.metadata.get("source", "Unknown"))

        # Display answer
        st.header("Answer")
        st.write(answer)

        # Display summary
        st.subheader("Summary")
        st.write(summary)

        # Display relevant source URL(s)
        st.subheader("Source(s):")
        for url in relevant_urls:
            st.write(url)
    except Exception as e:
        st.error(f"Error occurred during querying: {e}")