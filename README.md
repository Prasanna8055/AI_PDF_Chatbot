# AI PDF Chatbot

## Project Overview
AI PDF Chatbot is an AI-powered application that allows users to upload PDF documents and ask questions based on their content.

The chatbot uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from uploaded PDF files and generate answers.

## Technologies Used
- Python
- Streamlit
- Ollama
- Llama 3.2 (1B)
- Nomic Embed Text
- PyPDF
- NumPy

## Features
- Upload PDF documents.
- Extract text from PDF pages.
- Split text into smaller chunks.
- Generate embeddings using Nomic Embed Text.
- Retrieve relevant information using semantic and keyword search.
- Generate answers using Llama 3.2.
- Display source page references.

## Installation

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Install Ollama and download the required models:

```bash
ollama pull llama3.2:1b
ollama pull nomic-embed-text
```

5. Run the application:

```bash
streamlit run app.py
```

## Note
Ollama must be installed and running locally to use this application.