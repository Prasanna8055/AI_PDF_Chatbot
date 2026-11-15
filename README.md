# AI PDF Chatbot using Retrieval-Augmented Generation (RAG)

## Project Overview
The AI PDF Chatbot is an intelligent document question-answering application that allows users to upload PDF files and ask questions based on their content.

It uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from uploaded documents and generate answers using a locally running Large Language Model (LLM).

## Features
- Upload PDF documents.
- Extract text from PDF pages.
- Split extracted text into smaller chunks.
- Generate embeddings using Nomic Embed Text.
- Retrieve relevant information using hybrid search.
- Generate answers using Llama 3.2.
- Display source page references.
- Interactive chatbot interface using Streamlit.
- Runs locally using Ollama.

## Technologies Used
- Python
- Streamlit
- Ollama
- Llama 3.2 (1B)
- Nomic Embed Text
- Retrieval-Augmented Generation (RAG)
- NumPy
- PyPDF

## Project Architecture

1. User uploads a PDF document.
2. The application extracts text from the PDF.
3. Extracted text is divided into smaller chunks.
4. Embeddings are generated for the chunks.
5. The user's question is converted into an embedding.
6. Relevant chunks are retrieved using hybrid search.
7. Retrieved information is passed to the Llama 3.2 model.
8. The chatbot generates an answer and displays source references.

## Project Structure

```text
AI_PDF_Chatbot/
│
├── app.py
├── rag_utils.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Prasanna8055/AI_PDF_Chatbot.git
cd AI_PDF_Chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Install Ollama models

Install Ollama, then run:

```bash
ollama pull llama3.2:1b
ollama pull nomic-embed-text
```

### 6. Run the application

```bash
streamlit run app.py
```

## Future Improvements
- Improve answer accuracy and retrieval quality.
- Support multiple PDF documents.
- Add chat history.
- Improve the user interface.

## Author
**Prasanna Gudi**

GitHub: [Prasanna8055](https://github.com/Prasanna8055)