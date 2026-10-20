# KnowFlow — Document RAG Assistant

KnowFlow is a beginner-friendly Retrieval-Augmented Generation (RAG) application. Upload text-based PDFs, index their text in Chroma using Gemini embeddings, and ask questions answered from retrieved document passages.

## Stack
- Python
- Streamlit chat UI
- LangChain document loading, splitting, and retrieval
- Chroma vector database
- Gemini embeddings and Gemini Flash for generation

## Setup (Windows PowerShell)

1. Install Python 3.11 or 3.12 from https://www.python.org/downloads/ if it is not already installed. During installation, enable **Add Python to PATH**.
2. Open this folder in VS Code.
3. Open **Terminal → New Terminal**.
4. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use Command Prompt and run `.venv\Scripts\activate.bat`, or run `python -m pip install -r requirements.txt` without activating.

5. Install packages:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

6. Copy `.env.example` to a new file named `.env`. Add a Google AI Studio API key from https://aistudio.google.com/apikey.
7. Start the app:

   ```powershell
   streamlit run app.py
   ```

8. Upload a text-based PDF in the sidebar, click **Index uploaded PDFs**, then ask a question.

## How the RAG flow works

1. `PyPDFLoader` extracts text and page metadata.
2. `RecursiveCharacterTextSplitter` splits pages into overlapping chunks.
3. `GoogleGenerativeAIEmbeddings` converts chunks into vectors.
4. Chroma stores vectors and their source metadata.
5. `similarity_search` retrieves the four most relevant chunks for a question.
6. Gemini generates an answer using the retrieved context, and the app displays the source passages.

## Limitations of this starter version

- No authentication or multi-user access control.
- No LangGraph workflow yet; retrieval is a linear pipeline.
- Scanned PDFs need OCR.
- A new indexing run creates a new Chroma collection. The local Chroma data is stored in `chroma_db/`.
- The app has not been benchmarked. Do not claim accuracy or latency improvements without measuring them.

## Security

Never commit `.env` or publish your API key. `.gitignore` excludes `.env` and the local vector database.
