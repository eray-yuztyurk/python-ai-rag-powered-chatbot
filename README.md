<h1 align="center"> RAG-Powered Chatbot </h1>

A multilingual document Q&A chatbot built with Retrieval-Augmented Generation (RAG). Upload PDFs, ask questions in any language, and get accurate answers extracted from your documents.

---
<p align="center">
 <img width="1878" height="869" alt="image" src="https://github.com/user-attachments/assets/c8d19563-8c1d-4f8b-90e5-d58fe6935a81" />
</p>

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [License](#license)

## Features

- **Multi-Model Support**: Choose between local models (TinyLlama, Phi-2) or cloud APIs (Google Gemini, Groq)
- **Multilingual**: Automatically detects query and document languages, translates when needed for optimal results
- **Smart Embedding Matching**: Uses language-aware vector search to find relevant content
- **Source References**: Shows which document pages and sections were used to generate the answer, with relevance scores
- **Caching**: Models and embeddings are cached to improve performance
- **Clean UI**: Simple Gradio interface for easy interaction

## Architecture

```
User Query → Language Detection → Translation (if needed) → Vector Search → LLM → Answer
                                           ↓
                                    PDF Document
                                           ↓
                                   Text Chunks (1000 chars)
                                           ↓
                                   Multilingual Embeddings
                                           ↓
                                   In-Memory Vector Store
```

## Tech Stack

**Core Frameworks:**
- LangChain - RAG pipeline orchestration
- Gradio - Web interface
- HuggingFace Transformers - Local model inference

**LLM Providers:**
- Google Gemini (via langchain-google-genai)
- Groq (via langchain-groq)
- Local models (via HuggingFace)

**Utilities:**
- langdetect - Language detection
- sentence-transformers - Multilingual embeddings
- PyPDF - Document parsing

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/python-ai-rag-powered-chatbot.git
   cd python-ai-rag-powered-chatbot
   ```

2. **Create virtual environment**
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Create a `.env` file in the project root with your API keys:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

**Getting API Keys:**
- **Gemini**: Get free API key at [Google AI Studio](https://makersuite.google.com/app/apikey) (1500 requests/day)
- **Groq**: Sign up at [Groq Console](https://console.groq.com/) for free tier access

## Usage

1. **Start the application**
   ```bash
   python main.py
   ```

2. **Open in browser**
   - The app will automatically open at `http://127.0.0.1:7860`
   - Or manually navigate to the URL shown in terminal

3. **Use the chatbot**
   - Select model type (Local or API)
   - Choose specific model from dropdown
   - Upload a PDF document
   - Type your question in any language
   - Click "Get Answer"

## How It Works

### 1. Document Processing
When you upload a PDF:
- Document is loaded and split into 1000-character chunks
- Language is detected from the first chunk
- Chunks are embedded using multilingual model (`intfloat/multilingual-e5-small`)
- Embeddings stored in in-memory vector database

### 2. Query Processing
When you ask a question:
- Query language is detected
- If query language ≠ document language, query is translated
- Translated query searches vector database for top 3 relevant chunks
- Retrieved content + original query sent to LLM
- Answer returned in original query language
- **Source references** displayed showing:
  - Document name and page number
  - Relevance score (0-100%)
  - Content preview from each retrieved chunk

### 3. Response Parsing
Different LLM providers return different formats:
- Gemini: List of dicts with text fields
- Groq: Plain string
- Local models: String with special tokens

The response parser handles all formats and cleans output automatically.

## Project Structure

```
python-ai-rag-powered-chatbot/
├── main.py              # Gradio UI and orchestration
├── worker.py            # Core RAG pipeline (PDF loading, embeddings, vector store)
├── utils.py             # Helper functions (language detection, response parsing)
├── requirements.txt     # Python dependencies
├── .env                 # API keys (not in repo)
├── .gitignore          # Git exclusions
├── LICENSE             # MIT License
└── data/
    └── sample_data/    # Example PDFs for testing
```

### Key Components

**`main.py`** - Application entry point
- `get_llm_response()`: Routes requests to appropriate model
- `get_rag_answer()`: Orchestrates full RAG pipeline with language handling
- Gradio Blocks UI with conditional model selection

**`worker.py`** - RAG operations
- `pdf_loader()`: Loads PDF documents
- `text_splitter()`: Chunks documents (1000 chars, 200 overlap)
- `load_embedding_model()`: Caches embedding model
- `inmemory_vector_store_creator()`: Creates vector database
- `get_content()`: Performs similarity search and returns content, language, and formatted references
- Model initializers for local, Gemini, and Groq models

**`utils.py`** - Utilities
- `detect_language()`: Detects text language using langdetect
- `get_document_language()`: Caches document language detection
- `build_rag_prompt()`: Constructs LLM prompts
- `get_model_registry()`: Lazy-loads model configurations
- `parse_response_content()`: Normalizes responses from different providers

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: Local models require significant RAM (1.5-3GB). For low-resource systems, use API models instead.
