"""
RAG Worker Module
Handles core RAG operations: document loading, chunking, embeddings, and vector search.
"""

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface.chat_models import ChatHuggingFace
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from utils import get_document_language
from dotenv import load_dotenv
import os

load_dotenv()

# Global caches for models to avoid reloading
embeddings_model_cache = {}
llm_models_cache = {}

def pdf_loader(file_path: str) -> list[Document]:
    """Load PDF document and convert to LangChain Document objects."""
    loader = PyPDFLoader(file_path=file_path)
    documents = loader.load()
    return documents


def text_splitter(documents):
    """
    Split documents into chunks for processing.
    
    Uses recursive character splitting to maintain context while keeping
    chunks small enough for embedding models.
    
    Args:
        documents: List of LangChain Document objects
        
    Returns:
        List of document chunks (1000 chars each, 200 char overlap)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(documents)


def load_embedding_model(model_name="intfloat/multilingual-e5-small"):
    """
    Load and cache embedding model for vector generation.
    
    Uses multilingual model to support cross-language search.
    Model is cached globally to avoid reloading on each request.
    
    Args:
        model_name: HuggingFace model identifier
        
    Returns:
        HuggingFaceEmbeddings instance
    """
    if model_name not in embeddings_model_cache:
        embeddings_model_cache[model_name] = HuggingFaceEmbeddings(model_name=model_name)
    
    return embeddings_model_cache[model_name]

# Document Embedder (it is not in active use, just for showcase how to embed documents)
def document_embedder(document):
    """
    Embeds the document chunks using the specified embedding model.
    Args:
        document (list[Document]): List of document chunks to be embedded.
    Returns:
        list: List of embedded document vectors.
    """
    
    embeddings = load_embedding_model()
    embedded_docs = embeddings.embed_documents([doc.page_content for doc in document])
    return embedded_docs

def inmemory_vector_store_creator(splitted_docs):
    """
    Create vector database from document chunks.
    
    Embeds all chunks and stores them in memory for fast similarity search.
    
    Args:
        splitted_docs: List of document chunks
        
    Returns:
        InMemoryVectorStore with embedded documents
    """
    vector_store = InMemoryVectorStore.from_documents(
        documents=splitted_docs,
        embedding=load_embedding_model()
    )
    return vector_store


def get_content(query, file_path):
    """
    Main RAG pipeline: load PDF, chunk it, search for relevant content.
    
    This function orchestrates the entire retrieval process:
    1. Load and parse PDF
    2. Split into chunks
    3. Detect document language
    4. Create vector store
    5. Search for top-3 most relevant chunks
    
    Args:
        query: User's search query
        file_path: Path to uploaded PDF
        
    Returns:
        tuple: (formatted_content, document_language)
        
    Raises:
        ValueError: If query is empty or file is missing
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty. Please provide a valid query.")

    if file_path is None:
        raise ValueError("No PDF file uploaded.")

    # Load and process document
    documents = pdf_loader(file_path)
    splitted_docs = text_splitter(documents)
    
    # Detect document language (cached for this file)
    doc_language = get_document_language(file_path, splitted_docs[0].page_content)
    
    # Create vector store and search
    vector_store = inmemory_vector_store_creator(splitted_docs)
    results = vector_store.similarity_search_with_score(query, k=3)
    
    # Format results
    content = "\n\n".join(
        f"Score: {score}\nSource: {doc.metadata}\nContent: {doc.page_content}"
        for doc, score in results
    )
    
    return content, doc_language

def local_chatbot_initializer(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """
    Initialize local HuggingFace model for chat.
    
    Runs inference on your machine - requires GPU/sufficient RAM.
    Model is cached after first load.
    
    Args:
        model_name: HuggingFace model identifier
        
    Returns:
        ChatHuggingFace instance
    """
    if model_name in llm_models_cache:
        return llm_models_cache[model_name]

    # Create text generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model_name,
        max_new_tokens=512,
        temperature=0.7,
        device_map="auto"
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    chat_model = ChatHuggingFace(llm=llm)
    
    llm_models_cache[model_name] = chat_model
    return chat_model


def api_gemini_initializer(model_name="models/gemini-flash-latest"):
    """
    Initialize Google Gemini API model.
    
    Fast, cloud-based inference with free tier (1500 req/day).
    Requires GEMINI_API_KEY in .env file.
    
    Args:
        model_name: Gemini model identifier
        
    Returns:
        ChatGoogleGenerativeAI instance
    """
    if model_name in llm_models_cache:
        return llm_models_cache[model_name]
    
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.7,
        max_tokens=512
    )
    
    llm_models_cache[model_name] = llm
    return llm


def api_groq_initializer(model_name="llama-3.1-8b-instant"):
    """
    Initialize Groq API model.
    
    Ultra-fast inference with LPU technology.
    Requires GROQ_API_KEY in .env file.
    
    Args:
        model_name: Groq model identifier
        
    Returns:
        ChatGroq instance
    """
    if model_name in llm_models_cache:
        return llm_models_cache[model_name]
    
    llm = ChatGroq(
        model=model_name,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.7,
        max_tokens=512
    )
    
    llm_models_cache[model_name] = llm
    return llm
