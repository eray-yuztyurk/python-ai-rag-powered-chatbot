from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface.chat_models import ChatHuggingFace
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline

from dotenv import load_dotenv

load_dotenv()

embeddings_model_cache = {}
llm_models_cache = {}

# PDF Documents Loader
def pdf_loader(file_path:str) -> list[Document]:
    loader = PyPDFLoader(file_path=file_path)
    documents = loader.load()
    return documents


# Text into Chunks Splitter
def text_splitter(documents):
    """
    Splits documents into smaller chunks for easier processing.
    Uses RecursiveCharacterTextSplitter from LangChain with a chunk size of 1000 characters and
    an overlap of 200 characters between chunks.
    Args:
        documents (list[Document]): List of documents to be split.
    Returns:
        list[Document]: List of splitted document chunks.
    """
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    splitted_docs = text_splitter.split_documents(documents)
    return splitted_docs

# Embedding model loader
def load_embedding_model(model_name="intfloat/multilingual-e5-small"):
    """
    Loads and caches the embedding model.
    Args:
        model_name (str): Name of the embedding model to load.
    Returns:
        HuggingFaceEmbeddings: Loaded embedding model.
    """
    
    if not model_name in embeddings_model_cache:
        embeddings_model_cache[model_name] = HuggingFaceEmbeddings(model_name=model_name)
        embeddings = embeddings_model_cache[model_name]
    else:
        embeddings = embeddings_model_cache[model_name]
    return embeddings

# Document Embedder
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

# InMemory Vector Store Creator
def inmemory_vector_store_creator(splitted_docs):
    """
    Creates an in-memory vector store from the splitted document chunks.
    Args:
        splitted_docs (list[Document]): List of splitted document chunks.
    Returns:
        InMemoryVectorStore: In-memory vector store containing the document embeddings.
    """
    
    vector_store = InMemoryVectorStore.from_documents(
        documents=splitted_docs,
        embedding=load_embedding_model())
    ids = vector_store.add_documents(splitted_docs)
    return vector_store

# Get Content Pipeline
def get_content(query: str, file_path: str):
    """
    Retrieves relevant content from the document based on the query.
    Args:
        query (str): User query to search for relevant content.
        file_path (str): Path to the PDF document.
    Returns:
        tuple: Splitted documents, embedded documents, and found relevant documents.
    """
    
    documents = pdf_loader(file_path)
    splitted_docs = text_splitter(documents)
    embedded_docs = document_embedder(splitted_docs)
    vector_store = inmemory_vector_store_creator(splitted_docs)
    found_in_docs = vector_store.similarity_search(query, k=3)
    worked_docs = [
        "\n\n".join((f"Score: {score} \n Source: {doc.metadata} \n Content: {doc.page_content}") 
                    for doc, score in vector_store.similarity_search_with_score(query, k=3))
        ]   
    return splitted_docs, embedded_docs, worked_docs

# Chat Model Initializer
def chatbot_model_initializer(model_name="Qwen/Qwen2.5-1.5B-Instruct"):
    """
    Initializes and caches the chatbot model.
    Args:
        model_name (str): Name of the chatbot model to load.
    Returns:
        ChatHuggingFace: Initialized chatbot model. 
    """
    
    pipe = pipeline(
        "text-generation",
        model=model_name,
        max_new_tokens=1024,
        temperature=0.6,
        device_map="auto",
        torch_dtype="auto",
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    return ChatHuggingFace(llm=llm)
