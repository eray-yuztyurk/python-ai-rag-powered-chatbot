"""
Utility Module
Helper functions for language detection, prompt building, and response parsing.
"""

from langdetect import detect, LangDetectException

# Cache to avoid re-detecting language for same document
document_language_cache = {}


def detect_language(text):
    """
    Detect the language of given text using langdetect.
    
    Args:
        text: Text to analyze
        
    Returns:
        Language code (e.g., 'en', 'tr', 'es')
        Defaults to 'en' if detection fails
    """
    try:
        return detect(text)
    except LangDetectException:
        return 'en'


def get_document_language(file_path, first_chunk_text):
    """
    Detect and cache document language.
    
    Uses first 500 characters for detection. Caches result
    to avoid re-processing same document.
    
    Args:
        file_path: Document path (used as cache key)
        first_chunk_text: First chunk of document
        
    Returns:
        Language code
    """
    if file_path not in document_language_cache:
        sample = first_chunk_text[:500] if len(first_chunk_text) > 500 else first_chunk_text
        document_language_cache[file_path] = detect_language(sample)
    return document_language_cache[file_path]


def build_rag_prompt(system_prompt, doc_content, query):
    """
    Construct prompt for LLM with retrieved context.
    
    Args:
        system_prompt: System instructions
        doc_content: Retrieved document chunks
        query: User's question
        
    Returns:
        Formatted prompt string
    """
    return f"""
{system_prompt}

Document Content:
{doc_content}

User Question: {query}

IMPORTANT: Answer in the SAME LANGUAGE as the user's question.

Answer:
"""


def _get_model_initializers():
    """Lazy import to avoid circular dependency between utils and worker."""
    from worker import local_chatbot_initializer, api_gemini_initializer, api_groq_initializer
    return local_chatbot_initializer, api_gemini_initializer, api_groq_initializer


def get_model_registry():
    """
    Build model registry with lazy-loaded initializers.
    
    Separates model configuration from initialization to avoid
    circular imports. Registry includes both local and API models.
    """
    local_init, gemini_init, groq_init = _get_model_initializers()
    
    return {
        "local": {
            "TinyLlama 1.1B": {
                "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "description": "Lightweight model, 1.5GB RAM",
                "initializer": local_init,
                "params": {"model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"}
            },
            "Microsoft Phi-2": {
                "name": "microsoft/phi-2",
                "description": "Better quality, 2.7GB RAM",
                "initializer": local_init,
                "params": {"model_name": "microsoft/phi-2"}
            }
        },
        "api": {
            "Gemini Flash": {
                "name": "models/gemini-flash-latest",
                "description": "Fast and free (1500 req/day)",
                "initializer": gemini_init,
                "params": {"model_name": "models/gemini-flash-latest"}
            },
            "Groq Llama": {
                "name": "llama-3.1-8b-instant",
                "description": "Ultra fast, free tier available",
                "initializer": groq_init,
                "params": {"model_name": "llama-3.1-8b-instant"}
            }
        }
    }

# Backward compatibility
MODEL_REGISTRY = None  # Will be initialized lazily

def parse_response_content(response):
    """
    Parses the response content from various LLM providers.
    Args:
        response: The response object from the LLM.
    Returns:
        str: Parsed text content.
    """
    try:
        content = response.content
        
        # 1. String format (Groq, OpenAI, Local models)
        if isinstance(content, str):
            # Special cleaning for TinyLlama and similar local models
            cleaned = content
            
            # Remove prompt tags (<|user|>, <|assistant|>, </s>)
            if '<|assistant|>' in cleaned:
                # Take content after the last <|assistant|> tag
                parts = cleaned.split('<|assistant|>')
                if len(parts) > 1:
                    cleaned = parts[-1]
            
            # Remove stop tokens
            stop_tokens = ['</s>', '<|user|>', '<|endoftext|>']
            for token in stop_tokens:
                if token in cleaned:
                    cleaned = cleaned.split(token)[0]
            
            # Strip leading/trailing whitespace
            cleaned = cleaned.strip()
            
            # Reduce repetitive sentences (simple approach)
            # If same sentence repeats 3+ times, show only twice
            lines = cleaned.split('\n')
            seen = {}
            filtered_lines = []
            for line in lines:
                line_stripped = line.strip()
                if line_stripped:
                    seen[line_stripped] = seen.get(line_stripped, 0) + 1
                    if seen[line_stripped] <= 2:  # Allow max 2 repetitions
                        filtered_lines.append(line)
            
            return '\n'.join(filtered_lines) if filtered_lines else cleaned
        
        # 2. List format (Gemini's new format)
        elif isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict):
                    # Find 'text' field
                    texts.append(item.get('text', str(item)))
                else:
                    texts.append(str(item))
            return " ".join(texts)
        
        # 3. Dict format
        elif isinstance(content, dict):
            return content.get('text', content.get('content', str(content)))
        
        # 4. Other formats (for future providers)
        else:
            return str(content)
            
    except Exception as e:
        # Last resort: return error message
        return f"⚠️ Response parsing error: {str(e)}"