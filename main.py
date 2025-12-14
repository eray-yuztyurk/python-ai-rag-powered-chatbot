"""
RAG-Powered Chatbot - Main Application
A multilingual document Q&A system using Retrieval-Augmented Generation.

This module provides the Gradio UI and orchestrates the RAG pipeline.
"""

from worker import get_content, local_chatbot_initializer, api_gemini_initializer, api_groq_initializer
from utils import parse_response_content, get_model_registry, build_rag_prompt, detect_language
import gradio as gr

# Initialize model registry
MODEL_REGISTRY = get_model_registry()

# System instruction for the LLM
system_prompt = """
You are a helpful AI assistant.
Answer user queries using the 'Document Content' provided below.

IMPORTANT: 
- Find relevant information from the context, regardless of language
- Answer in the SAME LANGUAGE as the user's question
- If information is not available, respond with "I don't know" (in user's language)
"""

def get_llm_response(prompt, llm_using_approach="Run model with API (No GPU required, but may incur costs)", model_name="Gemini Flash"):
    """
    Routes prompts to the selected LLM and returns parsed response.
    
    Args:
        prompt: The text prompt to send to the model
        llm_using_approach: Model type selection (Local or API)
        model_name: Specific model name from dropdown
        
    Returns:
        str: Parsed response from the LLM
    """
    # Select and initialize model based on user choice
    if "Local" in llm_using_approach:
        if model_name == "TinyLlama 1.1B":
            model_full_name = MODEL_REGISTRY["local"]["TinyLlama 1.1B"]["params"]["model_name"]
            model_initializer = local_chatbot_initializer(model_name=model_full_name)
        else:
            model_full_name = MODEL_REGISTRY["local"]["Microsoft Phi-2"]["params"]["model_name"]
            model_initializer = local_chatbot_initializer(model_name=model_full_name)
    else:
        if model_name == "Groq Llama":
            model_full_name = MODEL_REGISTRY["api"]["Groq Llama"]["params"]["model_name"]
            model_initializer = api_groq_initializer(model_name=model_full_name)
        else:
            model_full_name = MODEL_REGISTRY["api"]["Gemini Flash"]["params"]["model_name"]
            model_initializer = api_gemini_initializer(model_name=model_full_name)

    # Get response and parse it
    chat_model = model_initializer
    response = chat_model.invoke(prompt)
    parsed_response = parse_response_content(response)
    return parsed_response

def get_rag_answer(llm_using_approach, model_name, query, file_path):
    """
    Main RAG pipeline with multilingual support.
    
    Workflow:
    1. Detect query language
    2. Retrieve relevant document chunks via vector search
    3. Translate query if needed for better embedding match
    4. Generate answer in original query language
    
    Args:
        llm_using_approach: Model type (Local/API)
        model_name: Specific model selection
        query: User's question
        file_path: Path to uploaded PDF
        
    Returns:
        str: Answer from the LLM or error message
    """
    try:
        if file_path is None:
            return "⚠️ No PDF file uploaded!\n\nPlease use the 'Upload your document' button to select a PDF file."
        
        # Step 1: Detect query language
        query_lang = detect_language(query)
        
        # Step 2: Get document content via RAG (returns content + document language)
        doc_content, doc_lang = get_content(query, file_path)
        
        # Step 3: Handle language mismatch for better embedding search
        if query_lang != doc_lang:
            # Translate query to document language for better vector search
            translate_prompt = f"Translate this query to {doc_lang} language, reply ONLY with translation: {query}"
            query_translated = get_llm_response(translate_prompt, llm_using_approach, model_name)
            
            # Re-run RAG with translated query
            doc_content, _ = get_content(query_translated, file_path)
            
            # Build prompt with both original and translated queries
            prompt = f"""
{system_prompt}

Document Content:
{doc_content}

User Question (Original - {query_lang}): {query}
User Question (Translated - {doc_lang}): {query_translated}

IMPORTANT: Answer in {query_lang} language (the user's original language).

Answer:
"""
        else:
            # Same language - use simple prompt
            prompt = build_rag_prompt(system_prompt, doc_content, query)
        
        return get_llm_response(prompt, llm_using_approach, model_name)

    except ValueError as e:
        return f"⚠️ {str(e)}\n\nPlease upload a PDF file using the 'Upload your document' button."
    
    except Exception as e:
        return f"⚠️ An error occurred:\n{str(e)}\n\nPlease check:\n• PDF file is uploaded\n• Model selection is correct\n• Internet connection (for API models)"



# Gradio User Interface
with gr.Blocks(theme=gr.themes.Citrus()) as _ui:
    gr.Markdown("<h1 style='text-align: center;'>RAG Powered Chatbot</h1>")
    gr.Markdown("<h4 style='text-align: center;'>Ask questions about your documents in any language</h4>")

    with gr.Row():
        # Left column: Model selection and file upload
        with gr.Column():
            approach = gr.Radio(
                choices=[
                    "Run model on Local Machine (Requires GPU and sufficient VRAM)",
                    "Run model with API (No GPU required, but may incur costs)"
                ], 
                label="Model Type", 
                value="Run model with API (No GPU required, but may incur costs)"
            )

            model_name = gr.Dropdown(
                choices=["Gemini Flash", "Groq Llama"], 
                label="Select Model", 
                value="Gemini Flash"
            )

            file_input = gr.File(
                label="Upload your document", 
                file_types=[".pdf"]
            )

        # Right column: Query input and output
        with gr.Column():
            query = gr.Textbox(
                label="Enter your query here", 
                lines=3,
                placeholder="Type your question in any language..."
            )
            
            submit_btn = gr.Button("Get Answer", variant="primary")
            output = gr.Textbox(label="Answer", lines=10)
    
    def update_model_dropdown(selected_approach):
        """Update model choices based on selected approach."""
        if "Local" in selected_approach:
            choices = list(MODEL_REGISTRY["local"].keys())
        else:
            choices = list(MODEL_REGISTRY["api"].keys())
        return gr.Dropdown(choices=choices, value=choices[0])
    
    # Wire up event handlers
    approach.change(
        fn=update_model_dropdown,
        inputs=approach,
        outputs=model_name
    )
    
    submit_btn.click(
        fn=get_rag_answer,
        inputs=[approach, model_name, query, file_input],
        outputs=output
    )

if __name__ == "__main__":
    _ui.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)