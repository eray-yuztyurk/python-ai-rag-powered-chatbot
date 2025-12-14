
from worker import get_content, chatbot_model_initializer
import gradio as gr

data_file_path = "data/sample_data/machine_learning_wiki.pdf"

system_prompt ="""
    You have access to a tool that retrieves context from a document. 
    Help answer user queries by using below 'Document Content'. If information is not available,
    respond with "I don't know".
"""

def get_llm_response(prompt):
    chat_model = chatbot_model_initializer()
    response = chat_model.invoke(prompt)
    return response.content

def get_rag_answer(query, file_path=data_file_path):
    doc_content = get_content(query, file_path)

    prompt=f"""
        {system_prompt}
        Document Content: {doc_content}
        User Query: {query}
        Answer:
        """
    return get_llm_response(prompt)

_ui = gr.Interface(
    fn=get_rag_answer,
    inputs=[
        gr.Textbox(label="Query"), 
        gr.UploadButton(label="Upload your document", file_types=[".pdf"])],
    outputs=gr.Textbox(label="Answer"),
    title="RAG Powered Chatbot",
    description="A chatbot powered by Retrieval-Augmented Generation (RAG) using LangChain and Gradio."
)

if __name__ == "__main__":
    _ui.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)