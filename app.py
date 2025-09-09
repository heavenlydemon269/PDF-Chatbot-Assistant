import streamlit as st
from pypdf import PdfReader
import io
from transformers import pipeline
import time
import re
import os # Import the os module

# --- Model Loading (Cached for Performance) ---

@st.cache_resource
def load_qa_pipeline():
    """Loads your locally fine-tuned question-answering pipeline."""
    # NOTE: You need to make sure the 'squad_finetuned_model' directory is
    # available in the same location as your Streamlit script.
    model_path = "./squad_finetuned_model"
    
    if not os.path.isdir(model_path):
        st.error(f"Model directory not found at: {model_path}")
        st.error("Please make sure your fine-tuned model is in the correct directory.")
        return None

    st.info("Loading your fine-tuned QA model...")
    qa_pipeline = pipeline("question-answering", model=model_path, tokenizer=model_path)
    st.info("Fine-tuned QA model loaded successfully!")
    return qa_pipeline

@st.cache_resource
def load_translator(model_name):
    """Loads a specific Hugging Face translation pipeline."""
    translator = pipeline("translation", model=model_name)
    return translator

# --- Helper Functions ---

def get_pdfs_text(pdf_docs_bytes):
    """Extracts text from a list of PDF files."""
    text = ""
    for pdf_bytes in pdf_docs_bytes:
        try:
            pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            st.error(f"Error reading a PDF: {e}")
    return text

def detect_language_and_get_translator(question):
    """
    Detects language keywords in the question and returns the appropriate translator.
    NOTE: This is a simple keyword-based approach.
    """
    # Using regex for case-insensitive matching
    if re.search(r'\bin hindi\b', question, re.IGNORECASE):
        return load_translator("Helsinki-NLP/opus-mt-en-hi")
    if re.search(r'\bin marathi\b', question, re.IGNORECASE):
         # Note: You would need to find a suitable en-mr model on Hugging Face
         st.warning("A local Marathi translation model is not loaded in this demo.")
         return None
    if re.search(r'\bin french\b', question, re.IGNORECASE):
        return load_translator("Helsinki-NLP/opus-mt-en-fr")
    # Add more languages here as needed
    return None


def get_local_response(qa_pipeline, context, question):
    """
    Gets a response from the local models, with translation if requested.
    """
    if qa_pipeline is None:
        return "The Question-Answering model is not loaded. Please check the model path."
    try:
        # 1. Get the answer in English first
        qa_result = qa_pipeline(question=question, context=context)
        english_answer = qa_result['answer']

        # 2. Check if translation is needed
        translator = detect_language_and_get_translator(question)

        # 3. Translate if a model was found
        if translator:
            translated_result = translator(english_answer)
            return translated_result[0]['translation_text']
        
        return english_answer

    except Exception as e:
        return f"An error occurred with the local model: {e}"

def stream_response(text):
    """Yields text word by word for a streaming effect."""
    for word in text.split():
        yield word + " "
        time.sleep(0.05)

# --- Streamlit App ---

st.set_page_config(page_title="Local Multilingual PDF Chatbot", layout="wide")

st.title("📄 Local Multilingual PDF Chatbot")
st.write("This chatbot runs on your custom-trained model and can translate answers.")

# Load the main QA model
qa_pipeline = load_qa_pipeline()

# --- Sidebar for PDF Upload ---
with st.sidebar:
    st.header("Configuration")
    uploaded_files = st.file_uploader("Upload your PDF(s)", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        st.success(f"{len(uploaded_files)} PDF(s) uploaded successfully!")
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                pdf_docs_bytes = [file.getvalue() for file in uploaded_files]
                st.session_state.pdf_text = get_pdfs_text(pdf_docs_bytes)
                st.session_state.messages = [{"role": "assistant", "content": "I've processed the documents. What would you like to know?"}]
            st.success("Documents processed!")

# --- Main Chat Interface ---

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question (e.g., 'What is NLP in hindi?')"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    if "pdf_text" not in st.session_state or not st.session_state.pdf_text:
        st.warning("Please upload and process at least one PDF document first.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_local_response(
                    qa_pipeline, 
                    st.session_state.pdf_text,
                    prompt
                )
            st.write_stream(stream_response(response))
        st.session_state.messages.append({"role": "assistant", "content": response})

