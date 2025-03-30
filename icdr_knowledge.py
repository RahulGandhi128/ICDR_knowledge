from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
import os
from PyPDF2 import PdfReader
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter # Keep import but make chunking optional
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from io import BytesIO
import streamlit as st
from PIL import Image
import requests
from pathlib import Path
import pickle
import tempfile
import shutil

# Add after imports
# Set wide layout
st.set_page_config(layout="wide")

# --- CUSTOMIZATION ---
PRIMARY_COLOR = "#C8102E"  # Red from the logo
SECONDARY_COLOR = "#223A70" # Dark blue from the logo
BACKGROUND_COLOR = "#FFFFFF" # White background
TEXT_COLOR = "#000000" # Black text

# Apply custom theme
st.markdown(
    f"""
    <style>
    body {{
        color: {TEXT_COLOR};
        background-color: {BACKGROUND_COLOR};
    }}
    .stApp {{
        background-color: {BACKGROUND_COLOR};
    }}
    .stButton>button {{
        color: white;
        background-color: {PRIMARY_COLOR};
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 5px;
    }}
    .stButton>button:hover {{
        background-color: {SECONDARY_COLOR};
    }}
    .chat-message {{
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }}
    .chat-message.user {{
        background-color: #e6f3ff;
        border-left: 5px solid {PRIMARY_COLOR};
    }}
    .chat-message.assistant {{
        background-color: #f0f2f6;
        border-left: 5px solid {SECONDARY_COLOR};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

def display_logo():
    logo_url = "https://raw.githubusercontent.com/RahulGandhi128/ICDR_knowledge/main/image001.png" # Replace with your actual raw URL

    col1, col2 = st.columns([1, 4])
    with col1:
        try:
            response = requests.get(logo_url)
            response.raise_for_status() # Raise an exception for bad status codes
            image = Image.open(BytesIO(response.content))
            st.image(image, use_container_width=True)
        except requests.exceptions.RequestException as e:
            st.warning(f"Error loading company logo from URL: {e}")
        except Exception as e:
            st.warning(f"Error opening image: {e}")

    with col2:
        st.title("ICDR Regulations Assistant")
        st.write("Ask questions about ICDR regulations and procedures.")

# Set Google API key
google_api_key = "AIzaSyCcUFY04YwiLbCdYFvjXzWg-ze0LOtYKmY"

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    if isinstance(pdf_file, BytesIO):  # If already a file-like object
        pdfReader = PdfReader(pdf_file)
    else:  # If it's a file path
        pdfReader = PdfReader(pdf_file)

    all_text = ""
    for page in pdfReader.pages:
        text = page.extract_text()
        if text:
            all_text += text.encode('ascii', 'ignore').decode('ascii') + "\n"
    return all_text

# Function to split text into smaller chunks (now optional - set large chunk size to effectively disable)
def get_text_chunks(text, chunking_enabled=True): # Added chunking_enabled flag
    if chunking_enabled:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100) # Large chunk size to minimize splitting
        return text_splitter.split_text(text)
    else:
        return [text] # If chunking disabled, treat the whole text as a single chunk


# Compliance check function
def get_compliance_chain():
    prompt_template = """
    You are an expert AI assistant specializing in ICDR (International Centre for Dispute Resolution) regulations and procedures. Your role is to provide accurate guidance and interpretation of ICDR rules and procedures based on the official ICDR documentation provided in the context.

    When analyzing queries, please:
    1. Reference specific ICDR articles and sections when applicable with page numbers and paragraphs.
    2. Explain procedures and requirements clearly
    3. Highlight any relevant deadlines or time limits
    4. Provide accurate interpretations of ICDR rules and guidelines
    5. If information is not covered in the ICDR documents, explicitly state that

    Context (ICDR Documentation):\n {context} \n
    User Question:\n {submission} \n

    ICDR Analysis and Response:
    """

    model = ChatGoogleGenerativeAI(
        google_api_key=google_api_key,
        model="gemini-2.0-flash-thinking-exp-01-21",
        temperature=0.1,
        max_output_tokens=10000
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "submission"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Function to check compliance against stored regulatory documents
def check_compliance(user_submission):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(google_api_key=google_api_key, model="models/embedding-001")

        # Load vector store from GitHub
        vector_store = load_vector_store_from_github()
        if vector_store is None:
            return {"output_text": "Failed to load knowledge base."}, []

        print(f"User Submission for Similarity Search: '{user_submission}'")
        relevant_docs = vector_store.similarity_search(user_submission, k=25)

        if not relevant_docs:
            print("No relevant documents found by similarity search!")
        else:
            print(f"Number of relevant documents retrieved: {len(relevant_docs)}")
            print(f"First document preview: {relevant_docs[0].page_content[:200]}...")

        chain = get_compliance_chain()
        response = chain({"input_documents": relevant_docs, "submission": user_submission}, return_only_outputs=True)

        return response, relevant_docs
    except Exception as e:
        return {"output_text": f"Error processing query: {str(e)}"}, []

# Run this once to store compliance documents permanently (Only run this once initially, or when you update documents)
# load_documents()  # Commented out for cloud deployment

def load_vector_store_from_github():
    """Load FAISS vector store from GitHub"""
    embeddings = GoogleGenerativeAIEmbeddings(google_api_key=google_api_key, model="models/embedding-001")
    repo_base_url = "https://raw.githubusercontent.com/RahulGandhi128/ICDR_knowledge/main/"
    faiss_file_name = "index.faiss"

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            local_faiss_path = os.path.join(tmpdir, faiss_file_name)
            faiss_url = repo_base_url + faiss_file_name

            # Debugging: Check if the FAISS file exists
            check_response = requests.head(faiss_url)
            if check_response.status_code != 200:
                st.error(f"FAISS file not found at {faiss_url}")
                return None

            response = requests.get(faiss_url)
            response.raise_for_status()  # Raise an exception for bad responses
            with open(local_faiss_path, 'wb') as f:
                f.write(response.content)

            # Load FAISS
            vector_store = FAISS.load_local(tmpdir, embeddings)
            return vector_store
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading FAISS vector store: {e}")
        return None
    except Exception as e:
        st.error(f"Error loading FAISS vector store: {e}")
        return None

def load_vector_store_from_github():
    """Load FAISS vector store from GitHub"""
    embeddings = GoogleGenerativeAIEmbeddings(google_api_key=google_api_key, model="models/embedding-001")
    repo_base_url = "https://raw.githubusercontent.com/RahulGandhi128/ICDR_knowledge/main/" # Adjust if files are in a subfolder
    faiss_file_name = "index.faiss" # Using the requested filename

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            local_faiss_path = os.path.join(tmpdir, faiss_file_name)
            # Download the .faiss file
            faiss_url = repo_base_url + faiss_file_name
            response = requests.get(faiss_url)
            response.raise_for_status() # Raise an exception for bad status codes
            with open(local_faiss_path, 'wb') as f:
                f.write(response.content)

            # Load the FAISS index
            vector_store = FAISS.load_local(tmpdir, embeddings)
            return vector_store
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading vector store files: {e}")
        return None
    except Exception as e:
        st.error(f"Error loading vector store from local files: {e}")
        return None

def run_chatbot():
    display_logo()

    # Chat interface
    if st.session_state.messages:
        for message in st.session_state.messages:
            with st.container():
                st.markdown(f"""
                    <div class="chat-message {message['role']}">
                        <b>{message['role'].title()}:</b><br>{message['content']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # User input
    user_query = st.text_input("Ask about ICDR regulations:", key="user_input")

    if st.button("Submit", key="submit"):
        if user_query:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_query})

            # Get AI response
            with st.spinner("Analyzing your query..."):
                response, _ = check_compliance(user_query)
                ai_response = response['output_text']

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": ai_response})

            # Clear input using the new rerun method
            st.rerun()

    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()


if __name__ == "__main__":
    try:
        import faiss
    except ImportError:
        st.error("Please install FAISS first: pip install faiss-cpu")
        st.stop()

    # Initialize Google API key from Streamlit secrets
    google_api_key = st.secrets["GOOGLE_API_KEY"]

    run_chatbot()
