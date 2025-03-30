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
    logo_path = r"C:\Users\Asus\OneDrive\Desktop\New_berry\image001.png"
    
    col1, col2 = st.columns([1, 4])
    with col1:
        try:
            image = Image.open(logo_path)
            st.image(image, use_container_width=True)
        except FileNotFoundError:
            st.warning("Company logo not found. Please check the file path.")
    
    with col2:
        st.title("ICDR Regulations Assistant")
        st.write("Ask questions about ICDR regulations and procedures.")

# Set Google API key
google_api_key = "AIzaSyCcUFY04YwiLbCdYFvjXzWg-ze0LOtYKmY"

# Add after the Google API key definition
FAISS_INDEX_NAME = "faiss_index_icdr"
FAISS_INDEX_PATH = os.path.join(r"C:\Users\Asus\OneDrive\Desktop\New_berry\programs\python\rag", FAISS_INDEX_NAME)

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

# # Function to extract text from a webpage # Removed as per request
# def extract_text_from_url(url):
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, 'html.parser')
#     article_content = soup.find_all('p')
#     text = '\n'.join([p.get_text() for p in article_content])
#     return text.encode('ascii', 'ignore').decode('ascii')

# Function to split text into smaller chunks (now optional - set large chunk size to effectively disable)
def get_text_chunks(text, chunking_enabled=True): # Added chunking_enabled flag
    if chunking_enabled:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100) # Large chunk size to minimize splitting
        return text_splitter.split_text(text)
    else:
        return [text] # If chunking disabled, treat the whole text as a single chunk


# Function to store documents permanently in FAISS
def update_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(google_api_key=google_api_key, model="models/embedding-001")

        # Load existing ICDR-specific FAISS DB if available, otherwise create a new one
        if os.path.exists(FAISS_INDEX_PATH):
            vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            vector_store.add_texts(text_chunks)  # Add new data to existing FAISS index
        else:
            vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

        vector_store.save_local(FAISS_INDEX_PATH)  # Save updated FAISS index
        return True
    except ImportError:
        st.error("FAISS library not found. Please install it using: pip install faiss-cpu")
        return False
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return False

#Function to load documents into FAISS permanently
def load_documents(chunk_document=True): # Added chunk_document parameter to control chunking
    all_text = ""

    # List of PDF files to add permanently
    pdf_files = [r"C:\Users\Asus\OneDrive\Desktop\New_berry\1717132711878.pdf"] # Replace with your actual PDF path if needed, or keep it as is if the path is correct
    print(f"Loading PDF files: {pdf_files}") # DEBUGGING STEP 1: Check if load_documents is called and PDF list

    for pdf in pdf_files:
        print(f"Processing PDF: {pdf}") # DEBUGGING STEP 2: Check if loop starts and PDF path is printed
        try:  # Added try-except to catch file opening errors
            with open(pdf, "rb") as file:
                pdf_text = extract_text_from_pdf(file)
                print(f"Text extracted from PDF, length: {len(pdf_text)}") # DEBUGGING STEP 3: Check if text is extracted and length
                all_text += pdf_text
        except Exception as e:
            print(f"Error opening or processing PDF {pdf}: {e}") # DEBUGGING STEP 4: Catch PDF errors

    # # List of URLs to add permanently # Removed URL loading
    # urls = []
    # for url in urls:  # No changes needed for URLs if you're not using them currently
    #     all_text += extract_text_from_url(url)

    # Process and store documents in FAISS
    if chunk_document:
        text_chunks = get_text_chunks(all_text, chunking_enabled=True) # Chunking enabled by default
        print(f"Text chunking ENABLED. Number of text chunks created: {len(text_chunks)}") # DEBUGGING STEP 5a: Chunking enabled message
    else:
        text_chunks = get_text_chunks(all_text, chunking_enabled=False) # Chunking disabled
        print(f"Text chunking DISABLED. Treating document as single chunk.") # DEBUGGING STEP 5b: Chunking disabled message


    update_vector_store(text_chunks)
    print("FAISS index updated.") # DEBUGGING STEP 6: Confirm FAISS update


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
    embeddings = GoogleGenerativeAIEmbeddings(google_api_key=google_api_key, model="models/embedding-001")
    vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

    print(f"User Submission for Similarity Search: '{user_submission}'") # DEBUGGING STEP 7: Print user submission
    # Retrieve relevant regulatory content for evaluation
    relevant_docs = vector_store.similarity_search(user_submission, k=25)
    print(f"Number of relevant documents retrieved: {len(relevant_docs)}") # DEBUGGING STEP 8: Check number of retrieved docs

    if not relevant_docs: # DEBUGGING STEP 9: Check if no docs are retrieved
        print("No relevant documents found by similarity search!")
    else:
        print("First retrieved document (for debugging):") # DEBUGGING STEP 10: Print content of first retrieved doc
        print(f"Page Content (first 200 chars): {relevant_docs[0].page_content[:200]}...")


    # Run compliance check
    chain = get_compliance_chain()
    response = chain({"input_documents": relevant_docs, "submission": user_submission}, return_only_outputs=True)

    return response, relevant_docs

# Run this once to store compliance documents permanently (Only run this once initially, or when you update documents)
# load_documents()  # Commented out after initial loading. Uncomment to reload documents if needed.


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
    use_chunking = True
    
    try:
        import faiss
    except ImportError:
        st.error("Please install FAISS first: pip install faiss-cpu")
        st.stop()
    
    # Load documents if ICDR-specific FAISS index doesn't exist
    if not os.path.exists(FAISS_INDEX_PATH):
        print(f"Initial FAISS index creation for ICDR at: {FAISS_INDEX_PATH}")
        if not load_documents(chunk_document=use_chunking):
            st.error("Failed to create ICDR FAISS index. Please check the errors above.")
            st.stop()
        print("ICDR FAISS index created.")
    
    run_chatbot()
