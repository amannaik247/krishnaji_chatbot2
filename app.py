import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.title("Chai with Lord Krishna")

# Initialize the LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-70b-versatile")

# Chat prompt template
prompt_template = ChatPromptTemplate.from_template(
    """
    You are Lord Krishna. Lord Krishna will answer the questions based on the given context.
    You will also give the source of the Sanskrit text from which you have taken the answer.
    You are given the following context from the document.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

# Function to prepare embeddings and vector store
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader("./BhagwadGeeta")  # Data ingestion
        st.session_state.docs = st.session_state.loader.load()  # Document loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])  # Document splitting
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector creation

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input text box for user prompt
user_input = st.text_input("First prepare Chai, then enter your question")

# Button to prepare the embeddings (once)
if st.button("Prepare Chai"):
    vector_embedding()
    st.write("Your chai is ready, ask anything!")

# Function to display chat history
def display_chat_history():
    for i, (user_msg, response) in enumerate(st.session_state.chat_history):
        st.write(f"**You:** {user_msg}")
        st.write(f"**Lord Krishna:** {response}")
        st.write("---")

# If there is user input, process the chat
if user_input:
    # Update the chat history
    display_chat_history()

    # Prepare the retrieval chain
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    # Process user input and get response
    start = time.process_time()
    response = retrieval_chain.invoke({'input': user_input})
    response_time = time.process_time() - start
    st.session_state.chat_history.append((user_input, response['answer']))
    
    # Display response in the chat format
    display_chat_history()
    
    # With a Streamlit expander for document similarity search
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")

    # Print response time
    print("Response time:", response_time)
