import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from pymongo import MongoClient
from dotenv import load_dotenv
import time


# ------Page Setup---------







# Load environment variables
import os
load_dotenv()

# Load API keys
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
mongo_uri = os.getenv("MONGO_URI2")  # MongoDB URI


# Initialize the LLM (GROQ API)
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-70b-versatile")


# Initialize MongoDB client

client = MongoClient(mongo_uri)
db = client["krishnaji_chatbot"]  # Database name
collection = db["bhagavad_gita"]  # Collection name


embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Define the chat prompt for Lord Krishna
prompt = ChatPromptTemplate.from_template(
"""
You are Lord Krishna, the divine teacher and guide from the Mahabharata.
You will answer the user's question with the wisdom of Sanskrit verses, giving multiple Sanskrit verses relevant to the context.
In a new line, for each verse provide its meaning, followed by an interpretation that relates to the user's input on new line.
Finally, cite the source of the Sanskrit verses.
You dont have to always give answers in sanskrit verses. Only give sanskrit verse references and meanings when relevant.
You can also give answers in other languages, but make sure to cite the source.
You are given the following context from the document.
<context>
{context}
</context>
Questions: {input}

"""
)


# Streamlit input for user's question
prompt1 = st.chat_input("Sit, relax and ask any question you have for Lord Krishna")

# Initialize vector search using MongoDBAtlasVectorSearch with Google Generative AI Embeddings
vectorStore = MongoDBAtlasVectorSearch(
    collection, embeddings, index_name="vector_index"
)


compressor = LLMChainExtractor.from_llm(llm)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
document_chain=create_stuff_documents_chain(llm,prompt)
compression_retriever = ContextualCompressionRetriever(
base_compressor=compressor,
base_retriever=vectorStore.as_retriever()
)
retrieval_chain=create_retrieval_chain(compression_retriever,document_chain)


# Handle user input
if prompt1:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt1})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt1)
    
    start=time.process_time()
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response=retrieval_chain.invoke({'input':prompt1})
        
     # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response['answer']})
    print("Response time :",time.process_time()-start)
    st.write(response['answer'])
    
    with st.expander("Document Similarity Search"):
            # Find the relevant chunks
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")

