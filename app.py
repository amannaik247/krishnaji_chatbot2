import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import time

# Load environment variables
load_dotenv()

# Load API keys
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
mongo_uri = os.getenv("MONGO_URI2")  # MongoDB URI

# Streamlit App Title
st.title("Chai with Lord Krishna")

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
You are given the following context from the document.
<context>
{context}
</context>
Questions: {input}

"""
)

# Streamlit input for user's question
prompt1 = st.text_input("Sit, relax and ask any question you have!")

# Initialize vector search using MongoDBAtlasVectorSearch with Google Generative AI Embeddings
vectorStore = MongoDBAtlasVectorSearch(
    collection, embeddings, index_name="vector_index"
)

docs = vectorStore.max_marginal_relevance_search(prompt1, K=1)
compressor = LLMChainExtractor.from_llm(llm)


# Handle user input
if prompt1 or st.button("Send"):
    document_chain=create_stuff_documents_chain(llm,prompt)
    compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorStore.as_retriever()
    )
    retrieval_chain=create_retrieval_chain(compression_retriever,document_chain)
    start=time.process_time()
    response=retrieval_chain.invoke({'input':prompt1})
    print("Response time :",time.process_time()-start)
    st.write(response['answer'])
    
    with st.expander("Document Similarity Search"):
            # Find the relevant chunks
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")

