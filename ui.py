import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API keys
GROQCHAT_API_KEY = "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"
GEMINI_API_KEY = "AIzaSyA3I3JhnKntMwUZEWTW7f5nvH2iBx_rjWM"

if not GROQCHAT_API_KEY or not GEMINI_API_KEY:
    st.error("Server error")
    st.stop()

os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# Initialize LLM
llm = ChatGroq(
    groq_api_key=GROQCHAT_API_KEY,
    model_name="llama-3.1-8b-instant"
)

# Load FAISS index
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

try:
    vectors = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
except Exception as e:
    st.error(f"Error loading FAISS index: {e}")
    st.stop()

# Create Retriever from FAISS
retriever = vectors.as_retriever()

# Define Prompt
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Provide the most accurate response based on the user's question.
    Suggest the best option based on cost and quantity.

    <context>
    {context}
    </context>
    Question: {input}
    """
)

# Create Retrieval Chain
documents_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, documents_chain)

# Streamlit UI
st.set_page_config(page_title="Tendercuts bot", page_icon="🤖")
st.title("🔍 Best Suggest Bot for Tendercuts")
st.header("Buy fresh Flesh🥩 at Tendercuts with Tenders..")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_query = st.chat_input("Flesh suggestion ....")
if user_query:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Retrieve response
    response = retrieval_chain.invoke({'input': user_query})
    bot_response = response['answer']

    # Append bot response
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    with st.chat_message("assistant"):
        st.markdown(bot_response)
