import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import re

GROQCHAT_API_KEY = "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"
GEMINI_API_KEY = "AIzaSyA3I3JhnKntMwUZEWTW7f5nvH2iBx_rjWM"

if not GROQCHAT_API_KEY or not GEMINI_API_KEY:
    st.error("Server error: Missing API keys")
    st.stop()

os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

llm = ChatGroq(
    groq_api_key=GROQCHAT_API_KEY,
    model_name="llama-3.1-8b-instant"
)

embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

try:
    vectors = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
except Exception as e:
    st.error(f"Error loading FAISS index: {e}")
    st.stop()

retriever = vectors.as_retriever()

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Provide the most accurate response based on the user's question.
    Suggest the best option based on cost and quantity.
    Provide the content with the image URL of the item and the website link.

    <context>
    {context}
    </context>
    Question: {input}
    """
)

documents_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, documents_chain)

st.set_page_config(page_title="Tendercuts Bot", page_icon="🥩")
st.title("🔍 Best Suggest Bot for Tendercuts")
st.header("Buy fresh Flesh🥩 at Tendercuts with Tenders..")

st.sidebar.header("Type")
selected_item = st.sidebar.radio("Choose your category:", ["Seafood", "Meat", "Chicken"], index=None)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "displayed" not in st.session_state:
    st.session_state.displayed = False


def clean_response(text):
    text = re.sub(r'\b(None|image|URL)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def display_response_with_images(response_text):
    image_pattern = r'https?://\S+\.(webp|png|jpg|jpeg)'
    link_pattern = r'https?://[^\s]+\b'

    response_text = clean_response(response_text)

    content_parts = re.split(f'({image_pattern}|{link_pattern})', response_text)

    for part in content_parts:
        if isinstance(part, str) and re.match(image_pattern, part):
            st.image(part, caption="Product Image", use_container_width=True)
        elif isinstance(part, str) and re.match(link_pattern, part):
            st.markdown(f"[Click Here]({part})", unsafe_allow_html=True)
        else:
            st.markdown(part)


if selected_item and not st.session_state.displayed:
    query = f"Tell me about {selected_item} options at Tendercuts."
    response = retrieval_chain.invoke({'input': query})
    bot_response = clean_response(response['answer'])

    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    st.session_state.displayed = True

    with st.chat_message("assistant"):
        display_response_with_images(bot_response)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_query = st.chat_input(f"Ask about ...")
if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    response = retrieval_chain.invoke({'input': user_query})
    bot_response = clean_response(response['answer'])

    st.session_state.messages.append({"role": "assistant", "content": bot_response})

    with st.chat_message("assistant"):
        display_response_with_images(bot_response)

    st.session_state.displayed = False
