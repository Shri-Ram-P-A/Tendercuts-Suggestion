# Tendercuts Bot - AI-Powered Flesh Suggestion Assistant

Tendercuts Bot is an AI-powered chatbot that provides the best recommendations for purchasing fresh meat based on cost and quantity. It utilizes retrieval-augmented generation (RAG) with FAISS for efficient vector storage and retrieval and integrates with Groq's Llama model and Gemini for enhanced language understanding.

## Features

- 🔍 **AI-Powered Recommendations**: Get accurate suggestions based on user queries and context.
- 🧠 **Retrieval-Augmented Generation (RAG)**: Uses FAISS for storing and retrieving relevant context efficiently.
- 🤖 **Groq Llama Model**: Utilizes the `llama-3.1-8b-instant` model for fast and effective responses.
- 📊 **Google Gemini Embeddings**: Leverages `GoogleGenerativeAIEmbeddings` for vector-based searches.
- 🛠 **Streamlit UI**: Simple and interactive chatbot interface for user-friendly experience.

## Installation

### Prerequisites

Ensure you have Python installed (>=3.8) and install the required dependencies using:

```bash
pip install -r requirements.txt
```

### Required API Keys

You need to set up API keys for Groq and Gemini.

- **Groq API Key**: `GROQCHAT_API_KEY`
- **Gemini API Key**: `GEMINI_API_KEY`

Store them in a `.env` file:

```plaintext
GROQCHAT_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
```

## Usage

Run the chatbot using Streamlit:

```bash
streamlit run app.py
```

## Project Structure

```
├── app.py                # Main Streamlit app
├── requirements.txt      # Required dependencies
├── .env                 # API keys (not included in repo)
├── faiss_index/         # FAISS vector storage
└── README.md            # Project documentation
```

## Technologies Used

- **Python**
- **Streamlit**
- **LangChain** (Groq & Gemini Integration)
- **FAISS** (Vector Search)
- **Google Generative AI** (Embeddings)

## Acknowledgments

Special thanks to [LangChain](https://www.langchain.com/) and [FAISS](https://faiss.ai/) for enabling efficient retrieval-augmented AI solutions.

---

### Author

Developed by Shriram P A 

