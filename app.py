import streamlit as st
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

# Set up the page
st.set_page_config(page_title="DnD DM Assistant", page_icon="🐉")
st.title("🐉 DnD 5e Rules Assistant")
st.markdown("Ask me any rules question, and I will cite the official sources!")

# --- The Latency Fix ---
# This decorator ensures the model is only loaded ONCE when the app starts.
@st.cache_resource
def load_vector_db():
    persist_directory = os.path.join(os.path.dirname(__file__), "chroma_db")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory=persist_directory, embedding_function=embedding_model)

# Initialize database and Gemini client
vector_db = load_vector_db()
client = genai.Client()

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("E.g., How do I grapple a creature?"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # 1. Retrieve context
        results = vector_db.similarity_search(prompt, k=3)
        context_text = "\n\n".join([doc.page_content for doc in results])
        
        # 2. Build the LLM prompt
        llm_prompt = f"""You are an expert Dungeons & Dragons Dungeon Master. 
        Answer the user's question clearly and concisely using ONLY the provided rulebook context below. 
        If the answer is not contained in the context, say "I cannot find the answer in the provided rulebooks."

        Context:
        {context_text}

        Question:
        {prompt}

        Answer:"""
        
        # 3. Call Gemini 3 Flash
        response = client.models.generate_content(
            model="gemini-3-flash-preview", 
            contents=llm_prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="low"),
                temperature=1.0 
            )
        )
        
        # Display the result
        message_placeholder.markdown(response.text)
        
    st.session_state.messages.append({"role": "assistant", "content": response.text})