import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# New official SDK imports
from google import genai 
from google.genai import types 

# 1. Load the secret API key from the .env file
load_dotenv()

def ask_dnd_assistant(query):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    persist_directory = os.path.join(script_dir, "..", "chroma_db")

    # 2. Load the Vector Database (LangChain is still great for this part)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)

    # 3. Retrieve the top 3 relevant chunks
    results = vector_db.similarity_search(query, k=3)
    
    # Combine the text of the chunks into one big string
    context_text = "\n\n".join([doc.page_content for doc in results])

    # 4. Initialize the official Gemini 3 Client
    # It automatically detects the GEMINI_API_KEY from your .env file
    client = genai.Client()

    # 5. Create the prompt 
    # Because Gemini 3 is a reasoning model, we use precise, direct instructions[cite: 619].
    # We also place the specific question at the end, after the data context[cite: 623].
    prompt = f"""You are an expert Dungeons & Dragons Dungeon Master. 
Answer the user's question clearly and concisely using ONLY the provided rulebook context below. 
If the answer is not contained in the context, say "I cannot find the answer in the provided rulebooks."

Context:
{context_text}

Question:
{query}

Answer:"""

    print("Thinking...\n" + "-"*40)
    
    # 6. Call the Gemini 3 API [cite: 112]
    response = client.models.generate_content(
        model="gemini-3-flash-preview", 
        contents=prompt,
        config=types.GenerateContentConfig(
            # Setting thinking to low minimizes latency and cost for simple chat tasks 
            thinking_config=types.ThinkingConfig(thinking_level="low"),
            # We keep temperature at the recommended default of 1.0 for Gemini 3 [cite: 151]
            temperature=1.0 
        )
    )
    
    print(f"Question: {query}\n")
    print(f"DM Assistant:\n{response.text}")
    print("-" * 40)

if __name__ == "__main__":
    test_question = "How do I grapple a creature?"
    ask_dnd_assistant(test_question)