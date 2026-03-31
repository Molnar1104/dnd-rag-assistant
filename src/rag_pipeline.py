import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# New official SDK imports
from google import genai 
from google.genai import types 

# 1. Load the secret API key from the .env file
load_dotenv()

DISTANCE_THRESHOLD = 1.2

def ask_dnd_assistant(query):
    """Retrieves relevant rulebook chunks and generates an answer using Gemini."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    persist_directory = os.path.join(script_dir, "..", "chroma_db")

    # 2. Load the Vector Database (LangChain is still great for this part)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)

    # 3. Retrieve the top 5 relevant chunks with scores
    results = vector_db.similarity_search_with_score(query, k=5)

    # 4. Determine confidence based on distance scores
    all_below_threshold = all(score <= DISTANCE_THRESHOLD for _, score in results)
    some_below_threshold = any(score <= DISTANCE_THRESHOLD for _, score in results)

    if all_below_threshold:
        confidence = "High"
    elif some_below_threshold:
        confidence = "Medium"
    else:
        confidence = "Low"

    # 5. Build context with labeled chunk indices
    context_parts = []
    for i, (doc, score) in enumerate(results):
        context_parts.append(f"[Chunk {i+1}] (distance: {score:.4f})\n{doc.page_content}")
    context_text = "\n\n".join(context_parts)

    # 6. Initialize the official Gemini 3 Client
    client = genai.Client()

    # 7. Create the prompt with chunk indices and confidence instruction
    prompt = f"""You are an expert Dungeons & Dragons Dungeon Master.
Answer the user's question clearly and concisely using ONLY the provided rulebook context below.
When referencing information, cite the chunk it came from (e.g. [Chunk 1], [Chunk 2]).
If the answer is not contained in the context, say "I cannot find the answer in the provided rulebooks."

At the very end of your answer, add a line:
Confidence: {confidence}

Context:
{context_text}

Question:
{query}

Answer:"""

    print("Thinking...\n" + "-"*40)

    # 8. Call the Gemini 3 API
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="low"),
            temperature=1.0
        )
    )

    print(f"Question: {query}\n")
    print(f"DM Assistant:\n{response.text}")
    print("-" * 40)

if __name__ == "__main__":
    test_question = "How do I grapple a creature?"
    ask_dnd_assistant(test_question)