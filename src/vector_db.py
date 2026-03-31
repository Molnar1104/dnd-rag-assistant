import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from text_chunker import chunk_dnd_text

def build_vector_db():
    """Chunks the extracted rulebook text, embeds it, and persists the vectors to ChromaDB."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_txt = os.path.join(script_dir, "..", "data", "extracted_rules.txt")
    persist_directory = os.path.join(script_dir, "..", "chroma_db")

    print("Fetching text chunks...")
    chunks = chunk_dnd_text(input_txt)

    if not chunks:
        return

    print("Downloading/Initializing all-MiniLM-L6-v2...")

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        show_progress=True
    )

    print(f"Embedding {len(chunks)} chunks and saving to {persist_directory}...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    print("Success! High-dimension vector database built and saved locally.")

if __name__ == "__main__":
    build_vector_db()