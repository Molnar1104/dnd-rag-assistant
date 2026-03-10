import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from text_chunker import chunk_dnd_text

def build_vector_db():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_txt = os.path.join(script_dir, "..", "data", "extracted_rules.txt")
    persist_directory = os.path.join(script_dir, "..", "chroma_db")

    print("Fetching text chunks...")
    chunks = chunk_dnd_text(input_txt)
    
    if not chunks:
        return

    print("Downloading/Initializing EmbeddingGemma 300M...")
    
    # We added "show_progress_bar": True to the encode_kwargs!
    # Move the progress bar instruction outside of encode_kwargs
    embedding_model = HuggingFaceEmbeddings(
        model_name="google/embeddinggemma-300m",
        encode_kwargs={"prompt_name": "Retrieval-document"},
        show_progress=True  # <--- LangChain handles it directly here!
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