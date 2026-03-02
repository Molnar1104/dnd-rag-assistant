import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from text_chunker import chunk_dnd_text

def build_vector_db():
    # 1. Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_txt = os.path.join(script_dir, "..", "data", "extracted_rules.txt")
    persist_directory = os.path.join(script_dir, "..", "chroma_db")

    # 2. Get the text chunks from our previous script
    print("Fetching text chunks...")
    chunks = chunk_dnd_text(input_txt)
    
    if not chunks:
        print("No chunks found. Exiting.")
        return

    # 3. Initialize the Embedding Model
    # We are using a fast, highly-rated open source model from HuggingFace
    print("Downloading/Initializing HuggingFace Embedding Model...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 4. Build and persist the Chroma Vector Database
    print(f"Embedding {len(chunks)} chunks and saving to {persist_directory}...")
    
    # This command creates the DB, embeds all the text, and saves it to the folder
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    
    print("Success! Vector database built and saved locally.")

if __name__ == "__main__":
    build_vector_db()