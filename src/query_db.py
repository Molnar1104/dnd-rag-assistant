import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DISTANCE_THRESHOLD = 1.2

def test_query():
    """Loads the vector database and runs a test similarity search with relevance filtering."""
    # 1. Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    persist_directory = os.path.join(script_dir, "..", "chroma_db")

    if not os.path.exists(persist_directory):
        print("Error: Chroma DB folder not found. Did you run vector_db.py?")
        return

    # 2. Load the same Embedding Model
    print("Loading HuggingFace Embedding Model...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 3. Load the persisted database
    print("Loading local Vector Database...")
    vector_db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )

    # 4. Define our test question
    query = "How do I grapple a creature?"
    print(f"\nSearching for: '{query}'\n")
    print("-" * 40)

    # 5. Perform the Similarity Search
    results = vector_db.similarity_search_with_score(query, k=5)

    if not results:
        print("No results found.")
        return

    # 6. Filter by relevance score threshold
    filtered_results = [(doc, score) for doc, score in results if score <= DISTANCE_THRESHOLD]

    if len(filtered_results) < len(results):
        print(f"⚠ Warning: {len(results) - len(filtered_results)} of {len(results)} chunks "
              f"exceeded the distance threshold of {DISTANCE_THRESHOLD} and were filtered out.")

    if not filtered_results:
        print("No results passed the relevance threshold.")
        return

    # 7. Print the results
    for i, (doc, score) in enumerate(filtered_results):
        # The score is a distance metric (often L2 distance). Lower is usually better!
        print(f"--- Result {i+1} (Distance Score: {score:.4f}) ---")
        print(doc.page_content)
        print("-" * 40 + "\n")

if __name__ == "__main__":
    test_query()