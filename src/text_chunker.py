import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_dnd_text(input_txt_path):
    """
    Reads a cleaned text file and splits it into overlapping chunks.
    """
    if not os.path.exists(input_txt_path):
        print(f"Error: Could not find {input_txt_path}")
        return []

    print(f"Reading {input_txt_path}...")
    with open(input_txt_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Initialize the chunker
    # chunk_size: How many characters per chunk (approx. 200-250 words)
    # chunk_overlap: How many characters to overlap so we don't lose context between chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )

    # Split the text
    chunks = text_splitter.create_documents([text])
    print(f"Success. Split the rulebook into {len(chunks)} overlapping chunks.")

    # Second pass: print chunk statistics for tuning
    if chunks:
        chunk_lengths = [len(c.page_content) for c in chunks]
        avg_length = sum(chunk_lengths) / len(chunk_lengths)
        print(f"\n--- CHUNK STATISTICS ---")
        print(f"Total chunks: {len(chunks)}")
        print(f"Average chunk length: {avg_length:.0f} characters")
        print(f"Min: {min(chunk_lengths)} / Max: {max(chunk_lengths)} characters")
        print("------------------------\n")

    # Let's print the first chunk just to verify it looks clean
    if chunks:
        print("--- SAMPLE CHUNK ---")
        print(chunks[0].page_content)
        print("--------------------\n")

    return chunks

if __name__ == "__main__":
    # Dynamically build paths to avoid Windows relative path issues
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_txt = os.path.join(script_dir, "..", "data", "extracted_rules.txt")
    
    # Run the chunker
    document_chunks = chunk_dnd_text(input_txt)