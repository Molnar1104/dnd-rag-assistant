import fitz  # PyMuPDF
import os
import re
import glob

def extract_text_from_all_pdfs(data_dir, output_path):
    all_text = ""
    # Find all PDF files in the data directory
    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
    
    if not pdf_files:
        print(f"Error: No PDFs found in {data_dir}")
        return

    print(f"Found {len(pdf_files)} rulebooks. Starting extraction...\n" + "-"*40)

    for pdf_path in pdf_files:
        print(f"Processing: {os.path.basename(pdf_path)}...")
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                blocks = page.get_text("blocks")
                # Sort blocks top-to-bottom, left-to-right (handles multi-column DnD layouts)
                blocks.sort(key=lambda b: (b[1], b[0])) 

                for block in blocks:
                    text = block[4] 
                    
                    # Clean up PDF unicode artifacts
                    text = text.replace('\u2028', ' ').replace('\u2029', ' \n\n')
                    text = text.replace('\xa0', ' ')
                    text = text.replace('\n', ' ').strip()
                    text = re.sub(r' +', ' ', text)
                    
                    if text: 
                        all_text += text + "\n\n"
            print(f"✔ Successfully extracted {len(doc)} pages.")
        except Exception as e:
            print(f"Error reading {os.path.basename(pdf_path)}: {e}")
        
    # Save the combined text
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(all_text)
    print("-" * 40 + f"\nSuccess! Combined all text into {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data")
    output_txt = os.path.join(script_dir, "..", "data", "extracted_rules.txt")
    
    extract_text_from_all_pdfs(data_dir, output_txt)