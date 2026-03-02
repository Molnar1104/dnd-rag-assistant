import fitz  # PyMuPDF
import os
import re

def extract_text_from_dnd_pdf(pdf_path, output_path):
    """
    Extracts text from a multi-column PDF rulebook using block extraction.
    """
    if not os.path.exists(pdf_path):
        print(f"Error: Could not find {pdf_path}")
        return

    print(f"Opening {pdf_path}...")
    doc = fitz.open(pdf_path)
    extracted_text = ""

    # Iterate through each page
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        # Extract text as blocks to maintain column reading order
        blocks = page.get_text("blocks")
        
        # Sort blocks vertically (y-axis) and then horizontally (x-axis) because of the way dnd rulebooks position text
        # This helps ensure we read down a column before jumping to the next
        blocks.sort(key=lambda b: (b[1], b[0])) 

        for block in blocks:
            text = block[4] 
            
            # Clean up weird PDF unicode separators (Line Separator, Paragraph Separator, Non-breaking space)
            text = text.replace('\u2028', ' ').replace('\u2029', ' \n\n')
            text = text.replace('\xa0', ' ')
            
            # Clean up standard mid-sentence line breaks
            text = text.replace('\n', ' ').strip()
            
            # Collapse multiple spaces into a single space
            text = re.sub(r' +', ' ', text)
            
            if text: 
                extracted_text += text + "\n\n"
                
    # Save the cleaned text
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(extracted_text)
        
    print(f"Success! Extracted {len(doc)} pages to {output_path}")

if __name__ == "__main__":
    # Get the directory where THIS script is located (the 'src' folder)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the path accurately relative to the script
    input_pdf = os.path.join(script_dir, "..", "data", "sample_rulebook.pdf")
    output_txt = os.path.join(script_dir, "..", "data", "extracted_rules.txt")
    
    extract_text_from_dnd_pdf(input_pdf, output_txt)