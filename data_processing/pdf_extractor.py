import pdfplumber
import os
import json
from pathlib import Path

class PDFExtractor:
    def __init__(self, raw_data_dir="data/raw", processed_data_dir="data/processed"):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file using pdfplumber."""
        text_content = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text:
                        text_content.append({
                            "page": page_num,
                            "text": text.strip()
                        })
                        
            return text_content
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return []
    
    def process_all_pdfs(self):
        """Process all PDF files in the raw data directory."""
        pdf_files = list(self.raw_data_dir.glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {self.raw_data_dir}")
            return
        
        results = {}
        
        for pdf_file in pdf_files:
            print(f"Processing {pdf_file.name}...")
            text_content = self.extract_text_from_pdf(pdf_file)
            
            if text_content:
                # Save extracted text as JSON
                output_file = self.processed_data_dir / f"{pdf_file.stem}_extracted.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "source": pdf_file.name,
                        "total_pages": len(text_content),
                        "pages": text_content
                    }, f, indent=2, ensure_ascii=False)
                
                # Save as plain text for easier processing
                text_file = self.processed_data_dir / f"{pdf_file.stem}_text.txt"
                with open(text_file, 'w', encoding='utf-8') as f:
                    for page_data in text_content:
                        f.write(f"--- Page {page_data['page']} ---\n")
                        f.write(page_data['text'])
                        f.write("\n\n")
                
                results[pdf_file.name] = {
                    "pages": len(text_content),
                    "json_file": str(output_file),
                    "text_file": str(text_file)
                }
                
                print(f"✓ Extracted {len(text_content)} pages from {pdf_file.name}")
            else:
                print(f"✗ Failed to extract text from {pdf_file.name}")
        
        return results

if __name__ == "__main__":
    extractor = PDFExtractor()
    results = extractor.process_all_pdfs()
    
    if results:
        print("\nExtraction Summary:")
        for filename, info in results.items():
            print(f"- {filename}: {info['pages']} pages")
    else:
        print("\nNo PDFs processed. Please add PDF files to the data/raw/ directory.")