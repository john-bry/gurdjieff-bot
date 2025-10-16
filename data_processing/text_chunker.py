import re
import json
import tiktoken
from pathlib import Path
from typing import List, Dict, Any

class TextChunker:
    def __init__(self, chunk_size=1000, chunk_overlap=200, model="text-embedding-ada-002"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.encoding_for_model(model)
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page headers/footers patterns
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove common OCR artifacts
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\'\"]', ' ', text)
        
        # Normalize quotes
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r"[''']", "'", text)
        
        return text.strip()
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences while preserving context."""
        # Simple sentence splitting (can be improved with spacy/nltk)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.encoding.encode(text))
    
    def create_chunks(self, text: str, source_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create overlapping chunks from text."""
        cleaned_text = self.clean_text(text)
        sentences = self.split_into_sentences(cleaned_text)
        
        chunks = []
        current_chunk = ""
        current_sentences = []
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if self.count_tokens(potential_chunk) > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append({
                    "text": current_chunk.strip(),
                    "token_count": self.count_tokens(current_chunk),
                    "source": source_info,
                    "chunk_id": len(chunks)
                })
                
                # Start new chunk with overlap
                overlap_text = self._create_overlap(current_sentences)
                current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                current_sentences = [sentence]
            else:
                current_chunk = potential_chunk
                current_sentences.append(sentence)
        
        # Add final chunk if it exists
        if current_chunk.strip():
            chunks.append({
                "text": current_chunk.strip(),
                "token_count": self.count_tokens(current_chunk),
                "source": source_info,
                "chunk_id": len(chunks)
            })
        
        return chunks
    
    def _create_overlap(self, sentences: List[str]) -> str:
        """Create overlap text from the end of current sentences."""
        if not sentences:
            return ""
        
        overlap_text = ""
        overlap_tokens = 0
        
        # Add sentences from the end until we reach overlap size
        for sentence in reversed(sentences):
            potential_overlap = sentence + " " + overlap_text if overlap_text else sentence
            tokens = self.count_tokens(potential_overlap)
            
            if tokens <= self.chunk_overlap:
                overlap_text = potential_overlap
                overlap_tokens = tokens
            else:
                break
        
        return overlap_text.strip()
    
    def process_extracted_json(self, json_file_path: Path) -> List[Dict[str, Any]]:
        """Process extracted JSON file and create chunks."""
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        all_chunks = []
        
        # Combine all pages into one text for better chunking
        full_text = ""
        for page_data in data["pages"]:
            full_text += page_data["text"] + "\n\n"
        
        source_info = {
            "filename": data["source"],
            "total_pages": data["total_pages"],
            "processed_file": str(json_file_path)
        }
        
        chunks = self.create_chunks(full_text, source_info)
        all_chunks.extend(chunks)
        
        return all_chunks
    
    def save_chunks(self, chunks: List[Dict[str, Any]], output_file: Path):
        """Save chunks to JSON file."""
        chunk_data = {
            "total_chunks": len(chunks),
            "chunking_config": {
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap
            },
            "chunks": chunks
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    chunker = TextChunker()
    processed_dir = Path("data/processed")
    
    json_files = list(processed_dir.glob("*_extracted.json"))
    
    if not json_files:
        print("No extracted JSON files found. Run pdf_extractor.py first.")
    else:
        for json_file in json_files:
            print(f"Chunking {json_file.name}...")
            chunks = chunker.process_extracted_json(json_file)
            
            output_file = processed_dir / f"{json_file.stem.replace('_extracted', '_chunks')}.json"
            chunker.save_chunks(chunks, output_file)
            
            print(f"✓ Created {len(chunks)} chunks, saved to {output_file.name}")