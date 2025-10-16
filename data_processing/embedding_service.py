import openai
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

class EmbeddingService:
    def __init__(self, model="text-embedding-ada-002"):
        self.model = model
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Generate embeddings for multiple texts in batches."""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)
                
                print(f"Generated embeddings for batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
                
            except Exception as e:
                print(f"Error generating embeddings for batch {i//batch_size + 1}: {e}")
                # Add empty embeddings for failed batch
                embeddings.extend([[] for _ in batch])
        
        return embeddings
    
    def process_chunks_file(self, chunks_file: Path) -> Dict[str, Any]:
        """Process chunks file and generate embeddings."""
        with open(chunks_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        chunks = data["chunks"]
        texts = [chunk["text"] for chunk in chunks]
        
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.generate_embeddings_batch(texts)
        
        # Add embeddings to chunks
        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[i]
            chunk["embedding_model"] = self.model
        
        # Update data with embedding info
        data["embedding_info"] = {
            "model": self.model,
            "total_embeddings": len([e for e in embeddings if e]),
            "failed_embeddings": len([e for e in embeddings if not e])
        }
        
        return data
    
    def save_embeddings(self, data: Dict[str, Any], output_file: Path):
        """Save chunks with embeddings to file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please create a .env file with your OpenAI API key.")
        exit(1)
    
    embedding_service = EmbeddingService()
    processed_dir = Path("data/processed")
    embeddings_dir = Path("data/embeddings")
    embeddings_dir.mkdir(exist_ok=True)
    
    chunks_files = list(processed_dir.glob("*_chunks.json"))
    
    if not chunks_files:
        print("No chunk files found. Run text_chunker.py first.")
    else:
        for chunks_file in chunks_files:
            print(f"Processing {chunks_file.name}...")
            
            try:
                data_with_embeddings = embedding_service.process_chunks_file(chunks_file)
                
                output_file = embeddings_dir / f"{chunks_file.stem}_embeddings.json"
                embedding_service.save_embeddings(data_with_embeddings, output_file)
                
                info = data_with_embeddings["embedding_info"]
                print(f"✓ Generated {info['total_embeddings']} embeddings")
                if info['failed_embeddings'] > 0:
                    print(f"⚠ {info['failed_embeddings']} embeddings failed")
                print(f"Saved to {output_file.name}\n")
                
            except Exception as e:
                print(f"✗ Failed to process {chunks_file.name}: {e}\n")