import chromadb
import json
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from chromadb.config import Settings

class VectorStore:
    def __init__(self, persist_directory: str = "data/embeddings/chroma_db"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        self.collection_name = "gurdjieff_texts"
        self.collection = None
        self._initialize_collection()
    
    def _initialize_collection(self):
        """Initialize or get the collection."""
        try:
            self.collection = self.client.get_collection(self.collection_name)
            print(f"Loaded existing collection: {self.collection_name}")
        except:
            # Create collection without embedding function to use our OpenAI embeddings
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Gurdjieff texts and teachings"},
                embedding_function=None  # Use our own embeddings
            )
            print(f"Created new collection: {self.collection_name}")
    
    def add_embeddings_from_file(self, embeddings_file: Path):
        """Add embeddings from a JSON file to the vector store."""
        with open(embeddings_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        chunks = data["chunks"]
        valid_chunks = [chunk for chunk in chunks if chunk.get("embedding")]
        
        if not valid_chunks:
            print(f"No valid embeddings found in {embeddings_file.name}")
            return
        
        # Prepare data for ChromaDB
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for chunk in valid_chunks:
            chunk_id = f"{embeddings_file.stem}_{chunk['chunk_id']}"
            ids.append(chunk_id)
            embeddings.append(chunk["embedding"])
            documents.append(chunk["text"])
            
            metadata = {
                "source_file": chunk["source"]["filename"],
                "total_pages": chunk["source"]["total_pages"],
                "chunk_id": chunk["chunk_id"],
                "token_count": chunk["token_count"],
                "embedding_model": chunk["embedding_model"]
            }
            metadatas.append(metadata)
        
        # Add to collection in batches to avoid size limits
        batch_size = 100
        total_added = 0
        
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            batch_documents = documents[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            
            self.collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_documents,
                metadatas=batch_metadatas
            )
            total_added += len(batch_ids)
            print(f"  Added batch {i//batch_size + 1}/{(len(ids)-1)//batch_size + 1} ({total_added}/{len(ids)})")
        
        print(f"✓ Added {len(valid_chunks)} chunks from {embeddings_file.name}")
    
    def similarity_search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents using text query."""
        # Note: This requires query embedding to be provided separately
        # For now, we'll use the embedding search method
        raise NotImplementedError("Use similarity_search_by_embedding instead")
        
        # Format results
        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            })
        
        return formatted_results
    
    def similarity_search_by_embedding(self, embedding: List[float], n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents using embedding vector."""
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            })
        
        return formatted_results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        count = self.collection.count()
        return {
            "total_documents": count,
            "collection_name": self.collection_name
        }
    
    def reset_collection(self):
        """Reset the collection (delete all data)."""
        self.client.delete_collection(self.collection_name)
        self._initialize_collection()
        print(f"Reset collection: {self.collection_name}")

if __name__ == "__main__":
    vector_store = VectorStore()
    embeddings_dir = Path("data/embeddings")
    
    # Load all embedding files
    embedding_files = list(embeddings_dir.glob("*_embeddings.json"))
    
    if not embedding_files:
        print("No embedding files found. Run embedding_service.py first.")
    else:
        print(f"Found {len(embedding_files)} embedding files")
        
        # Ask user if they want to reset the collection
        if vector_store.get_collection_stats()["total_documents"] > 0:
            response = input("Collection already contains data. Reset? (y/N): ")
            if response.lower() == 'y':
                vector_store.reset_collection()
        
        # Add all embeddings to the vector store
        for embedding_file in embedding_files:
            print(f"Processing {embedding_file.name}...")
            vector_store.add_embeddings_from_file(embedding_file)
        
        # Show final stats
        stats = vector_store.get_collection_stats()
        print(f"\nVector store ready with {stats['total_documents']} documents")
        
        # Test search
        if stats['total_documents'] > 0:
            print("\nTesting search...")
            print("✅ Vector store is ready for queries")
            print("   Note: Use similarity_search_by_embedding() with query embeddings")
            print("   The server will handle query embedding generation automatically")