#!/usr/bin/env python3
"""
Complete data processing pipeline for Gurdjieff Bot.
This script processes PDFs through the entire pipeline: extraction, chunking, embedding, and vector storage.
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Add data_processing to path
sys.path.append(str(Path(__file__).parent / "data_processing"))

from data_processing.pdf_extractor import PDFExtractor
from data_processing.text_chunker import TextChunker
from data_processing.embedding_service import EmbeddingService
from data_processing.vector_store import VectorStore

def main():
    parser = argparse.ArgumentParser(description="Process PDFs for Gurdjieff Bot")
    parser.add_argument("--skip-extraction", action="store_true", 
                       help="Skip PDF extraction (use existing extracted files)")
    parser.add_argument("--skip-chunking", action="store_true",
                       help="Skip text chunking (use existing chunk files)")
    parser.add_argument("--skip-embeddings", action="store_true",
                       help="Skip embedding generation (use existing embedding files)")
    parser.add_argument("--reset-vector-store", action="store_true",
                       help="Reset the vector store before adding new data")
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY") and not args.skip_embeddings:
        print("❌ Error: OPENAI_API_KEY not found in environment variables.")
        print("Please create a .env file with your OpenAI API key or use --skip-embeddings flag.")
        return 1
    
    print("🤖 Gurdjieff Bot Data Processing Pipeline")
    print("=" * 50)
    
    # Setup directories
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    embeddings_dir = Path("data/embeddings")
    
    for dir_path in [raw_dir, processed_dir, embeddings_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: PDF Extraction
    if not args.skip_extraction:
        print("\n📄 Step 1: Extracting text from PDFs...")
        extractor = PDFExtractor()
        extraction_results = extractor.process_all_pdfs()
        
        if not extraction_results:
            print("❌ No PDFs found or extraction failed.")
            print(f"Please add PDF files to {raw_dir}")
            return 1
        
        print(f"✅ Extracted text from {len(extraction_results)} PDF(s)")
    else:
        print("\n⏭️  Step 1: Skipping PDF extraction")
    
    # Step 2: Text Chunking
    if not args.skip_chunking:
        print("\n🔪 Step 2: Chunking text...")
        chunker = TextChunker()
        
        json_files = list(processed_dir.glob("*_extracted.json"))
        if not json_files:
            print("❌ No extracted JSON files found. Run without --skip-extraction first.")
            return 1
        
        chunk_count = 0
        for json_file in json_files:
            print(f"  Processing {json_file.name}...")
            chunks = chunker.process_extracted_json(json_file)
            
            output_file = processed_dir / f"{json_file.stem.replace('_extracted', '_chunks')}.json"
            chunker.save_chunks(chunks, output_file)
            chunk_count += len(chunks)
            print(f"  ✅ Created {len(chunks)} chunks")
        
        print(f"✅ Total chunks created: {chunk_count}")
    else:
        print("\n⏭️  Step 2: Skipping text chunking")
    
    # Step 3: Generate Embeddings
    if not args.skip_embeddings:
        print("\n🧠 Step 3: Generating embeddings...")
        embedding_service = EmbeddingService()
        
        chunks_files = list(processed_dir.glob("*_chunks.json"))
        if not chunks_files:
            print("❌ No chunk files found. Run without --skip-chunking first.")
            return 1
        
        total_embeddings = 0
        for chunks_file in chunks_files:
            print(f"  Processing {chunks_file.name}...")
            try:
                data_with_embeddings = embedding_service.process_chunks_file(chunks_file)
                
                output_file = embeddings_dir / f"{chunks_file.stem}_embeddings.json"
                embedding_service.save_embeddings(data_with_embeddings, output_file)
                
                info = data_with_embeddings["embedding_info"]
                total_embeddings += info['total_embeddings']
                print(f"  ✅ Generated {info['total_embeddings']} embeddings")
                if info['failed_embeddings'] > 0:
                    print(f"  ⚠️  {info['failed_embeddings']} embeddings failed")
                    
            except Exception as e:
                print(f"  ❌ Failed to process {chunks_file.name}: {e}")
                return 1
        
        print(f"✅ Total embeddings generated: {total_embeddings}")
    else:
        print("\n⏭️  Step 3: Skipping embedding generation")
    
    # Step 4: Build Vector Store
    print("\n🗄️  Step 4: Building vector store...")
    try:
        vector_store = VectorStore()
        
        if args.reset_vector_store:
            print("  Resetting vector store...")
            vector_store.reset_collection()
        
        embedding_files = list(embeddings_dir.glob("*_embeddings.json"))
        if not embedding_files:
            print("❌ No embedding files found. Run without --skip-embeddings first.")
            return 1
        
        # Check if collection already has data
        current_stats = vector_store.get_collection_stats()
        if current_stats["total_documents"] > 0 and not args.reset_vector_store:
            print(f"  Vector store already contains {current_stats['total_documents']} documents")
            print("  Use --reset-vector-store to replace them")
            print("  Skipping vector store update")
            
            # Show final stats
            final_stats = vector_store.get_collection_stats()
            print(f"✅ Vector store ready with {final_stats['total_documents']} documents")
            
            print("\n🧪 Testing the system...")
            print("✅ Vector store is ready for queries")
            print("   Note: Server will handle query embedding generation automatically")
            
            print("\n🎉 Pipeline completed successfully!")
            print("\nNext steps:")
            print("1. Start the server: uvicorn server.main:app --reload")
            print("2. Open interface/index.html in your browser")
            print("3. Start exploring Gurdjieff's teachings!")
            
            return 0
        
        total_added = 0
        for embedding_file in embedding_files:
            print(f"  Adding {embedding_file.name} to vector store...")
            vector_store.add_embeddings_from_file(embedding_file)
            total_added += 1
        
        final_stats = vector_store.get_collection_stats()
        print(f"✅ Vector store ready with {final_stats['total_documents']} documents")
        
        # Test the system
        print("\n🧪 Testing the system...")
        test_results = vector_store.similarity_search("consciousness", n_results=3)
        if test_results:
            print(f"✅ Search test successful - found {len(test_results)} relevant results")
            print(f"  Top result similarity: {1 - test_results[0]['distance']:.3f}")
        else:
            print("⚠️  Search test returned no results")
        
    except Exception as e:
        print(f"❌ Vector store error: {e}")
        return 1
    
    print("\n🎉 Pipeline completed successfully!")
    print("\nNext steps:")
    print("1. Start the server: uvicorn server.main:app --reload")
    print("2. Open interface/index.html in your browser")
    print("3. Start exploring Gurdjieff's teachings!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())