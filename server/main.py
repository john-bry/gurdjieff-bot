from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import openai
import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from data_processing.vector_store import VectorStore
from data_processing.embedding_service import EmbeddingService

load_dotenv()

app = FastAPI(title="Gurdjieff Bot API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
vector_store = VectorStore()
embedding_service = EmbeddingService()
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ChatRequest(BaseModel):
    message: str
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.7

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]]
    token_usage: Optional[Dict[str, int]] = None

class SearchRequest(BaseModel):
    query: str
    n_results: Optional[int] = 5

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "healthy", "message": "Gurdjieff Bot API is running"}

@app.get("/stats")
async def get_stats():
    """Get vector store statistics."""
    try:
        stats = vector_store.get_collection_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """Search for relevant documents."""
    try:
        # Generate embedding for the query
        query_embedding = embedding_service.generate_embedding(request.query)
        if not query_embedding:
            raise HTTPException(status_code=500, detail="Failed to generate query embedding")
        
        results = vector_store.similarity_search_by_embedding(
            embedding=query_embedding,
            n_results=request.n_results
        )
        return SearchResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat_with_gurdjieff(request: ChatRequest):
    """Chat with the Gurdjieff bot using RAG."""
    try:
        # Generate embedding for the query
        query_embedding = embedding_service.generate_embedding(request.message)
        if not query_embedding:
            raise HTTPException(status_code=500, detail="Failed to generate query embedding")
        
        # Search for relevant context
        search_results = vector_store.similarity_search_by_embedding(
            embedding=query_embedding,
            n_results=5
        )
        
        # Prepare context from search results
        context_texts = []
        sources = []
        
        for result in search_results:
            if result["distance"] < 0.8:  # Filter by similarity threshold
                context_texts.append(result["text"])
                sources.append({
                    "source": result["metadata"]["source_file"],
                    "chunk_id": result["metadata"]["chunk_id"],
                    "distance": result["distance"],
                    "preview": result["text"][:200] + "..."
                })
        
        # Build the prompt
        context = "\n\n".join(context_texts[:3])  # Use top 3 most relevant
        
        system_prompt = """You are a knowledgeable assistant specializing in the teachings and works of George Ivanovich Gurdjieff. 

You have access to relevant excerpts from Gurdjieff's texts and teachings. Use this context to provide thoughtful, accurate responses that reflect his philosophy and methods.

Key principles to remember:
- Gurdjieff emphasized the importance of conscious effort and voluntary suffering
- He taught about the three centers of human functioning: thinking, feeling, and moving
- His work focused on self-observation, self-remembering, and awakening from mechanical behavior
- Be precise and avoid speculation beyond what the texts support
- If the context doesn't contain relevant information, say so clearly

Context from Gurdjieff's teachings:
{context}

Answer based on this context and your knowledge of Gurdjieff's work."""

        messages = [
            {"role": "system", "content": system_prompt.format(context=context)},
            {"role": "user", "content": request.message}
        ]
        
        # Get response from OpenAI
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        return ChatResponse(
            response=response.choices[0].message.content,
            sources=sources,
            token_usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.get("/health")
async def health_check():
    """Detailed health check."""
    health_status = {
        "api": "healthy",
        "openai_key": "configured" if os.getenv("OPENAI_API_KEY") else "missing",
        "vector_store": "unknown",
        "document_count": 0
    }
    
    try:
        stats = vector_store.get_collection_stats()
        health_status["vector_store"] = "healthy"
        health_status["document_count"] = stats["total_documents"]
    except Exception as e:
        health_status["vector_store"] = f"error: {str(e)}"
    
    return health_status

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)