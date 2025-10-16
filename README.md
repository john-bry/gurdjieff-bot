# Gurdjieff Bot

A RAG-powered chatbot for exploring the works of George Gurdjieff using LLM and vector search.

## Project Structure

- `data_processing/` - Scripts for PDF extraction, text cleaning, and chunking
- `server/` - FastAPI backend with RAG implementation
- `interface/` - Simple web frontend (HTML/CSS/jQuery)
- `data/raw/` - Original PDF files
- `data/processed/` - Cleaned and chunked text
- `data/embeddings/` - Vector database storage

## Tech Stack

- **LLM**: OpenAI GPT models
- **Vector DB**: ChromaDB
- **Backend**: FastAPI (Python)
- **Frontend**: HTML/CSS/jQuery
- **PDF Processing**: PyPDF2/pdfplumber

## Getting Started

1. Place your Gurdjieff PDF files in `data/raw/`
2. Install dependencies: `pip install -r requirements.txt`
3. Run data processing: `python data_processing/process_pdfs.py`
4. Start server: `uvicorn server.main:app --reload`
5. Open `interface/index.html` in browser