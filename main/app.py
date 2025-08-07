import os
import hashlib
import requests
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from urllib.parse import urlparse
import PyPDF2
import io
from sentence_transformers import SentenceTransformer
import pinecone
from groq import Groq
from flask import Flask, request, jsonify
import logging
from concurrent.futures import ThreadPoolExecutor
import json
import re
from dotenv import load_dotenv
load_dotenv()
import re
import json
from functools import wraps
import secrets


# Configuration
@dataclass
class Config:
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY")
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "rag-documents")
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # Local model for embeddings
    GROQ_MODEL: str = "llama-3.1-8b-instant"
    CHUNK_SIZE: int = 600
    CHUNK_OVERLAP: int = 100
    MAX_TOKENS: int = 4000
    API_KEY: str = os.getenv("RAG_API_KEY", "your_api_key")
config = Config()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    """Handles PDF download and text extraction"""
    
    @staticmethod
    def download_pdf(url: str) -> bytes:
        """Download PDF from URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Verify it's a PDF
            if not response.content.startswith(b'%PDF'):
                raise ValueError("Downloaded file is not a valid PDF")
            
            return response.content
        except Exception as e:
            logger.error(f"Error downloading PDF: {e}")
            raise
    
    @staticmethod
    def extract_text_from_pdf(pdf_content: bytes) -> str:
        """Extract text from PDF content"""
        try:
            pdf_file = io.BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text += f"\n--- Page {page_num + 1} ---\n"
                        text += page_text
                except Exception as e:
                    logger.warning(f"Error extracting text from page {page_num + 1}: {e}")
            
            if not text.strip():
                raise ValueError("No text could be extracted from PDF")
            
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise

class TextChunker:
    """Handles text chunking with overlap"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page markers
        text = re.sub(r'\n--- Page \d+ ---\n', '\n', text)
        
        return text.strip()
    
    @staticmethod
    def create_chunks(text: str, chunk_size: int = 400, overlap: int = 50) -> List[Dict[str, Any]]:
        """Create overlapping text chunks"""
        cleaned_text = TextChunker.clean_text(text)
        words = cleaned_text.split()
        
        if not words:
            return []
        
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_text = ' '.join(words[start:end])
            
            # Create chunk metadata
            chunk_data = {
                'id': f"chunk_{chunk_id}",
                'text': chunk_text,
                'start_word': start,
                'end_word': end,
                'word_count': len(chunk_text.split())
            }
            
            chunks.append(chunk_data)
            chunk_id += 1
            
            # Move start position with overlap
            if end >= len(words):
                break
            start = max(start + chunk_size - overlap, start + 1)
        
        logger.info(f"Created {len(chunks)} chunks from text")
        return chunks

class EmbeddingGenerator:
    """Generates embeddings using SentenceTransformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Loaded embedding model: {model_name}, dimension: {self.dimension}")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

class PineconeManager:
    """Manages Pinecone vector database operations"""
    
    def __init__(self):
        self.pc = pinecone.Pinecone(api_key=config.PINECONE_API_KEY)
        self.index_name = config.PINECONE_INDEX_NAME
        self.index = None
        self._ensure_index_exists()
    
    def _ensure_index_exists(self):
        """Create index if it doesn't exist"""
        try:
            # List existing indexes
            indexes = self.pc.list_indexes()
            index_names = [idx.name for idx in indexes]
            
            if self.index_name not in index_names:
                logger.info(f"Creating Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=384,  # Dimension for all-MiniLM-L6-v2
                    metric="cosine",
                    spec=pinecone.ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
            
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Error setting up Pinecone index: {e}")
            raise
    
    def store_vectors(self, document_id: str, chunks: List[Dict], embeddings: List[List[float]]):
        """Store vectors in Pinecone"""
        try:
            vectors_to_upsert = []
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vector_id = f"{document_id}_chunk_{i}"
                metadata = {
                    'document_id': document_id,
                    'chunk_id': chunk['id'],
                    'text': chunk['text'],
                    'word_count': chunk['word_count']
                }
                
                vectors_to_upsert.append({
                    'id': vector_id,
                    'values': embedding,
                    'metadata': metadata
                })
            
            # Upsert in batches of 100
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                self.index.upsert(vectors=batch)
            
            logger.info(f"Stored {len(vectors_to_upsert)} vectors for document {document_id}")
            
        except Exception as e:
            logger.error(f"Error storing vectors: {e}")
            raise
    
    def search_similar(self, query_embedding: List[float], top_k: int = 5, document_id: str = None) -> List[Dict]:
        """Search for similar vectors"""
        try:
            filter_dict = {'document_id': document_id} if document_id else None
            
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            
            return [
                {
                    'text': match.metadata['text'],
                    'score': match.score,
                    'chunk_id': match.metadata['chunk_id']
                }
                for match in results.matches
            ]
            
        except Exception as e:
            logger.error(f"Error searching vectors: {e}")
            raise

class RAGGenerator:
    """Handles RAG prompt generation and LLM interaction"""
    
    def __init__(self):
        self.client = Groq(api_key=config.GROQ_API_KEY)
    
    def create_rag_prompt(self, question: str, context_chunks: List[Dict]) -> str:
        """Create RAG prompt with context - IMPROVED VERSION"""
        context = "\n\n".join([
            f"Context {i+1}: {chunk['text']}"
            for i, chunk in enumerate(context_chunks)
        ])
        
        prompt = f"""You are an expert insurance policy analyst. Answer the following question based ONLY on the provided context from the insurance document. 

IMPORTANT INSTRUCTIONS:
- Be specific and include exact numbers, percentages, and timeframes mentioned in the context
- If specific details are mentioned, include them (e.g., "30 days", "36 months", "5%", etc.)
- Structure your answer clearly and comprehensively
- If the context contains the information but in different sections, synthesize it into a complete answer
- Only say "not specified" if the information is truly not available in any of the provided contexts
- Answer in a professional, clear manner suitable for insurance customers

Context from the insurance policy:
{context}

Question: {question}

Provide a detailed, accurate answer based on the context above. Include specific details, numbers, and conditions mentioned in the policy."""
        
        return prompt
    
    def generate_answer(self, question: str, context_chunks: List[Dict]) -> Dict[str, Any]:
        """Generate answer using Groq - IMPROVED VERSION"""
        try:
            prompt = self.create_rag_prompt(question, context_chunks)
            
            response = self.client.chat.completions.create(
                model=config.GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert insurance policy analyst who provides accurate, detailed answers based on policy documents."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=config.MAX_TOKENS,
                temperature=0.1,  # Lower temperature for more consistent answers
                top_p=0.9
            )
            
            # Get the raw answer
            answer_text = response.choices[0].message.content.strip()
            
            # Clean up the answer but preserve structure
            # Remove any JSON formatting if the model added it
            if answer_text.startswith('{') and answer_text.endswith('}'):
                try:
                    answer_json = json.loads(answer_text)
                    answer_text = answer_json.get("answer", answer_text)
                except json.JSONDecodeError:
                    pass
            
            return {
                "answer": answer_text,
                "confidence": "high",
                "sources_used": [f"context_{i+1}" for i in range(len(context_chunks))]
            }
                
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "confidence": "low",
                "sources_used": []
            }

class DocumentRAGSystem:
    """Main RAG system orchestrator"""
    
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.text_chunker = TextChunker()
        self.embedding_generator = EmbeddingGenerator(config.EMBEDDING_MODEL)
        self.pinecone_manager = PineconeManager()
        self.rag_generator = RAGGenerator()
    
    def generate_document_id(self, url: str) -> str:
        """Generate unique document ID from URL"""
        return hashlib.md5(url.encode()).hexdigest()
    
    def process_document(self, url: str) -> str:
        """Process document and store in vector database"""
        document_id = self.generate_document_id(url)
        
        try:
            logger.info(f"Processing document: {url}")
            
            # Download and extract text
            pdf_content = self.pdf_processor.download_pdf(url)
            text = self.pdf_processor.extract_text_from_pdf(pdf_content)
            
            # Create chunks
            chunks = self.text_chunker.create_chunks(
                text, 
                chunk_size=config.CHUNK_SIZE, 
                overlap=config.CHUNK_OVERLAP
            )
            
            if not chunks:
                raise ValueError("No chunks created from document")
            
            # Generate embeddings
            chunk_texts = [chunk['text'] for chunk in chunks]
            embeddings = self.embedding_generator.generate_embeddings(chunk_texts)
            
            # Store in Pinecone
            self.pinecone_manager.store_vectors(document_id, chunks, embeddings)
            
            logger.info(f"Successfully processed document {document_id}")
            return document_id
            
        except Exception as e:
            logger.error(f"Error processing document {url}: {e}")
            raise
    
    def answer_questions(self, document_url: str, questions: List[str]) -> List[Dict[str, Any]]:
        """Answer questions using RAG - IMPROVED VERSION"""
        document_id = self.generate_document_id(document_url)
        
        # First, ensure document is processed
        try:
            self.process_document(document_url)
        except Exception as e:
            logger.warning(f"Document processing failed, continuing: {e}")
        
        answers = []
        
        for question in questions:
            try:
                # Generate question embedding
                question_embedding = self.embedding_generator.generate_embeddings([question])[0]
                
                # Search for more relevant chunks with higher top_k
                relevant_chunks = self.pinecone_manager.search_similar(
                    question_embedding, 
                    top_k=8,  # Increased from 5 to get more context
                    document_id=document_id
                )
                
                if not relevant_chunks:
                    answers.append({
                        "answer": "No relevant information found in the document for this question.",
                        "confidence": "low",
                        "sources_used": []
                    })
                    continue
                
                # Filter chunks by relevance score (keep only highly relevant ones)
                filtered_chunks = [chunk for chunk in relevant_chunks if chunk['score'] > 0.7]
                
                if not filtered_chunks:
                    # If no high-score chunks, use top 3 from original results
                    filtered_chunks = relevant_chunks[:3]
                
                # Generate answer
                answer = self.rag_generator.generate_answer(question, filtered_chunks)
                answers.append(answer)
                
            except Exception as e:
                logger.error(f"Error answering question '{question}': {e}")
                answers.append({
                    "answer": f"Error processing question: {str(e)}",
                    "confidence": "low",
                    "sources_used": []
                })
        
        return answers

# Flask API
app = Flask(__name__)
rag_system = DocumentRAGSystem()

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        
        if not auth_header:
            return jsonify({'error': 'Authorization header is required'}), 401
        
        if not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Authorization header must start with "Bearer "'}), 401
        
        token = auth_header[7:]  # Remove "Bearer " prefix
        
        if token != config.API_KEY:
            return jsonify({'error': 'Invalid API key'}), 401
        
        return f(*args, **kwargs)
    return decorated_function
def process_answers_for_quality(raw_answers):
        """Process answers while preserving quality and detail"""
        processed_answers = []
    
        for ans in raw_answers:
            try:
                raw = ans['answer']

                # Remove only problematic markdown, preserve structure
                raw = raw.replace('```json', '').replace('```', '')
                raw = re.sub(r'\*\*(.*?)\*\*', r'\1', raw)  # Remove bold formatting
                raw = raw.replace('```', '')  # Remove any remaining code blocks

                # Try to extract JSON if it exists, but be more careful
                json_match = re.search(r'\{[^{}]*"answer"[^{}]*\}', raw, re.DOTALL)
                if json_match:
                    try:
                        json_obj = json.loads(json_match.group())
                        answer_text = json_obj.get("answer", raw).strip()
                    except json.JSONDecodeError:
                        answer_text = raw.strip()
                else:
                    answer_text = raw.strip()

            # Clean whitespace but preserve sentence structure
            # Remove excessive newlines but keep paragraph breaks
                answer_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', answer_text)  # Max 2 newlines
                answer_text = re.sub(r'[ \t]+', ' ', answer_text)  # Remove extra spaces/tabs
                answer_text = answer_text.strip()

                processed_answers.append(answer_text)

            except Exception as e:
                logger.error(f"Error processing answer: {e}")
                processed_answers.append("Could not parse answer.")
    
        return processed_answers

@app.route('/hackrx/run', methods=['POST'])
@require_api_key
def process_hackrx_request():
    """Main API endpoint - IMPROVED VERSION"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        document_url = data.get('documents')
        questions = data.get('questions', [])
        
        if not document_url:
            return jsonify({'error': 'No document URL provided'}), 400
        
        if not questions:
            return jsonify({'error': 'No questions provided'}), 400
        
        # Validate URL
        try:
            result = urlparse(document_url)
            if not all([result.scheme, result.netloc]):
                raise ValueError("Invalid URL")
        except Exception:
            return jsonify({'error': 'Invalid document URL'}), 400
        
        logger.info(f"Processing {len(questions)} questions for document")
        
        # Process questions
        raw_answers = rag_system.answer_questions(document_url, questions)
        
        # Process answers with improved quality preservation
        processed_answers = process_answers_for_quality(raw_answers)
        
        return jsonify({
            'answers': processed_answers
        })
        
    except Exception as e:
        logger.error(f"Error in API endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'RAG system is running'})

if __name__ == '__main__':
    # Validate required environment variables
    required_vars = ['GROQ_API_KEY', 'PINECONE_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        exit(1)
    
    logger.info("Starting RAG API server...")
    app.run(debug=True, host='0.0.0.0', port=5000)
