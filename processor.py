import os
import PyPDF2
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer

class DocumentProcessor:
    def __init__(self, model_name='nlpaueb/legal-bert-base-uncased'):
        """
        Initialize document processor with embedding model
        """
        self.embedding_model = SentenceTransformer(model_name)
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file
        """
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """
        Split text into chunks
        """
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            current_chunk.append(word)
            current_length += len(word)

            if current_length > chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks
    
    def process_documents(self, document_folder: str) -> Dict[str, Dict]:
        """
        Process documents in a folder
        """
        processed_docs = {}
        
        for filename in os.listdir(document_folder):
            if filename.endswith('.pdf'):
                filepath = os.path.join(document_folder, filename)
                
                # Extract text
                text = self.extract_text_from_pdf(filepath)
                
                # Chunk text
                chunks = self.chunk_text(text)
                
                # Generate embeddings
                embeddings = [self.embedding_model.encode(chunk) for chunk in chunks]
                
                processed_docs[filename] = {
                    'text': text,
                    'chunks': chunks,
                    'embeddings': embeddings
                }
        
        return processed_docs