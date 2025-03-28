import torch
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer, AutoModelForQuestionAnswering

model_path = "/home/diti/legal-bert-base-uncased"  # Ensure this path exists

class LegalQAModel:
    def __init__(self, model_path: str):
        """
        Initialize Question Answering Model
        """
        self.embedding_model = SentenceTransformer(model_path)  # Use model path, not instance
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    
    def find_best_match(self, 
                         question: str, 
                         document_chunks: List[str], 
                         document_embeddings: List[np.ndarray]) -> Tuple[str, float]:
        """
        Find most relevant document chunk
        """
        question_embedding = self.embedding_model.encode(question)
        
        similarities = [np.dot(question_embedding, chunk_embedding) / 
                        (np.linalg.norm(question_embedding) * np.linalg.norm(chunk_embedding)) 
                        for chunk_embedding in document_embeddings]
        
        best_match_idx = np.argmax(similarities)
        return document_chunks[best_match_idx], similarities[best_match_idx]
    
    def answer_question(self, question: str, context: str) -> Dict[str, str]:
        """
        Generate answer based on context
        """
        inputs = self.tokenizer(
            question, 
            context, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        
        start_index = torch.argmax(start_logits)
        end_index = torch.argmax(end_logits)
        
        input_ids = inputs["input_ids"][0]
        answer_tokens = input_ids[start_index:end_index+1]
        answer = self.tokenizer.decode(answer_tokens)
        
        return {
            'answer': answer.strip(),
            'start_index': start_index.item(),
            'end_index': end_index.item()
        }
