import os
import streamlit as st
from processor import DocumentProcessor
from model import LegalQAModel
from transformers import AutoModel

def main():
    st.set_page_config(page_title="AI Legal Assistant", page_icon="⚖️")
    
    st.title("⚖️ AI Legal Assistant")
    
    
    st.sidebar.header("Document Management")
    uploaded_files = st.sidebar.file_uploader(
        "Upload Legal Documents", 
        type=['pdf'], 
        accept_multiple_files=True
    )
    
    # Initialize processors
    doc_processor = DocumentProcessor()
    
    # Load model
    model_path = "nlpaueb/legal-bert-base-uncased"
    model = AutoModel.from_pretrained(model_path)  # Load the model
    qa_model = LegalQAModel(model_path)  # Pass model to LegalQAModel
    
    # Document processing
    processed_docs = {}
    if uploaded_files:
        # Create data directory
        os.makedirs('data/legal_documents', exist_ok=True)
        
        # Save uploaded files
        for file in uploaded_files:
            file_path = os.path.join('data/legal_documents', file.name)
            with open(file_path, 'wb') as f:
                f.write(file.getvalue())
        
        # Process documents
        processed_docs = doc_processor.process_documents('data/legal_documents')
        st.sidebar.success(f"Processed {len(processed_docs)} documents")
    
    # Question answering section
    st.header("Legal Question Answering")
    user_question = st.text_input("Enter your legal question:")
    
    # Process question if documents are uploaded
    if user_question and processed_docs:
        # Find best matching document
        best_doc_name = None
        best_context = None
        highest_similarity = -1
        
        for doc_name, doc_info in processed_docs.items():
            # Find best matching chunk
            best_chunk, similarity = qa_model.find_best_match(
                user_question, 
                doc_info['chunks'], 
                doc_info['embeddings']
            )
            
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_doc_name = doc_name
                best_context = best_chunk
        
        # Generate answer
        if best_context:
            result = qa_model.answer_question(user_question, best_context)
            
            st.subheader("Answer")
            st.write(result['answer'])
            
            st.subheader("Source Information")
            st.write(f"Document: {best_doc_name}")
            st.write(f"Relevance: {highest_similarity:.2%}")
        else:
            st.warning("Could not find a relevant answer in the uploaded documents.")

if __name__ == "__main__":
    main()
