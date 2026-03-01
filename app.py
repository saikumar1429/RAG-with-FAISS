import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def generate_embeddings(texts, _model):
    return _model.encode(texts)

def load_documents_from_file(uploaded_file):
    content = uploaded_file.read().decode("utf-8")
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    documents = [f"{lines[i]} {lines[i+1]}" for i in range(0, len(lines) - 1, 2)]
    return documents

def create_faiss_index(_model, documents):
    document_embeddings = generate_embeddings(documents, _model)
    d = document_embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(np.array(document_embeddings).astype("float32"))
    return index, document_embeddings

def retrieve(query, index, documents, _model, top_k=2):
    query_embeddings = generate_embeddings([query], _model)
    distances, indices = index.search(np.array(query_embeddings).astype("float32"), top_k)
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if 0 <= idx < len(documents):
            results.append((documents[idx], dist))
    return results

def main():
    st.title("RAG application with Faiss")
    st.markdown("Upload a `.txt` file with Q&A pairs (2 lines per pair).")
    
    uploaded_file = st.file_uploader("Upload your text file", type="txt")
    
    if uploaded_file:
        documents = load_documents_from_file(uploaded_file)
        st.write(f"Loaded {len(documents)} document pairs.")
        
        _model = load_model()
        index, _ = create_faiss_index(_model, documents)
        
        query = st.text_input("Ask a question:")
        
        if query:
            results = retrieve(query, index, documents, _model)
            st.subheader("🔎 Search Results:")
            for doc, score in results:
                st.write(f"**Score**: {score:.4f}")
                st.write(doc)
                st.markdown("---")

if __name__ == "__main__":
    main()
