# RAG Application with FAISS and Sentence Transformers

This project implements a simple Retrieval-Augmented Generation (RAG) style application using FAISS for vector similarity search and Sentence Transformers for text embeddings. The application is built with Streamlit and allows users to upload a text file containing question-answer pairs for semantic retrieval.

---

## Project Overview

This application demonstrates how semantic search works using vector embeddings and FAISS indexing.

Workflow:

1. User uploads a text file containing Q&A pairs.
2. Each pair is converted into embeddings using a SentenceTransformer model.
3. FAISS creates a vector index for fast similarity search.
4. User enters a query.
5. The system retrieves the most similar document pairs based on vector distance.

This project focuses on the retrieval component of a RAG pipeline.

---

## Features

- Upload custom `.txt` document
- Automatic Q&A pair processing
- SentenceTransformer-based embeddings
- FAISS vector indexing
- Top-K similarity search
- Interactive Streamlit interface
- Cached model loading for better performance

---

## Technologies Used

- Python
- Streamlit
- SentenceTransformers
- FAISS
- NumPy

---

## Embedding Model

Model used:
all-MiniLM-L6-v2

This model converts text into dense vector embeddings for semantic similarity comparison.

---

## Input File Format

The uploaded `.txt` file must follow this format:

Each question and answer should be written on two consecutive lines.

Example:

What is AI?
Artificial Intelligence is the simulation of human intelligence in machines.

What is Machine Learning?
Machine Learning is a subset of AI that enables systems to learn from data.

Each pair of lines will be combined into a single document entry.

---

## Project Structure
