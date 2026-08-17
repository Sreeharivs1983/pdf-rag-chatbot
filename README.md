# TechNova NovaAI — RAG Company Knowledge Assistant

An AI-powered company knowledge assistant built using **Retrieval-Augmented Generation (RAG)**.

The system retrieves relevant information from a company knowledge base using semantic vector search and uses an LLM to generate grounded, natural-language answers.

## Live Demo

https://pdf-rag-chatbot-sreehari-vs.streamlit.app/

## GitHub

https://github.com/Sreeharivs1983/pdf-rag-chatbot

## Overview

The application allows users to ask questions about TechNova's company information, products, leadership, locations, hiring and other knowledge-base content.

Instead of sending the entire document to the LLM, the system:

1. Extracts text from the PDF.
2. Splits the text into smaller chunks.
3. Converts chunks into vector embeddings.
4. Stores the embeddings in a FAISS vector index.
5. Rewrites user questions into retrieval-friendly queries.
6. Retrieves the most relevant document chunks.
7. Filters results using similarity scoring.
8. Sends the relevant context to the LLM.
9. Generates a grounded answer based on the retrieved information.

## RAG Pipeline

```text
Company Knowledge PDF
        ↓
   PDF Extraction
        ↓
    Text Chunking
        ↓
    Embeddings
        ↓
   FAISS Index
        ↓
   User Question
        ↓
 Question Rewriting
        ↓
 Semantic Retrieval
        ↓
Similarity Filtering
        ↓
 Relevant Context
        ↓
    Groq LLM
        ↓
 Grounded Answer
