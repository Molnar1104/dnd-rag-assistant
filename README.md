---
title: DnD DM Assistant
emoji: 🐉
colorFrom: red
colorTo: yellow
sdk: streamlit
sdk_version: 1.55.0
app_file: app.py
pinned: false
---

# DnD 5e RAG Assistant

A Retrieval-Augmented Generation (RAG) application that acts as an expert Dungeons & Dragons Dungeon Master. It answers complex rules questions by semantically searching official rulebook text and synthesizing the results using Google's Gemini LLM.

**[Try the Live Demo Here](https://huggingface.co/spaces/Molki/dnd-rag-assistant)**

## Architecture & Tech Stack
This project uses a modern AI/Data Engineering pipeline to ensure the LLM never hallucinates rules, strictly grounding its answers in provided rulebook chunks.

* **Frontend UI:** Streamlit
* **LLM / Generation:** Google Gemini 3 Flash (via official `google-genai` SDK)
* **Vector Database:** ChromaDB (Local SQLite implementation)
* **Embeddings:** HuggingFace `all-MiniLM-L6-v2`
* **Orchestration / Text Splitters:** LangChain
* **CI/CD:** GitHub Actions (Automated sync to Hugging Face Spaces) & Git LFS for large database files.

## How it Works
1. **Data Extraction:** Raw DnD PDFs are parsed and cleaned using PyMuPDF.
2. **Chunking:** The text is split into overlapping 1000-character chunks to preserve rule context.
3. **Retrieval:** User queries are embedded, and ChromaDB retrieves the top 3 most mathematically relevant chunks.
4. **Generation:** Gemini reads only the retrieved chunks and formulates a clear, conversational answer.