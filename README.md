
# 🩺 Medical RAG Chatbot using Llama 3.2, ChromaDB and Flask

## 📌 Project Overview

This project is an end-to-end Medical Retrieval-Augmented Generation (RAG) chatbot designed to provide context-grounded answers from a medical reference library.

The system uses the Gale Encyclopedia of Medicine as the knowledge source. The 637-page medical reference is processed into smaller chunks, converted into vector embeddings using Sentence-Transformers, and stored in ChromaDB for semantic retrieval.

A quantized Llama 3.2 3B model is used for CPU-efficient local inference through llama-cpp-python. When a user submits a query, the system retrieves relevant medical context from ChromaDB and passes it to the LLM to generate a grounded response.

A Flask backend and responsive web interface provide an interactive conversational experience.

---

## 🎯 Objectives

- Build an end-to-end Retrieval-Augmented Generation pipeline
- Process and retrieve information from a large medical reference
- Implement semantic document retrieval using embeddings
- Store and search document embeddings using ChromaDB
- Generate context-grounded responses using a local LLM
- Optimize LLM inference for CPU-based systems
- Benchmark original and quantized model inference
- Develop an interactive web interface using Flask

---

## 🛠️ Technologies Used

- Python
- Llama 3.2 3B
- GGUF Quantization
- llama-cpp-python
- LangChain
- Sentence-Transformers
- ChromaDB
- Flask
- HTML
- CSS
- JavaScript
- PyPDF

---

## 📂 Project Structure

```text
medical-chatbot/
│
├── data/
│   └── Medical_book.pdf
│
├── chroma_db/
│   └── ChromaDB vector store
│
├── src/
│   ├── helper.py
│   ├── prompt.py
│   └── rag_pipeline.py
│
├── templates/
│   └── chat.html
│
├── app.py
├── requirements.txt
├── setup.py
├── .env
├── .gitignore
└── README.md