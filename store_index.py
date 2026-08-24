from langchain_community.vectorstores import Chroma

from src.helper import (
    load_pdf,
    split_documents,
    get_embeddings
)


PDF_PATH = "data/Medical_book.pdf"

CHROMA_PATH = "chroma_db"


print("Loading medical book...")

documents = load_pdf(PDF_PATH)

print(f"Loaded {len(documents)} pages")


print("Splitting documents...")

chunks = split_documents(documents)

print(f"Created {len(chunks)} chunks")


print("Loading embedding model...")

embeddings = get_embeddings()

print("Creating ChromaDB vector store...")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_PATH
)

print("ChromaDB created successfully!")
print(f"Stored at: {CHROMA_PATH}")
