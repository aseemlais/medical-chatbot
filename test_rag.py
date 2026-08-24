from llama_cpp import Llama
from langchain_community.vectorstores import Chroma

from src.helper import get_embeddings
from src.prompt import MEDICAL_RAG_PROMPT


CHROMA_PATH = "chroma_db"

MODEL_REPO = "hugging-quants/Llama-3.2-3B-Instruct-Q4_K_M-GGUF"
MODEL_FILE = "llama-3.2-3b-instruct-q4_k_m.gguf"


# -----------------------------
# Load embedding model
# -----------------------------

print("Loading embedding model...")

embeddings = get_embeddings()


# -----------------------------
# Load ChromaDB
# -----------------------------

print("Loading ChromaDB...")

vectorstore = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings
)


# -----------------------------
# Load Llama
# -----------------------------

print("Loading Llama...")

llm = Llama.from_pretrained(
    repo_id=MODEL_REPO,
    filename=MODEL_FILE,
    n_ctx=2048,
    verbose=False
)

print("Everything loaded successfully!")


# -----------------------------
# User question
# -----------------------------

question = "What is diabetes mellitus?"


# -----------------------------
# Retrieve relevant documents
# -----------------------------

print("Searching medical database...")

documents = vectorstore.similarity_search(
    question,
    k=3
)


# -----------------------------
# Create context
# -----------------------------

context = "\n\n".join(
    document.page_content
    for document in documents
)


# -----------------------------
# Create prompt
# -----------------------------

prompt = MEDICAL_RAG_PROMPT.format(
    context=context,
    question=question
)


# -----------------------------
# Send prompt to Llama
# -----------------------------

print("Generating answer...")

response = llm.create_chat_completion(
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
    max_tokens=300,
    temperature=0
)


# -----------------------------
# Extract answer
# -----------------------------

answer = response["choices"][0]["message"]["content"]


# -----------------------------
# Display result
# -----------------------------

print("\n========== QUESTION ==========\n")

print(question)


print("\n========== ANSWER ==========\n")

print(answer)
