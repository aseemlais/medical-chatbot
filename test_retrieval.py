from src.helper import get_embeddings
from langchain_community.vectorstores import Chroma


CHROMA_PATH = "chroma_db"


print("Loading embedding model...")

embeddings = get_embeddings()

print("Loading ChromaDB...")

vectorstore = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings
)


query = "What are the symptoms of diabetes?"

print("\nQuery:")
print(query)


results = vectorstore.similarity_search(
    query,
    k=3
)


print("\n========== RETRIEVED DOCUMENTS ==========\n")


for i, document in enumerate(results, start=1):

    print(f"----- Result {i} -----")

    print("Page:", document.metadata.get("page"))

    print("\nContent:")

    print(document.page_content[:1000])

    print("\n")

    