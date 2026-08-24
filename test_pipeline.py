from src.rag_pipeline import RAGPipeline


print("Initializing RAG pipeline...")

rag = RAGPipeline()

question = "What is diabetes mellitus?"

answer = rag.get_answer(question)

print("\n========== QUESTION ==========\n")
print(question)

print("\n========== ANSWER ==========\n")
print(answer)
