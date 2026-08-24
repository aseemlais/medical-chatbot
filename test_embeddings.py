from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print("Embedding model loaded!")

text = "Diabetes is a medical condition involving high blood sugar levels."

embedding = model.encode(text)

print("\nEmbedding shape:", embedding.shape)
print("First 10 values:", embedding[:10])
