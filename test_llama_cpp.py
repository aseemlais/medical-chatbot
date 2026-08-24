from llama_cpp import Llama
import time

MODEL_REPO = "hugging-quants/Llama-3.2-3B-Instruct-Q4_K_M-GGUF"
MODEL_FILE = "llama-3.2-3b-instruct-q4_k_m.gguf"

print("Loading quantized model...")

llm = Llama.from_pretrained(
    repo_id=MODEL_REPO,
    filename=MODEL_FILE,
    n_ctx=2048,
    verbose=False
)

print("Model loaded successfully!")

messages = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant. Give clear and concise answers."
    },
    {
        "role": "user",
        "content": "Explain what artificial intelligence is in simple words."
    }
]

start_time = time.time()

response = llm.create_chat_completion(
    messages=messages,
    max_tokens=50,
    temperature=0
)

end_time = time.time()

answer = response["choices"][0]["message"]["content"]

print("\n========== RESPONSE ==========\n")
print(answer)

print("\n========== PERFORMANCE ==========")
print(f"Generation time: {end_time - start_time:.2f} seconds")
print(
    f"Generated tokens: "
    f"{response['usage']['completion_tokens']}"
)

