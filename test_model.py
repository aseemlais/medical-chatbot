import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto"
)

print("Model loaded successfully!")

prompt = "Explain what artificial intelligence is in simple words."

inputs = tokenizer(
    prompt,
    return_tensors="pt"
)

start_time = time.time()

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False
    )

end_time = time.time()

response = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print("\n========== RESPONSE ==========\n")
print(response)

print("\n========== PERFORMANCE ==========")
print(f"Generation time: {end_time - start_time:.2f} seconds")
print(f"Generated tokens: {outputs.shape[1] - inputs['input_ids'].shape[1]}")
