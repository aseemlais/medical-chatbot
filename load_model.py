from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto"
)

print("Model loaded successfully!")
