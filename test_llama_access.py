from huggingface_hub import model_info

model_name = "meta-llama/Llama-3.2-3B-Instruct"

info = model_info(model_name)

print("Model access successful!")
print("Model:", info.id)
