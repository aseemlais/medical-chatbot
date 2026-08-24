from src.prompt import MEDICAL_RAG_PROMPT


context = """
Diabetes mellitus is a chronic condition in which the body
does not produce enough insulin or does not use insulin properly.
This can result in high blood sugar levels.
"""

question = "What is diabetes mellitus?"


prompt = MEDICAL_RAG_PROMPT.format(
    context=context,
    question=question
)


print("========== GENERATED PROMPT ==========\n")
print(prompt)
