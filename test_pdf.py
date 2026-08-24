from src.helper import load_pdf, split_documents


PDF_PATH = "data/Medical_book.pdf"

documents = load_pdf(PDF_PATH)

print("Number of pages:", len(documents))

chunks = split_documents(documents)

print("Number of chunks:", len(chunks))

print("\n========== FIRST CHUNK ==========\n")
print(chunks[0].page_content)

print("\n========== FIRST CHUNK METADATA ==========\n")
print(chunks[0].metadata)
