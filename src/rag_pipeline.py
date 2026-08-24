from llama_cpp import Llama
from langchain_community.vectorstores import Chroma

from src.helper import get_embeddings
from src.prompt import MEDICAL_RAG_PROMPT


CHROMA_PATH = "chroma_db"

MODEL_REPO = "hugging-quants/Llama-3.2-3B-Instruct-Q4_K_M-GGUF"
MODEL_FILE = "llama-3.2-3b-instruct-q4_k_m.gguf"


class RAGPipeline:

    def __init__(self):

        print("Loading embedding model...")
        self.embeddings = get_embeddings()

        print("Loading ChromaDB...")
        self.vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=self.embeddings
        )

        print("Loading Llama...")
        self.llm = Llama.from_pretrained(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            n_ctx=2048,
            verbose=False
        )

        print("RAG pipeline loaded successfully!")

    def get_answer(self, question):

        documents = self.vectorstore.similarity_search(
            question,
            k=3
        )

        context = "\n\n".join(
            document.page_content
            for document in documents
        )

        prompt = MEDICAL_RAG_PROMPT.format(
            context=context,
            question=question
        )

        response = self.llm.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=300,
            temperature=0
        )

        answer = response["choices"][0]["message"]["content"]

        return answer

    