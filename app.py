from flask import Flask, request, jsonify, render_template

from src.rag_pipeline import RAGPipeline


app = Flask(__name__)

print("Initializing RAG pipeline...")

rag = RAGPipeline()

print("RAG chatbot ready!")


@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/ask", methods=["POST"])
def ask():

    data = request.get_json()

    question = data.get("question", "").strip()

    if not question:
        return jsonify({
            "error": "Question cannot be empty."
        }), 400

    answer = rag.get_answer(question)

    return jsonify({
        "question": question,
        "answer": answer
    })


if __name__ == "__main__":
    app.run(debug=False)


