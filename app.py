from flask import Flask, render_template, request, jsonify

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma


# ----------------------------------------
# CREATE FLASK APPLICATION
# ----------------------------------------

app = Flask(__name__)


# ----------------------------------------
# LOAD DOCUMENT
# ----------------------------------------

loader = TextLoader(
    "documents/knowledge.txt",
    encoding="utf-8"
)

documents = loader.load()


print("Document loaded successfully!")


# ----------------------------------------
# SPLIT DOCUMENT INTO CHUNKS
# ----------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)


print("Document split into chunks:", len(chunks))


# ----------------------------------------
# CREATE EMBEDDINGS
# ----------------------------------------

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


print("Embedding model loaded!")


# ----------------------------------------
# CREATE VECTOR DATABASE
# ----------------------------------------

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)


print("Vector database created!")


# ----------------------------------------
# LOAD LOCAL LLM
# ----------------------------------------

llm = OllamaLLM(
    model="llama3.2"
)


print("Llama 3.2 loaded!")


# ----------------------------------------
# HOME PAGE
# ----------------------------------------

@app.route("/")
def home():

    return render_template("index.html")


# ----------------------------------------
# ASK QUESTION
# ----------------------------------------

@app.route("/ask", methods=["POST"])
def ask():

    data = request.get_json()

    question = data.get("question", "").strip()

    if not question:

        return jsonify({
            "answer": "Please enter a question."
        })


    print("\nUser Question:")
    print(question)


    # ------------------------------------
    # RETRIEVE RELEVANT DOCUMENTS
    # ------------------------------------

    results = vectorstore.similarity_search(
        question,
        k=3
    )


    context = "\n\n".join(
        document.page_content
        for document in results
    )


    print("Relevant documents retrieved!")


    # ------------------------------------
    # CREATE RAG PROMPT
    # ------------------------------------

    prompt = f"""
You are AI Nexus, a local RAG-based AI assistant.

Use the following context to answer the question.

If the answer is not available in the context,
say that the information is not available
in the provided documents.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""


    # ------------------------------------
    # SEND PROMPT TO LOCAL LLM
    # ------------------------------------

    answer = llm.invoke(prompt)


    print("Answer generated!")


    return jsonify({
        "answer": answer
    })


# ----------------------------------------
# START SERVER
# ----------------------------------------

if __name__ == "__main__":

    print("\n===================================")
    print("       AI NEXUS STARTING")
    print("===================================")
    print("Local RAG + Ollama + LLM")
    print("Open: http://127.0.0.1:5000")
    print("===================================\n")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )