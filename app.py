import os
from flask import Flask, request, jsonify, render_template
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Initialize Flask app
app = Flask(__name__, template_folder="templates")

# Hardcoded API key (for testing purposes only)
OPENAI_API_KEY = ''

# Directory containing processed text chunks
data_folder = r"C:\Users\mir48\Downloads\chatbot_data\output_chunks"
faiss_index_folder = "faiss_index"  # Folder where FAISS index is stored


# Function to recreate the FAISS index if it doesn't exist
def create_faiss_index():
    from langchain_community.vectorstores import FAISS
    texts = []
    metadata = []
    for file_name in os.listdir(data_folder):
        if file_name.endswith(".txt"):
            with open(os.path.join(data_folder, file_name), "r", encoding="utf-8") as f:
                texts.append(f.read())
                metadata.append({"source": file_name})

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadata)
    vector_store.save_local(faiss_index_folder)
    print("FAISS index created and saved.")


# Load FAISS index and initialize the QA chain
def load_qa_chain():
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Check if FAISS index exists
    if not os.path.exists(f"{faiss_index_folder}/index.faiss"):
        print("FAISS index not found. Recreating...")
        create_faiss_index()

    # Load FAISS index
    vector_store = FAISS.load_local(faiss_index_folder, embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    llm = ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=OPENAI_API_KEY)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)


qa_chain = load_qa_chain()


@app.route('/')
def home():
    return render_template('index.html')
    return "Welcome to the Chatbot API! Use the /chat endpoint for interaction."

@app.route('/chat', methods=['POST'])
def chat():
    if request.method == 'GET':
        return "This endpoint only supports POST requests. Please send a POST request.", 405

    data = request.json
    query = data.get("message", "")
    if not query:
        return jsonify({"error": "No query provided"}), 400
    try:
        response = qa_chain.run(query)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
