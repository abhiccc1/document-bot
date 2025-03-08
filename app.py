import bs4
import os
from flask import Flask, render_template, request, jsonify
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

app = Flask(__name__)

# Set your OpenAI API key
# os.environ["OPENAI_API_KEY"] = "sk-proj-jYXYHzg9BOUO02DW5k1RT3BlbkFJ07Oslv4oZQaDnda4ko2X"
os.environ["USER_AGENT"] = "DocBot/1.0"

# Initialize the RAG system
loader = WebBaseLoader("https://tradetron.tech/keyword/documentations")

# Load the documents
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# Initialize the language model separately
llm = ChatOpenAI(model="gpt-4o")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = hub.pull("rlm/rag-prompt")
# Initialize RetrievalQA correctly
qa = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    question = request.json['question']
    
    # Retrieve relevant documents
    retrieved_documents = qa.invoke(question)
    
    if retrieved_documents:
        print(retrieved_documents)
        # Generate answer using the language model
        answer = retrieved_documents
    else:
        answer = "Sorry, I couldn't find an answer to your question."

    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)
