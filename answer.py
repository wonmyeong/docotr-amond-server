import os
from flask import Flask, jsonify, request
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough

# Flask 앱 설정
app = Flask(__name__)

llm = ChatOpenAI(temperature=0.3, streaming=True, callbacks=[StreamingStdOutCallbackHandler()])

def embed_file(file_path):
    file_name = os.path.basename(file_path)
    cache_dir = LocalFileStore(f"./embeddings_cache/{file_name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

temp = 40
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
            Context : {context}
            """
        ),
        ("human", "{question}")
    ]
)

# Specify the file path you want to use
file_path = ".cache\\files\\dog_health.txt"  # Adjust this path as needed

retriever = embed_file(file_path)

chain = ({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough(),
}
| prompt
| llm)

@app.route('/ask', methods=['GET'])
def ask_question():
    doctor_question = "강아지 체온이" + str(temp) + "도이고 혈압은 정상이야 어떻게 해야해"
    response = chain.invoke(doctor_question)
    
    # response는 'AIMessageChunk' 객체이므로 text 속성을 직접 접근합니다.
    response_text = response.content
    
    return jsonify({"response": response_text})

if __name__ == '__main__':
    app.run(debug=True)
