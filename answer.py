import os,time
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
from dotenv import load_dotenv
import cx_Oracle

# Load .env file
load_dotenv()

# Get OpenAI API key from environment variables
openai_api_key = os.getenv('OPENAI_API_KEY')

# Flask app setup
app = Flask(__name__)

llm = ChatOpenAI(temperature=0.3, streaming=True, callbacks=[StreamingStdOutCallbackHandler()], openai_api_key=openai_api_key)

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
pressure= "normal"
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

def insert_response_to_db(response_text, r_seq):
    # Database connection details
    ic_loc = r"C:\\Users\\wmk51\\FULLSTACK-GPT\\instantclient-basic-windows.x64-23.4.0.24.05\\instantclient_23_4"

    # Update the PATH environment variable to include the Instant Client location
    os.environ['PATH'] = ic_loc + ';' + os.environ['PATH']

    dsn = cx_Oracle.makedsn("3.35.77.53", 1521, service_name="xe")
    username = "dog"
    password = "1234"

    connection = cx_Oracle.connect(user=username, password=password, dsn=dsn)
    cursor = connection.cursor()

    # Insert statement
    insert_query = """
    INSERT INTO DOG_RECOM (R_SEQ, R_CONTENT) VALUES (:r_seq, :response_text)
    """
    
    # Execute the insert query with the appropriate bind variables
    cursor.execute(insert_query, {"r_seq": r_seq, "response_text": response_text})
    connection.commit()

    cursor.close()
    connection.close()    
    


@app.route('/ask', methods=['GET'])
def ask_question():
    doctor_question = "강아지 체온이" + str(temp) + "도이고 혈압은" + pressure + "이야 어떻게 해야해"
    response = chain.invoke(doctor_question)
    r_seq = int(time.time()) 
    # Extract text from the response
    response_text = response.content
    
    # Insert the response into the database
    insert_response_to_db(response_text,r_seq)
    
    return jsonify({"response": response_text})

if __name__ == '__main__':
    app.run(debug=True)