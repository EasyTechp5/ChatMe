import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

# Get Google API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found. Please set it in your .env file.")

print("✅ Using Google API Key:", GOOGLE_API_KEY)

# Set default model (can change to "gemini-1.5-turbo" if quota is exceeded)
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-1.5-flash")


def load_and_store_pdf(pdf_path, persist_directory="chroma_db"):
    """Load PDF, store embeddings in Chroma, and create a RetrievalQA chain with custom prompt"""
    
    # --- 1️⃣ Read PDF ---
    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        if page.extract_text():
            text += page.extract_text()
    
    # --- 2️⃣ Split text into chunks ---
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    
    # --- 3️⃣ Create embeddings ---
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    
    # --- 4️⃣ Create Chroma vectorstore ---
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # --- 5️⃣ Create Google Gemini LLM ---
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=0.3,
        google_api_key=GOOGLE_API_KEY
    )
    
    # --- 6️⃣ Define prompt template ---
    template = """
You are a helpful and professional customer support assistant with 10 years experience for EasyTech Academy.
Always answer clearly, politely, and in a professional tone.

If the user asks something unrelated to EasyTech Academy, politely decline.

Kindly structure your response in a way that is easy to understand, you can include paragraphs, bullet points, and numbered lists where necessary.
Keep answers simple and straightforward.

If greeting, please respond politely and tell them they are welcome to EasyTech Academy, and ask how you can help.

Please use paragraphs where needed, make it neat and clear.

Please stop saying 'based on our records', also do not mention knowledge base or context.

Use the following context from our knowledge base to answer the question.
If the answer is not in the context, say "I'm sorry, kindly send your email."

Context: {context}
Question: {question}
Answer:
"""
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )
    
    # --- 7️⃣ Create RetrievalQA chain ---
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain


def query_bot(qa_chain, user_query):
    """Query the QA chain and return the answer"""
    result = qa_chain({"query": user_query})
    return result["result"]
