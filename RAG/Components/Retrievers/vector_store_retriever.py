from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

#Creating documents
documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]

embedding_model = GoogleGenerativeAIEmbeddings(model = "models/gemini-embedding-001")

vectorStore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_name="my_collection"
)

retriever = vectorStore.as_retriever(search_kwargs={"k" : 2})
query = "What is ChromaDB used for?"
results = retriever.invoke(query)


for i, doc in enumerate(results):
    print(f"--- Document {i+1} ---")
    print(doc.page_content)
    print()