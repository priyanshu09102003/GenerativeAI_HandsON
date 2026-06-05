from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddingModel = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")

result = embeddingModel.embed_query("Delhi is the capital of India")

print(str(result))
