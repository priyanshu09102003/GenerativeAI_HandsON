from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

chat_model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')
result = chat_model.invoke("What is the capital of Australia?")
print(result.content)