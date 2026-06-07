#Take a topic from the user and generate 5 interesting points on the topic using a Simple Chain 

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    template='Generate 5 interesting and unknown facts about {topic}',
    input_variables=['topic']
)

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

parser = StrOutputParser()

chain = prompt | model | parser     #This is a simple chain

result = chain.invoke({'topic': 'Generative AI'})

print(result)