#Poem summary by loading a text content

from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv


load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

prompt = PromptTemplate(
    template='Write a short summary for the following poem \n {poem}',
    input_variables=['poem']
)

parser = StrOutputParser()

#Document loader object
loader = TextLoader('cricket.txt', encoding='utf-8')

#loading the document
docs = loader.load()

print(docs[0].page_content)

chain = prompt | model | parser

summary = chain.invoke({'poem': docs[0].page_content})

print("Summary of the poem" , summary)


