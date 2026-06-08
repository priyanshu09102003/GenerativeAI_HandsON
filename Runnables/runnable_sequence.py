#Sequential chain to demonstrate the use case of Runnable_Sequence

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

load_dotenv()

prompt1 = PromptTemplate(
    template='Write a short joke on {topic}',
    input_variables=['topic']
)

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")
parser = StrOutputParser()

prompt2 = PromptTemplate(
    template='Explain the joke in simple terms\n {text}',
    input_variables=['text']
)

chain = RunnableSequence(prompt1 , model, parser, prompt2, model, parser)

result = chain.invoke({'topic' : 'AI'})

print(result)
