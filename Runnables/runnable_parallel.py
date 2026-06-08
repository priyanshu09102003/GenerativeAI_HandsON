#Parallel chain to demonstrate the use case of Runnable_Parallel

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence

load_dotenv()

prompt1 = PromptTemplate(
    template = 'Generate a short and engaging tweet about\n {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template = 'Generate an insightful LinkedIN post about\n {topic}',
    input_variables=['topic']
)

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")
parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(prompt1, model, parser),
    'linkedin' : RunnableSequence(prompt2, model, parser)
})

result = parallel_chain.invoke({'topic' : 'LangChain'})


print('Twitter Post: ', result['tweet'])
print()
print('LinkedIN Post: ', result['linkedin'])