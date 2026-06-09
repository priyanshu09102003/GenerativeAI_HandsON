#JOKE GENERATOR with Explanation
from langchain_google_genai import ChatGoogleGenerativeAI 
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence, RunnableParallel

load_dotenv()

prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")
parser = StrOutputParser()

prompt2 = PromptTemplate(
    template='Explain this joke in easy language - {text}',
    input_variables=['text']
)

joke_gen_chain = RunnableSequence(prompt1, model, parser)
parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2, model, parser)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)
result = final_chain.invoke({'topic': 'football'})

print(result['joke'])
print()
print(result['explanation'])