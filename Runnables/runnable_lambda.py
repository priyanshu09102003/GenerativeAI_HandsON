#Converting a WORD COUNT function to a runnable to count the number of words in a joke

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableParallel, RunnableLambda
from dotenv import load_dotenv

load_dotenv()

def word_count(text):
    return len(text.split())


prompt = PromptTemplate(
    template='Write 1 hilarious joke about {topic}',
    input_variables=['topic']
)

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')
parser = StrOutputParser()

joke_generation_chain = RunnableSequence(prompt, model, parser)

parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'word_count' : RunnableLambda(word_count)
})


final_chain = RunnableSequence(joke_generation_chain, parallel_chain)

result = final_chain.invoke({'topic' : 'Politics'})

print(result)