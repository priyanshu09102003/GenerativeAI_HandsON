#Take a topic from user and make a sequential chain and then do the following
# Detailed Report of the Topic -> Summarize tbe Detailed  - 5points (All these in a single cycle using chain)


from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

#Prompt1 -> To get the detailed report

prompt1 = PromptTemplate(
    template = 'Generate an extensively detailed report on {topic}',
    input_variables=['topic']
)

#Prompt2 -> To summarize the detailed report and get 5 important pointers
prompt2 = PromptTemplate(
    template = 'Extract 5 most important points from the following text and generate a 5 pointer summary for the text \n {text}',

    input_variables = ['text']
)

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')
parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser #This is a sequential chain as we are calling the LLM more than once

result = chain.invoke({'topic' : 'Asteroids in space'})


print(result)

chain.get_graph().print_ascii() #This function prints the entire chain