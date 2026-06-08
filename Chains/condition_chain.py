from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser 
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.runnables import RunnableBranch, RunnableLambda

load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")
parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment : Literal['POSITIVE', 'NEGATIVE'] = Field(description= "Give the sentiment of the feedback")

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into POSITIVE or NEGATIVE\n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction': parser2.get_format_instructions}
)


classifier_chain = prompt1 | model | parser2

customer_feedback = """This is a terrible smartphone. It heats up so fast and never gives a long battery life"""

sentiment_classification = classifier_chain.invoke({'feedback' : customer_feedback}).sentiment

print(sentiment_classification)


#Prompting the agent to respond to the feedback if it is classified as Positive
prompt_response_Positive = PromptTemplate(
    template='You are a feedback-responding agent. Write an appropriate reply to this positive feedback as a customer care agent so that the customer is satisfied. \n {feedback}',
    input_variables=['feedback']
)

#Prompting the agent to respond to the feedback if it is classified as Negative
prompt_response_Negative = PromptTemplate(
    template='You are a feedback-responding agent. Write an appropriate reply to this negative feedback as a customer care agent so that the customer is satisfied. \n {feedback}',
    input_variables=['feedback']
)

branched_chain = RunnableBranch(
    (lambda x : x.sentiment == 'POSITIVE' , prompt_response_Positive | model | parser),
    (lambda x : x.sentiment == 'NEGATIVE' , prompt_response_Negative | model | parser),

    RunnableLambda(lambda x : "Kindly talk to the customer support team") 
    #When the agent is unable to classify clearly - the sentiment
)


#Final chain
chain = classifier_chain | branched_chain

response = chain.invoke({'feedback' : customer_feedback})

print(response)

