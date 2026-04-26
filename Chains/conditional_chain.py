from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnableBranch
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(repo_id='Qwen/Qwen2.5-72B-Instruct',
                          max_new_tokens=512,
                          task = 'text-generation')

model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(template='classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instruction}',
                         input_variables=['feedback'],
                         partial_variables={'format_instruction':parser2.get_format_instructions()})

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(template='write an appropriate response to pisitive feedback \n {feedback}',
                         input_variables = ['feedback'])
prompt3 = PromptTemplate(template='write an appropriate response to negative feedback \n {feedback}',
                         input_variables = ['feedback'])

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x:x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: 'could not find sentiment')
)

chain = classifier_chain | branch_chain

final_result = chain.invoke({'feedback':'this phone is wonderful but it has some bettry issue. but the gaming performance is high'})

print(final_result)