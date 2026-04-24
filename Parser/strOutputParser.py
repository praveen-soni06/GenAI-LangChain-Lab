from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(repo_id='Qwen/Qwen2.5-72B-Instruct',\
                          max_new_tokens=512,
                          task='text-generation')

model = ChatHuggingFace(llm = llm)

template1 = PromptTemplate(template='write detailed report on {topic}',
                           input_variables=['topic'])

template2 = PromptTemplate(template='write a 5 line summary on the following text. \n {text}',
                           input_variables=['text'])

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic': "black hole"})

print(result)