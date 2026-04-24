from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(repo_id='Qwen/Qwen2.5-72B-Instruct',
                          task='text-generation',
                          max_new_tokens=512)

model = ChatHuggingFace(llm = llm)

prompt = PromptTemplate(template='Generate 5 interesting facts about {topic}',
                        input_variables=['topic'])

parser = StrOutputParser()

# creating chain/pipelines
chain = prompt | model | parser

result = chain.invoke({'topic':'uno'})

print(result)