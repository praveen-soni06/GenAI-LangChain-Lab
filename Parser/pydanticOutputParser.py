from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import Field, BaseModel
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(repo_id='Qwen/Qwen2.5-72B-Instruct',
                          task='text-generation',
                          max_new_tokens=512)

model = ChatHuggingFace(llm=llm)

class person(BaseModel):

    name: str = Field(description='name of the person')
    age: int = Field(gt=18, description='age of that person')
    place: str = Field(description='plaec of the person belongs to')

parser = PydanticOutputParser(pydantic_object=person)

template = PromptTemplate(template='generate the name, age and city of a fictional {place} person \n {format_instruction}',
                          input_variables=['place'],
                          partial_variables={'format_instruction':parser.get_format_instructions()})

chain = template | model | parser

result = chain.invoke({'place': 'bihar'})

print(result)