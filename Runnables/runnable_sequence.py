from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(repo_id='Qwen/Qwen2.5-72B-Instruct',
                          task='text-generation',
                          max_new_tokens=512)

model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()


prompt = PromptTemplate(template='make a joke about the {topic}',
                        input_variables=['topic'])

prompt2 = PromptTemplate(template='explain the following joke - {text}',
                         input_variables=['text'])

chain = RunnableSequence(prompt | model | parser | prompt2 | model | parser)

result = chain.invoke({'topic':'cricket'})

print(result)