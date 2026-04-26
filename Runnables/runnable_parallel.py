from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(repo_id='Qwen/Qwen2.5-72B-Instruct',
                          task='text-generation',
                          max_new_tokens=512)

model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()


prompt = PromptTemplate(template='generate a tweet about {topic}',
                        input_variables=['topic'])

prompt2 = PromptTemplate(template='generate a linkedin post about {topic}',
                         input_variables=['topic'])

paralle_chain = RunnableParallel({'tweet': RunnableSequence(prompt | model | parser),
                          'linkedin': RunnableSequence(prompt2 | model | parser)})


result = paralle_chain.invoke({'topic':'AI'})

print('tweet post: \n',result['tweet'])
print('\n\nLinkedin post: \n', result['linkedin'])