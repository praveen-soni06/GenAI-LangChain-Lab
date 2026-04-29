from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(repo_id='Qwen/Qwen2.5-72B-Instruct',
                          task='text-generation',
                          max_new_tokens=512)

model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()


prompt = PromptTemplate(template='write a joke about the {topic}',
                        input_variables=['topic'])

prompt2 = PromptTemplate(template='explain the following joke - {text}',
                         input_variables=['text'])

joke_gen_chain = RunnableSequence(prompt, model, parser)

parallel_chain = RunnableParallel({'joke': RunnablePassthrough(),
                                      'word_count': RunnableLambda(lambda s: len(s.split()))})


final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = final_chain.invoke({'topic': 'GenAI'})

print('Joke : \n',result['joke'])
print('\nword_count : \n', result['word_count'])