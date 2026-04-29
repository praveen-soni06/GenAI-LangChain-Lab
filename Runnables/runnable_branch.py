from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableBranch, RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(repo_id='Qwen/Qwen2.5-7B-Instruct',
                          task='text-generation',
                          max_new_tokens=512)

model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()

prompt1 = PromptTemplate(template='write a detailed report on {topic}',
                         input_variables=['topic'])

prompt2 = PromptTemplate(template='summarize the following text \n {text}',
                         input_variables=['text'])

to_dict = RunnableLambda(lambda x: {"text": x})

report_gen_chain = RunnableSequence(prompt1, model, parser)

branch_chain = RunnableBranch(
                              (lambda x: len(x.split())>50, RunnableSequence(to_dict, prompt2, model, parser)),
                              RunnablePassthrough()
                              )

final_chain = RunnableSequence(report_gen_chain, branch_chain)

result = final_chain.invoke({'topic':'AI vs GenAI'})

print(result)