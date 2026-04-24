from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

llm = HuggingFaceEndpoint(repo_id='Qwen/Qwen2.5-72B-Instruct',
                          task='text-generation',
                          max_new_tokens=512)

model = ChatHuggingFace(llm = llm)

chat_template = ChatPromptTemplate(
    [('system', 'you are {domain} expert'),
    ('human', 'tell me about {topic} used in cricket')])

prompt = chat_template.invoke({'domain':'cricket', 'topic':'free-hit'})

print(prompt)