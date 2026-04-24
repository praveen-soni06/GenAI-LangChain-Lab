from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(repo_id='Qwen/Qwen2.5-72B-Instruct',
                          task='text-generation',
                          max_new_tokens=512)

model = ChatHuggingFace(llm=llm)
