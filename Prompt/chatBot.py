# from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# import model from other folder
from model import model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

# llm = HuggingFaceEndpoint(repo_id='Qwen/Qwen2.5-72B-Instruct',
#                           task='text-generation',
#                           max_new_tokens=512)

# model = ChatHuggingFace(llm=llm)

# maintain Chathisory for LLM
chat_history = [SystemMessage(content='You are a helpful AI Assistent')]

while True:
    
    user_input = input('You: ')
    # store human msgs
    chat_history.append(HumanMessage(content=user_input))

    if user_input == 'exit':
        break
    else:
        result = model.invoke(user_input)
        # store Ai Messages
        chat_history.append(AIMessage(content=result.content))

        print('\nBot: ', result.content)

print(chat_history)