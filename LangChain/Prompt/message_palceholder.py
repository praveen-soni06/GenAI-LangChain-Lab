from model import model
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
import os

chat_template = ChatPromptTemplate([('system','you are a costumer support agent'),
                                    MessagesPlaceholder(variable_name='chat_history'),
                                    ('human', '{query}')])

chat_history = []

base_dir = os.path.dirname(__file__)
file_path = os.path.join(base_dir, 'chat_history.txt')

with open(file_path) as f:
    chat_history.extend(f.readlines())

print(chat_history)