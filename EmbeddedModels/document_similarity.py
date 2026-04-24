from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

documents = [
    "The capital of India is New Delhi, which is also the seat of the central government.",
    "Narendra Modi is the Prime Minister of India and leads the BJP political party.",
    "The Indian cricket team won the T20 World Cup in 2024 after a long wait.",
    "Machine learning is a subset of artificial intelligence that learns from data.",
    "Python is one of the most popular programming languages for data science and AI.",
    "The Amazon rainforest produces 20% of the world's oxygen and is home to millions of species.",
    "Virat Kohli is a famous Indian cricketer known for his aggressive batting style.",
    "Deep learning uses neural networks with many layers to solve complex problems.",
    "The Ganges is one of the most sacred rivers in India and flows through several states.",
    "Natural language processing helps computers understand and generate human language.",
]

doc_embed = embedding.embed_documents(documents)

query = input('Enter Your Query: ')

query_embed = embedding.embed_query(query)

score = cosine_similarity([query_embed], doc_embed)[0]

index, score = sorted(list(enumerate(score)), key= lambda x: x[1])[-1]

print('User: ',query)
print('\n Assistent: \n', documents[index])

print('\nSimilarty Score: ', score )