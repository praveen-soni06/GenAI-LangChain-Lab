from langchain_community.document_loaders import CSVLoader

loader = CSVLoader("C:\DataScience\GenAI\RAG\Document Loder\Social_Network_Ads.csv")

docs = loader.load()

print(docs[0])