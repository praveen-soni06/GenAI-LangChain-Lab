from langchain_community.document_loaders import WebBaseLoader

url = r'https://www.flipkart.com/new-elec-clp-march-at-store?pageUID=1777557668185'

loader = WebBaseLoader(url)

docs = loader.load()

print(docs[0].page_content)