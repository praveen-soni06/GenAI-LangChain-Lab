from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(path=r"C:\DataScience\GenAI\RAG\Document Loder\books",
                         glob='*.pdf',
                         loader_cls=PyPDFLoader
                         )

# docs = loader.load()

# docs = loader.lazy_load()

# for idx,content in enumerate(docs):
#     if idx == 345:
#         print(content.page_content)
#         break

docs = list(loader.lazy_load())
print(docs[345].page_content)