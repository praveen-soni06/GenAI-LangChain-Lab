import os
import streamlit as st

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from youtube_transcript_api._errors import IpBlocked
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

# LLM
llm = HuggingFaceEndpoint(
    repo_id='Qwen/Qwen2.5-72B-Instruct',
    task='text-generation',
    max_new_tokens=512,
    temperature=0.3
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

# UI
st.header("🤖 YouTube RAG Chatbot")

video_id = st.text_input("Enter YouTube Video ID:")
question = st.text_input("Ask your question:")

if st.button("Submit"):

    if not video_id or not question:
        st.warning("Please enter both Video ID and Question.")
        st.stop()

    try:
        with st.spinner("Fetching transcript..."):
            try:
                ytt_api = YouTubeTranscriptApi()
                transcript_list = ytt_api.fetch(video_id, languages=['en'])

                transcript = " ".join(chunk.text for chunk in transcript_list)

            except IpBlocked:
                st.error("🚫 IP blocked by YouTube. Try using mobile hotspot or wait.")
                st.stop()

            except TranscriptsDisabled:
                st.error("No captions available for this video.")
                st.stop()

    except TranscriptsDisabled:
        st.error("No captions available for this video.")
        st.stop()

    # 🔹 Text Splitting
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    # 🔹 Embeddings + FAISS
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )
    vector_store = FAISS.from_documents(chunks, embeddings)

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # 🔹 Prompt
    prompt = PromptTemplate(
        template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, say "I don't know."

Context:
{context}

Question:
{question}
""",
        input_variables=["context", "question"]
    )

    # 🔹 Formatter
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 🔹 RAG Chain
    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })

    rag_chain = parallel_chain | prompt | model | parser

    # 🔹 Output
    with st.spinner("Thinking..."):
        answer = rag_chain.invoke(question)

    st.subheader("🤖 Answer:")
    st.write(answer)