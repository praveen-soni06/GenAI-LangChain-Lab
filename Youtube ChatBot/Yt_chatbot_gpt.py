import os
import streamlit as st
from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from youtube_transcript_api._errors import IpBlocked

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough

# -------------------- ENV --------------------
load_dotenv()

# -------------------- LLM --------------------
llm = HuggingFaceEndpoint(
    repo_id='Qwen/Qwen2.5-72B-Instruct',
    task='text-generation',
    max_new_tokens=512,
    temperature=0.3
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

# -------------------- HELPERS --------------------

def extract_video_id(url_or_id):
    """Extract ID from full URL or return ID if already given"""
    if "youtube.com" in url_or_id or "youtu.be" in url_or_id:
        parsed = urlparse(url_or_id)
        return parse_qs(parsed.query).get("v", [url_or_id])[0]
    return url_or_id


@st.cache_resource
def build_vector_store(transcript):
    """Create FAISS index (cached for performance)"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )

    return FAISS.from_documents(chunks, embeddings)


def get_transcript(video_id):
    """Fetch transcript safely"""
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.fetch(video_id, languages=['en'])
        transcript = " ".join(chunk.text for chunk in transcript_list)

        if len(transcript) < 50:
            raise Exception("Transcript too short")

        return transcript

    except IpBlocked:
        st.error("🚫 IP blocked by YouTube. Try mobile hotspot or wait.")
        return None

    except TranscriptsDisabled:
        st.error("❌ No captions available for this video.")
        return None

    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


# -------------------- UI --------------------

st.set_page_config(page_title="YouTube RAG Chatbot", layout="centered")
st.title("🤖 YouTube RAG Chatbot")

video_input = st.text_input("Enter YouTube URL or Video ID:")
question = st.text_input("Ask your question:")

# -------------------- MAIN --------------------

if st.button("Submit"):

    if not video_input or not question:
        st.warning("Please enter both Video ID/URL and Question.")
        st.stop()

    video_id = extract_video_id(video_input)

    # Step 1: Get transcript
    with st.spinner("📥 Fetching transcript..."):
        transcript = get_transcript(video_id)

    if not transcript:
        st.stop()

    # Step 2: Build vector store
    with st.spinner("📚 Processing transcript..."):
        vector_store = build_vector_store(transcript)

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # Step 3: Prompt
    prompt = PromptTemplate(
        template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, say "I don't know."

Context:
{context}

Question:
{question}

Answer:
""",
        input_variables=["context", "question"]
    )

    # Step 4: Formatter
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Step 5: RAG Chain
    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })

    rag_chain = parallel_chain | prompt | model | parser

    # Step 6: Generate answer
    with st.spinner("🤖 Thinking..."):
        answer = rag_chain.invoke(question)

    st.subheader("🤖 Answer:")
    st.write(answer)