import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import os
import re
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from dotenv import load_dotenv

load_dotenv()


# ---- Helper: Extract video ID from any YouTube URL ----
def extract_video_id(url):
    pattern = r"(?:v=|youtu\.be/|embed/)([a-zA-Z0-9_-]{11})"
    match = re.search(pattern, url)
    return match.group(1) if match else None


# ---- Helper: format docs ----
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ---- Build RAG chain from video ID ----
@st.cache_resource(show_spinner="Processing video...")
def build_chain(video_id):

    # Transcript
    yyt_api = YouTubeTranscriptApi()
    fetched = yyt_api.fetch(video_id)
    transcript = " ".join(chunk.text for chunk in fetched)

    # Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.create_documents([transcript])

    # Embeddings + Vector Store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={'k': 4})

    # LLM
    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))

    # Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("human", """Answer the question based only on the context below.
        If the answer is not in the context, say 'I could not find this in the video.'

Context: {context}

Question: {question}
""")
    ])

    # Chain
    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })

    chain = parallel_chain | prompt | llm | StrOutputParser()
    return chain


# ---- Streamlit UI ----
st.title("🎥 YouTube RAG Chatbot")
st.caption("Ask anything about a YouTube video")

url = st.text_input("Enter YouTube URL", placeholder="https://www.youtube.com/watch?v=...")

if url:
    video_id = extract_video_id(url)

    if not video_id:
        st.error("Invalid YouTube URL. Please check and try again.")
    else:
        try:
            chain = build_chain(video_id)

            # Chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # Display previous messages
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            # Input
            if question := st.chat_input("Ask something about the video..."):
                st.session_state.messages.append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.markdown(question)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        answer = chain.invoke(question)
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

        except TranscriptsDisabled:
            st.error("This video does not have captions/transcripts enabled.")
        except Exception as e:
            st.error(f"Error: {e}")