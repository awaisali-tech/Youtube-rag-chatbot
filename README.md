# YouTube RAG Chatbot

Built this while learning LangChain from the CampusX playlist. The idea came from wanting to actually use what I was learning, so I built a chatbot that lets you chat with any YouTube video.

You paste a URL, it fetches the transcript, and you can ask anything about the video. It only answers from the video content — if something isn't mentioned in the video, it tells you that instead of making something up.

## How it works

1. Paste any YouTube URL
2. App fetches the transcript using youtube-transcript-api
3. Transcript gets split into chunks using LangChain's RecursiveCharacterTextSplitter
4. Chunks are embedded using HuggingFace (runs locally, no API needed)
5. Embeddings are stored in FAISS vector store
6. When you ask a question, the 4 most similar chunks are retrieved
7. Groq's LLaMA 3.1 generates an answer based only on those chunks

## Tech Stack

- **LangChain** — RAG pipeline and LCEL chains
- **Groq + LLaMA 3.1 8B Instant** — LLM (free and fast)
- **HuggingFace Embeddings** — all-MiniLM-L6-v2
- **FAISS** — vector similarity search
- **youtube-transcript-api** — transcript fetching
- **Streamlit** — chat UI

## Setup

**Clone the repo**
```bash
git clone https://github.com/awaisali-tech/Youtube-rag-chatbot.git
cd Youtube-rag-chatbot
```

**Create virtual environment with Python 3.11**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

**Install dependencies**
```bash
pip install -r requirements.txt
```

**Create a `.env` file in the project root**



GROQ_API_KEY=your_groq_api_key_here
Get a free API key at [console.groq.com](https://console.groq.com)

**Run the app**
```bash
streamlit run App.py
```

## Things to keep in mind

- The video must have captions enabled
- First load takes around 30 seconds to process the transcript — it gets cached after that
- Works with any YouTube URL format (youtube.com, youtu.be, embed links)

## Author

**Awais Ali**  
Final year CS student at UMT Lahore  
[GitHub](https://github.com/awaisali-tech) | [LinkedIn](https://linkedin.com/in/awais-ali-1a6616300)
