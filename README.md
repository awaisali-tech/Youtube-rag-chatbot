<div align="center">

# 🎥 YouTube RAG Chatbot

### Chat with YouTube videos using Retrieval-Augmented Generation

A lightweight, conversational **RAG application** that transforms YouTube transcripts into a searchable knowledge base and generates grounded answers with source-aware retrieval.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C)](https://www.langchain.com/)
[![Groq](https://img.shields.io/badge/Groq-GPT--OSS--20B-F55036)](https://groq.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-0467DF)](https://github.com/facebookresearch/faiss)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

### 🚀 [Try the Live Demo](https://youtube-rag-chatbot-y49r2gknnvmgmfvvqm5jao.streamlit.app)

</div>

---

## 📌 Overview

**YouTube RAG Chatbot** is a Retrieval-Augmented Generation application that lets users ask questions about the content of a YouTube video.

Paste a YouTube URL, wait for the transcript to be processed, and start chatting.

Unlike a normal LLM chatbot, the application retrieves relevant sections of the video's transcript before generating an answer. The model is instructed to answer from the retrieved evidence rather than relying on general knowledge.

The project focuses on keeping the RAG architecture **simple, fast, local-first, and lightweight** while improving retrieval quality and answer grounding.

---

## ✨ Features

- 🎬 **YouTube transcript extraction**
- 🧠 **Retrieval-Augmented Generation**
- 🔎 **Semantic search using FAISS**
- 🧩 **Token-aware transcript chunking**
- ⏱️ **Transcript timestamp preservation**
- 💬 **Conversation-aware follow-up questions**
- 🔄 **History-aware query rewriting**
- 🎯 **Lightweight retrieval reranking**
- 📚 **Evidence-based answer generation**
- 🔗 **Chunk/source citations**
- 🌍 **Transcript language fallback support**
- 🚫 **Hallucination-resistant prompting**
- ⚡ **Fast inference through Groq**
- 🖥️ **Interactive Streamlit chat interface**
- 💾 **Cached transcript/vector processing**

---

## 🧠 How It Works

```text
                    YouTube URL
                         │
                         ▼
              ┌─────────────────────┐
              │ Transcript Fetching │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Transcript Cleanup  │
              │ + Timestamp Retain  │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Token-aware Chunking│
              └──────────┬──────────┘
                         │
                         ▼
          ┌─────────────────────────────┐
          │ HuggingFace MiniLM Embedding│
          └──────────────┬──────────────┘
                         │
                         ▼
                  ┌─────────────┐
                  │    FAISS    │
                  │Vector Index │
                  └──────┬──────┘
                         │
                    User Question
                         │
                         ▼
          ┌─────────────────────────────┐
          │ Conversation-aware Query   │
          │        Rewriting           │
          └──────────────┬──────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Candidate Retrieval │
              │       Top 10        │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Lightweight Rerank  │
              │       Top 5         │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Grounded LLM Prompt │
              │ + Transcript Context│
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  Groq GPT-OSS-20B   │
              └──────────┬──────────┘
                         │
                         ▼
                Grounded Answer
                + Evidence Citations
```

---

## 🔍 RAG Pipeline

### 1. Transcript Retrieval

The app extracts the video ID from the supplied YouTube URL and retrieves its captions using `youtube-transcript-api`.

The transcript pipeline attempts to:

- Prefer English captions
- Use manually generated captions where available
- Fall back to auto-generated captions
- Handle alternative transcript languages more gracefully
- Preserve timestamps for retrieved evidence

---

### 2. Token-Aware Chunking

Instead of blindly splitting the transcript into small character windows, the transcript is grouped into larger context-preserving chunks.

The current strategy uses approximately:

```text
Target chunk size: ~180 tokens
Maximum chunk size: ~220 tokens
Overlap: ~40 tokens
```

Chunks try to respect caption and sentence boundaries while staying below the embedding model's token limit.

This helps prevent important sentences or ideas from being broken across unrelated chunks.

---

### 3. Embeddings

Each transcript chunk is converted into a dense vector using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The embedding model runs locally through HuggingFace / Sentence Transformers.

Embeddings are explicitly normalized before being stored.

---

### 4. Vector Search with FAISS

Transcript vectors are stored in an in-memory **FAISS** vector index.

For each question, the system initially retrieves a broader set of candidates:

```text
Top-K Candidates = 10
```

This improves recall compared with relying only on a very small number of retrieved chunks.

---

### 5. Lightweight Reranking

Retrieved candidates are reranked using a combination of:

- Semantic embedding similarity
- Exact lexical/keyword overlap

The highest-ranked chunks are then passed to the LLM:

```text
Final Context Chunks = 5
```

This provides a lightweight improvement in retrieval precision without adding a cross-encoder, external search engine, or additional database.

---

### 6. Conversational Retrieval

Follow-up questions often contain references such as:

> "Why did he say that?"

or

> "What happened after that?"

Embedding these questions directly can produce poor retrieval results.

The chatbot therefore uses recent conversation history to rewrite the user's question into a **standalone retrieval query** before searching FAISS.

Example:

```text
Previous question:
"What pricing strategy did the speaker recommend?"

Follow-up:
"Why did he recommend that?"

Resolved retrieval query:
"Why did the speaker recommend the pricing strategy discussed in the video?"
```

Conversation history helps understand the question, but previous AI answers are **not treated as transcript evidence**.

---

### 7. Grounded Answer Generation

Retrieved transcript chunks are passed to:

```text
openai/gpt-oss-20b
```

through the **Groq API**.

The model runs with:

```python
temperature = 0.0
```

to reduce response variability.

The system prompt instructs the model to:

- Answer only from retrieved transcript evidence
- Avoid unsupported assumptions
- Cite evidence chunks
- Abstain when the answer cannot be found

When sufficient evidence is unavailable, the chatbot responds:

> **"I could not find this in the video."**

---

## 🛡️ Hallucination Control

Several lightweight safeguards are used to improve answer grounding:

1. **Temperature 0.0**
2. **Transcript-only system prompt**
3. **Explicit evidence delimiters**
4. **Chunk-level citations**
5. **Citation validation**
6. **Conversation history separated from factual evidence**
7. **Fallback response when evidence is insufficient**

This does not guarantee that an LLM can never make a mistake, but it significantly reduces unsupported answers compared with unconstrained generation.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Frontend / UI | Streamlit |
| RAG Framework | LangChain / LCEL |
| LLM | GPT-OSS-20B via Groq |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Database | FAISS |
| Transcript Extraction | youtube-transcript-api |
| Tokenization | HuggingFace Transformers |
| Environment Variables | python-dotenv |
| Language | Python 3.11 |

---

## 📂 Project Structure

```text
Youtube-rag-chatbot/
│
├── App.py
│   └── Streamlit UI + complete RAG pipeline
│
├── requirements.txt
│   └── Python dependencies
│
├── README.md
│   └── Project documentation
│
├── .gitignore
│
└── .env
    └── GROQ_API_KEY (local only — never commit)
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/awaisali-tech/Youtube-rag-chatbot.git
cd Youtube-rag-chatbot
```

### 2. Create a Virtual Environment

Python **3.11** is recommended.

```bash
python3.11 -m venv .venv
```

Activate it:

#### macOS / Linux

```bash
source .venv/bin/activate
```

#### Windows

```bash
.venv\Scripts\activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Configure Groq API Key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

You can create a Groq API key from:

https://console.groq.com/

> ⚠️ Never commit `.env` or API credentials to GitHub.

---

### 5. Run the Application

```bash
streamlit run App.py
```

Streamlit will display a local URL, usually:

```text
http://localhost:8501
```

Open it in your browser and paste a YouTube video URL.

---

## 💬 Example Usage

Paste a video:

```text
https://www.youtube.com/watch?v=VIDEO_ID
```

Then ask:

```text
Give me an overview of this video.
```

```text
What are the main arguments made by the speaker?
```

```text
What did he say about artificial intelligence?
```

Then continue conversationally:

```text
Why did he believe that?
```

The chatbot uses recent conversation context to resolve the follow-up before retrieving new evidence.

---

## 🌐 Supported YouTube URLs

The application supports common YouTube URL formats including:

```text
youtube.com/watch?v=VIDEO_ID
youtu.be/VIDEO_ID
youtube.com/embed/VIDEO_ID
youtube.com/shorts/VIDEO_ID
youtube.com/live/VIDEO_ID
```

A raw 11-character YouTube video ID can also be processed.

---

## ☁️ Deployment

The application can be deployed directly using **Streamlit Community Cloud**.

### Streamlit Secrets

When deploying, configure your API key from:

```text
Manage App
→ Settings
→ Secrets
```

Add:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

Do not upload your `.env` file to GitHub.

### YouTube Cloud-IP Limitation

YouTube may occasionally block transcript requests originating from cloud-hosted IP addresses.

If this occurs, the app may display a transcript request/IP-blocked error even though the same video works locally.

This is a limitation of transcript access from some hosting environments rather than the FAISS/RAG pipeline itself.

---

## ⚡ Performance

The embedding model runs locally, so no embedding API is required.

After a video has been processed, the transcript/vector resources are cached by Streamlit, reducing repeated transcript fetching and embedding work while the cached application process remains active.

```text
YouTube Transcript
        ↓
Local Embeddings
        ↓
In-Memory FAISS
        ↓
Groq Generation
```

No external vector database or backend server is required.

---

## 🎯 Design Philosophy

This project deliberately avoids unnecessary infrastructure.

It does **not** require:

- ❌ Pinecone
- ❌ Chroma server
- ❌ PostgreSQL
- ❌ Graph databases
- ❌ Elasticsearch
- ❌ Microservices
- ❌ Separate backend API

Instead, it demonstrates how a strong RAG workflow can be built with:

```text
Streamlit + LangChain + MiniLM + FAISS + Groq
```

The goal is to improve **retrieval quality, grounding, conversational understanding, and usability** while keeping the architecture easy to understand and deploy.

---

## 🚧 Current Limitations

- A YouTube transcript must be available
- Transcript quality depends on the video's captions
- Auto-generated captions may contain transcription mistakes
- Some cloud IP addresses may be blocked by YouTube
- FAISS indexes are stored in memory rather than permanently persisted
- Retrieval quality can decrease for languages where the embedding model has weaker representation
- LLM responses should still be treated as AI-generated output

---

## 🔮 Possible Future Improvements

The project intentionally remains lightweight, but useful future improvements could include:

- [ ] Small RAG evaluation dataset
- [ ] Retrieval Recall@K measurement
- [ ] Answer faithfulness evaluation
- [ ] Better multilingual embedding support
- [ ] Clickable YouTube timestamp citations
- [ ] Streaming LLM responses
- [ ] User feedback buttons
- [ ] Retrieval debugging / score visualization
- [ ] Optional transcript caching

---

## 📊 What This Project Demonstrates

This project covers several practical RAG engineering concepts:

- Retrieval-Augmented Generation
- Semantic search
- Vector embeddings
- Vector similarity search
- Prompt engineering
- Hallucination mitigation
- Query rewriting
- Conversational RAG
- Retrieval reranking
- Context-window management
- Transcript preprocessing
- Source attribution
- LLM API integration
- Streamlit deployment

---

## 🤝 Contributing

Contributions, suggestions, and improvements are welcome.

If you find a bug or have an idea for improving retrieval quality, feel free to open an issue or submit a pull request.

---

## 👨‍💻 Author

### Awais Ali

Final-year Computer Science student at **University of Management and Technology (UMT), Lahore**.

Interested in:

- Artificial Intelligence
- Machine Learning
- Large Language Models
- Retrieval-Augmented Generation
- NLP
- Generative AI

[![GitHub](https://img.shields.io/badge/GitHub-awaisali--tech-181717?logo=github)](https://github.com/awaisali-tech)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Awais_Ali-0A66C2?logo=linkedin&logoColor=white)](https://linkedin.com/in/awais-ali-1a6616300)

---

<div align="center">

### ⭐ If you find this project useful, consider giving it a star!

Built with ❤️ using **LangChain, FAISS, HuggingFace, Groq & Streamlit**

</div>
