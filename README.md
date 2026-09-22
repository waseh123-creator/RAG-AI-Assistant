# 🤖 RAG AI Assistant

A Retrieval-Augmented Generation (RAG) application built with **Streamlit, FAISS, Sentence Transformers, and Groq**.

The application allows users to upload documents and ask questions about their content. It retrieves the most relevant document chunks using a FAISS vector database and uses an open-source language model hosted through Groq to generate grounded answers.

## 🚀 Features

* 📄 Upload PDF, TXT, and Markdown documents
* ✂️ Automatic text chunking
* 🧠 Open-source sentence-transformer embeddings
* 🔎 Semantic document search using FAISS
* 🤖 AI-powered answers using Groq
* 💬 Interactive Streamlit chat interface
* 📚 Display retrieved document sources
* 🔐 Secure Groq API key configuration through Streamlit Secrets
* ☁️ Ready for deployment on Streamlit Community Cloud

## 🏗️ Architecture

```text
User
 │
 │ Upload Documents
 ▼
Text Extraction
 │
 ▼
Text Chunking
 │
 ▼
Sentence Transformers
 │
 │ Embeddings
 ▼
FAISS Vector Database
 │
 │ Similarity Search
 ▼
Relevant Document Chunks
 │
 ▼
Groq API
 │
 ▼
Open-Source LLM
 │
 ▼
AI Generated Answer
```

## 🛠️ Tech Stack

| Technology            | Purpose                             |
| --------------------- | ----------------------------------- |
| Python                | Application development             |
| Streamlit             | Frontend and deployment             |
| FAISS                 | Vector database / similarity search |
| Sentence Transformers | Text embeddings                     |
| Groq                  | LLM inference API                   |
| `openai/gpt-oss-120b` | Language model                      |
| PyPDF                 | PDF text extraction                 |

## 📁 Project Structure

```text
rag-ai-assistant/
│
├── app.py
├── requirements.txt
└── README.md
```

## ⚙️ Local Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/rag-ai-assistant.git
cd rag-ai-assistant
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure the Groq API key

Create an environment variable called:

```text
GROQ_API_KEY
```

On Windows PowerShell:

```powershell
$env:GROQ_API_KEY="YOUR_GROQ_API_KEY"
```

### 4. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 🔑 Groq API

This project uses the Groq API for fast inference with an open-source language model.

The application is configured to use:

```text
openai/gpt-oss-120b
```

You need a Groq API key to generate AI responses.

**Never commit your API key to GitHub.**

## 📚 How to Use

1. Open the Streamlit application.
2. Upload one or more PDF, TXT, or Markdown files.
3. Click **Process Documents**.
4. The application extracts and chunks the text.
5. Sentence Transformers converts the chunks into embeddings.
6. FAISS stores the embeddings for similarity search.
7. Ask a question about your uploaded documents.
8. The application retrieves the most relevant chunks.
9. Groq generates an answer using the retrieved context.
10. Retrieved source information is displayed below the answer.

## 🧠 RAG Workflow

The application follows a Retrieval-Augmented Generation workflow:

```text
Document
   ↓
Text Extraction
   ↓
Chunking
   ↓
Embedding Generation
   ↓
FAISS Index
   ↓
User Question
   ↓
Question Embedding
   ↓
Similarity Search
   ↓
Top Relevant Chunks
   ↓
Groq LLM
   ↓
Final Answer
```

This allows the language model to answer questions using information retrieved from the user's documents instead of relying only on its pretrained knowledge.

## ☁️ Streamlit Cloud Deployment

The application can be deployed directly from GitHub using Streamlit Community Cloud.

### Steps

1. Push the project to GitHub.
2. Open Streamlit Community Cloud.
3. Select **Create App**.
4. Select this GitHub repository.
5. Select the `main` branch.
6. Set the main file to:

```text
app.py
```

7. Open **Advanced Settings**.
8. Add the following secret:

```toml
GROQ_API_KEY = "YOUR_GROQ_API_KEY"
```

9. Deploy the application.

## 🔐 Security

Do not store API keys directly inside Python files.

Bad:

```python
GROQ_API_KEY = "gsk_xxxxxxxxx"
```

Good:

```python
os.getenv("GROQ_API_KEY")
```

or Streamlit Secrets:

```python
st.secrets["GROQ_API_KEY"]
```

## ⚠️ Current Limitations

* FAISS index is currently stored in memory.
* Uploaded documents are not permanently stored.
* Restarting the application clears the current knowledge base.
* PDF extraction works best with text-based PDFs; scanned PDFs may require OCR.
* The application currently supports PDF, TXT, and Markdown files.

## 🔮 Future Improvements

Possible future enhancements include:

* Persistent vector database
* Document management
* User authentication
* Chat history persistence
* Multiple embedding models
* More file formats
* OCR for scanned documents
* Hybrid search
* Reranking
* Multiple LLM providers
* Streaming AI responses
* Advanced RAG evaluation
* Multi-stage AI workflows
* Additional knowledge sources

## 📄 License

This project is open-source and can be extended for learning and development purposes.

---

Built with **Python + Streamlit + FAISS + Sentence Transformers + Groq**.
