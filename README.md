Chatbot UPBC: Technical Documentation
A High-Performance RAG-Based Technical Query Engine

Chatbot UPBC is an advanced Retrieval-Augmented Generation (RAG) assistant designed for the Universidad Politécnica de Baja California. It leverages Google's Gemini Flash 1.5 and ChromaDB to provide precise, context-aware answers based on institutional PDF manuals and general engineering knowledge.

🛠️ System Architecture & "What it Does"
The system operates as a hybrid knowledge assistant. It doesn't just "chat"; it performs semantic searches across a local vector database to provide cited, verifiable answers.

Core Features:
Contextual Retrieval: Indexes local PDFs and retrieves relevant fragments to ground AI responses.

Dual Mode: Toggle between "Study Mode" (PDF-based) and "General Chat."

Real-time Observability: A built-in dashboard tracking latency, query count, and document coverage.

Feedback Loop: Integrated sentiment tracking (👍/👎) to monitor user satisfaction.

⚙️ How It Works (Technical Deep Dive)
This is an AI-powered Academic Assistant designed for students of the **"Programación Estructurada"** course at UPBC. It helps students instantly find information about:
* **Technical concepts:** Explanations of algorithms, syntax (C/Python), and data structures.
* **Source Verification:** Every answer includes citations from the official textbooks and lecture notes.

The application follows a standard RAG pipeline built with LangChain:Data 1. a) Ingestion: PDFs are loaded via PyPDFLoader and broken into chunks of 1500 characters with a 150-character overlap using RecursiveCharacterTextSplitter.

b) Vectorization: Text chunks are transformed into high-dimensional vectors using the all-MiniLM-L6-v2 HuggingFace model.
c) Storage: Vectors are stored in a local ChromaDB instance for persistent similarity searching.
d) Retrieval & Synthesis: When a user asks a question, the system performs a similarity search ($k=2$). If a match is found ($score < 1.0$), the text is injected into a system prompt.
e) LLM Execution: The prompt is processed by gemini-flash-latest with a mirroring instruction to ensure the response language matches the user's input.

## Setup Instructions

### Requirements
* Python 3.9 or higher
* A Google AI (Gemini) API Key (configured in `.env`)
* Course PDF documents located in the `docs/` directory.

### Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/tu-usuario/course-assistant.git](https://github.com/tu-usuario/course-assistant.git)
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the App
To start the web interface:
```bash
streamlit run streamlit_app.py

📦 requirements.txt
The following dependencies are required to run the environment effectively:

streamlit==1.32.0
langchain==0.1.12
langchain-google-genai==1.0.1
langchain-community==0.0.28
langchain-openai
langchain-huggingface
langchain-chroma
langchain-text-splitters
chromadb==0.4.24
sentence-transformers==2.6.0
pypdf==4.1.0
plotly
pandas==2.2.1
tenacity==8.2.3
google-generativeai==0.4.1
pysqlite3-binary
python-dotenv

🧪 Test Results & Metrics
During the development phase, the following performance benchmarks were observed:

Mean Latency: ~2.5 - 4.2 seconds per query (Model: Gemini Flash).

Retrieval Accuracy: High (>90%) for documents with clear headings; moderate for complex tables.


Resilience: Successfully handled ResourceExhausted (429) errors using exponential backoff via the tenacity library.
