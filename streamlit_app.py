import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except (ImportError, KeyError):
    # Esto evita que la app truene si ya está en un sistema con sqlite actualizado
    pass
import os
import streamlit as st
import shutil
import warnings
import time
import json
import pandas as pd
import html
import base64
import plotly.express as px
import plotly.graph_objects as go
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import google.api_core.exceptions

# --- 1. CONFIGURACIÓN ---
API_KEY_PREDEFINIDA = "" 

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
current_dir = os.getcwd()
os.environ['HF_HOME'] = os.path.join(current_dir, 'hf_cache')

warnings.filterwarnings("ignore")
st.set_page_config(
    page_title="Chatbot UPBC - Motor de Consulta Técnica (RAG)",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Directorios
DOCS_DIR = os.path.join(current_dir, "docs")
DB_DIR = os.path.join(current_dir, "chroma_db_hybrid")
METRICS_FILE = os.path.join(current_dir, "metrics_db.json")

if not os.path.exists(DOCS_DIR): os.makedirs(DOCS_DIR)

# --- 2. ESTILOS CSS ---
st.markdown("""
    <style>
    :root { --primary-color: #4A148C; --bg-color: #ffffff; }
    .stApp { background-color: var(--bg-color); }
    
    .block-container {
        padding-top: 2rem !important; 
        padding-bottom: 5rem !important;
    }

    .sticky-header {
        background-color: #FAFAFA;
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #E0E0E0;
        margin-bottom: 20px;
    }

    .citation-badge {
        display: inline-flex; align-items: center; background-color: #F3E5F5;
        border: 1px solid #AB47BC; color: #4A148C; padding: 4px 12px;
        border-radius: 20px; font-size: 0.85rem; font-weight: bold; margin: 0 8px 8px 0;
    }
    
    .general-badge {
        display: inline-flex; align-items: center; background-color: #FFF3E0;
        border: 1px solid #FF9800; color: #E65100; padding: 4px 12px;
        border-radius: 20px; font-size: 0.85rem; font-weight: bold; margin: 0 8px 8px 0;
    }

    .metric-container {
        background-color: #FAFAFA; border: 1px solid #E0E0E0; border-radius: 10px;
        padding: 12px; margin-bottom: 12px; text-align: center;
    }
    .metric-label { font-size: 0.75rem; color: #616161; text-transform: uppercase; font-weight: 700; }
    .metric-value { font-size: 1.5rem; color: #4A148C; font-weight: 800; margin-top: -5px;}
    
    .instruction-box {
        background-color: #E3F2FD; padding: 15px; border-radius: 10px;
        border-left: 5px solid #1565C0; margin-bottom: 15px; font-size: 0.9rem;
    }
    
    [data-testid="stSidebarUserContent"] {
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# --- 4. GESTIÓN DE DATOS ---
def load_metrics_db():
    default_db = {
        "global_stats": {
            "query_count": 0, "positive_feedback": 0, "total_feedback": 0,
            "answers_from_docs": 0, "answers_general": 0, "total_answers": 0,
            "api_errors": 0 
        },
        "history": {"latency": []},
        "feedback_counts": {"Positivo": 0, "Negativo": 0}
    }
    if not os.path.exists(METRICS_FILE): return default_db
    try:
        with open(METRICS_FILE, "r") as f: 
            data = json.load(f)
            if "feedback_counts" not in data: data["feedback_counts"] = {"Positivo": 0, "Negativo": 0}
            return data
    except: return default_db

def save_metrics_db(db):
    try: 
        with open(METRICS_FILE, "w") as f: 
            json.dump(db, f)
    except: 
        pass

if "db_metrics" not in st.session_state: st.session_state.db_metrics = load_metrics_db()
if "messages" not in st.session_state: st.session_state.messages = []
if "docs_uploaded_flag" not in st.session_state: st.session_state.docs_uploaded_flag = False

# --- 5. FUNCIONES CORE Y OPTIMIZACIÓN ---
@st.cache_resource
def load_models_safe(api_key):
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        if not api_key: return None, None
        llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", google_api_key=api_key, temperature=0.3)
        return embeddings, llm
    except: return None, None

@st.cache_resource
def get_chroma_db_safe(_embeddings):
    if os.path.exists(DB_DIR):
        try:
            return Chroma(persist_directory=DB_DIR, embedding_function=_embeddings)
        except: return None
    return None

@retry(
    retry=retry_if_exception_type(google.api_core.exceptions.ResourceExhausted),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def call_llm_with_retry(chain, inputs):
    return chain.invoke(inputs)

def create_vector_db(api_key):
    if not os.path.exists(DOCS_DIR): return False, "⚠️ Carpeta vacía."
    pdfs = [f for f in os.listdir(DOCS_DIR) if f.lower().endswith('.pdf')]
    if not pdfs: return False, "⚠️ No hay PDFs."
    
    embeddings, _ = load_models_safe(api_key)
    with st.spinner("🔄 Indexando documentos..."):
        try:
            documents = []
            for p in pdfs:
                loader = PyPDFLoader(os.path.join(DOCS_DIR, p))
                documents.extend(loader.load())
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
            chunks = text_splitter.split_documents(documents)
            
            if os.path.exists(DB_DIR): shutil.rmtree(DB_DIR, ignore_errors=True)
            Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=DB_DIR)
            
            get_chroma_db_safe.clear()
            st.session_state.docs_uploaded_flag = False
            return True, f"✅ Indexado: {len(chunks)} fragmentos."
        except Exception as e: return False, f"❌ Error: {e}"

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join(DOCS_DIR, uploaded_file.name), "wb") as f: f.write(uploaded_file.getbuffer())
        return True
    except: return False

def render_advanced_dashboard():
    stats = st.session_state.db_metrics["global_stats"]
    fb_counts = st.session_state.db_metrics.get("feedback_counts", {"Positivo": 0, "Negativo": 0})
    lat_history = st.session_state.db_metrics["history"]["latency"]
    
    st.title("📊 Tablero de Control y Métricas")
    st.markdown("Monitoreo en tiempo real del rendimiento del asistente.")
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Consultas Totales", stats.get('query_count', 0))
    with kpi2:
        total = stats.get('total_answers', 1)
        doc_rate = (stats.get('answers_from_docs', 0) / total * 100) if total > 0 else 0
        st.metric("Cobertura Documental", f"{doc_rate:.1f}%")
    with kpi3:
        pos = fb_counts.get('Positivo', 0)
        neg = fb_counts.get('Negativo', 0)
        total_fb = pos + neg
        sat = (pos / total_fb * 100) if total_fb > 0 else 0
        st.metric("Satisfacción", f"{sat:.0f}%", f"{total_fb} votos")
    with kpi4:
        avg_lat = sum(lat_history)/len(lat_history) if lat_history else 0
        st.metric("Latencia Promedio", f"{avg_lat:.2f} s")

    st.divider()

    g1, g2 = st.columns([1, 1])
    
    with g1:
        st.subheader("📈 Velocidad (Latencia)")
        if lat_history:
            y_data = lat_history[-50:]
            x_data = list(range(len(y_data)))
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines', fill='tozeroy', line=dict(color='#4A148C', width=3), name='Segundos'))
            fig_line.update_layout(xaxis_title="Últimas consultas", yaxis_title="Segundos", margin=dict(l=20, r=20, t=30, b=20), height=300)
            st.plotly_chart(fig_line, use_container_width=True)
        else: st.info("Sin datos.")
            
    with g2:
        st.subheader("💡 Satisfacción (Feedback)")
        labels = ['Positivo', 'Negativo']
        values = [fb_counts.get('Positivo', 0), fb_counts.get('Negativo', 0)]
        colors = ['#4A148C', '#FFD700'] 
        fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4, pull=[0, 0.1], marker=dict(colors=colors, line=dict(color='#000000', width=1)))])
        fig_pie.update_layout(margin=dict(l=20, r=20, t=0, b=20), height=300, showlegend=True)
        st.plotly_chart(fig_pie, use_container_width=True)

# --- 6. INTERFAZ ---
with st.sidebar:
    if os.path.exists("logo_upbc.png"): st.image("logo_upbc.png", width=180)
    else: st.image("https://upload.wikimedia.org/wikipedia/commons/3/37/Logo_UPBC.png", width=180)
    
    st.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h1 style="color: #4A148C; font-size: 1.5rem; margin:0;">Chatbot UPBC</h1>
            <p style="color: #666; font-size: 0.9rem;">Motor de Consulta Técnica (RAG)</p>
        </div>
        <hr style="margin: 10px 0; border-top: 2px solid #4A148C;">
    """, unsafe_allow_html=True)
    
    st.header("⚙️ Menú de Opciones")
    opcion = st.selectbox("Selecciona una vista:", ["💬 Chat Interactivo", "📂 Gestión de Documentos", "📊 Panel de Métricas"])
    
    st.markdown("---")
    
    if opcion != "📊 Panel de Métricas":
        if API_KEY_PREDEFINIDA:
            api_key = API_KEY_PREDEFINIDA
            st.success("✅ Licencia Activa")
        else:
            api_key = st.text_input("Google API Key", type="password")
        use_rag = st.checkbox("🔍 Modo Estudio (PDFs)", value=True)

    st.markdown("---")
    with st.expander("🚀 Futuras Mejoras", expanded=False):
        st.markdown("- 🧠 Memoria conversacional\n- 📝 Modo Examen\n- 🎴 Generación de Flashcards")

    st.markdown("---")
    if st.button("🗑️ Limpiar Historial", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if opcion == "📊 Panel de Métricas":
    render_advanced_dashboard()

elif opcion == "📂 Gestión de Documentos":
    st.title("📂 Gestión de Base de Conocimiento")
    uploaded = st.file_uploader("Subir archivos PDF", type="pdf", accept_multiple_files=True)
    if uploaded:
        for u in uploaded: save_uploaded_file(u)
        st.session_state.docs_uploaded_flag = True
        st.success("✅ Archivos subidos correctamente.")
        
    if st.button("🔄 Procesar/Actualizar Base de Datos Vectorial", type="primary"):
        if api_key:
            success, msg = create_vector_db(api_key)
            if success: st.success(msg)
            else: st.error(msg)
        else: st.warning("⚠️ Necesitas una API Key para procesar.")

    if os.path.exists(DOCS_DIR):
        files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".pdf")]
        if files:
            st.markdown("### 📄 Archivos en Memoria:")
            for f in files: st.text(f"• {f}")

elif opcion == "💬 Chat Interactivo":
    col_chat = st.container()

    with col_chat:
        st.markdown('<div class="sticky-header">', unsafe_allow_html=True)
        with st.expander("📘 Información del Tutor", expanded=False):
            st.markdown("- **Objetivo:** 🎓 Asistente RAG para ingeniería UPBC.\n- **Stack:** 🐍 Python, Streamlit, LangChain, Gemini, ChromaDB.\n- **Funcionalidad:** 📂 Indexación de manuales y reglamentos.\n- **Recuperación:** 🔍 Búsqueda semántica.")
        st.markdown('</div>', unsafe_allow_html=True)

        if not st.session_state.messages:
            st.markdown('<div class="instruction-box"><b>Bienvenido.</b><br>1. <b>Modo Estudio (Activado):</b> Busca en tus documentos.<br>2. <b>Modo Chat:</b> Conocimiento general.</div>', unsafe_allow_html=True)
            
        for m in st.session_state.messages:
            with st.chat_message(m["role"], avatar="🧑‍💻" if m["role"] == "user" else "🤖"):
                st.markdown(m["content"], unsafe_allow_html=True)

    user_input = st.chat_input("Escribe tu duda aquí...")

    if user_input:
        if not api_key: st.error("⚠️ Falta API Key (Configúrala en el menú lateral)")
        else:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with col_chat:
                with st.chat_message("user", avatar="🧑‍💻"): st.markdown(user_input)
                with st.chat_message("assistant", avatar="🤖"):
                    msg_placeholder = st.empty()
                    msg_placeholder.markdown("⏳ *Procesando...*")
                    start_time = time.time()
                    
                    try:
                        embeddings, llm = load_models_safe(api_key)
                        context_text = ""
                        valid_docs = []
                        is_general = True
                        
                        if use_rag and embeddings:
                            try:
                                db = get_chroma_db_safe(embeddings)
                                if db:
                                    results = db.similarity_search_with_score(user_input, k=2)
                                    for doc, score in results:
                                        if score < 1.0: 
                                            valid_docs.append(doc)
                                            is_general = False
                                    if valid_docs: context_text = "\n\n".join([d.page_content for d in valid_docs])
                            except: pass

                        # --- ÚNICO CAMBIO: INSTRUCCIÓN DE IDIOMA ESPEJO ---
                        template = """Eres un tutor experto de UPBC.
                        
                        INSTRUCCIÓN DE IDIOMA: Responde SIEMPRE en el mismo idioma en que se te haga la pregunta. 
                        Si preguntan en inglés, responde en inglés. Si preguntan en español, responde en español.
                        
                        CONTEXTO (Si aplica):
                        {context}
                        
                        PREGUNTA: {question}
                        
                        Responde de forma clara y concisa. Si vas a escribir código, usa bloques de formato Markdown (```python ... ```).
                        """
                        chain = ChatPromptTemplate.from_template(template) | llm | StrOutputParser()
                        response = call_llm_with_retry(chain, {"context": context_text, "question": user_input})
                        
                        final_html = response
                        citation_html = "<br><br><div style='margin-top:10px; border-top: 1px solid #eee; padding-top: 10px;'>"
                        
                        if not is_general and valid_docs:
                            citation_html += "<b>📚 Basado en Documentos:</b><br>"
                            for doc in valid_docs:
                                src = os.path.basename(doc.metadata.get("source", "Doc"))
                                page = doc.metadata.get("page", 0) + 1
                                citation_html += f"📄 {src} [P.{page}]<br>"
                            st.session_state.db_metrics["global_stats"]["answers_from_docs"] += 1
                        else:
                            st.session_state.db_metrics["global_stats"]["answers_general"] += 1
                            citation_html += f'<span style="background-color: #FFF3E0; color: #E65100; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: bold;">🌐 { "Chat Rápido" if not use_rag else "Conocimiento General" }</span>'
                        
                        citation_html += "</div>"
                        full_output = final_html + citation_html
                        msg_placeholder.markdown(full_output, unsafe_allow_html=True)
                        
                        end_time = time.time()
                        st.session_state.db_metrics["global_stats"]["query_count"] += 1
                        st.session_state.db_metrics["history"]["latency"].append(end_time - start_time)
                        st.session_state.db_metrics["global_stats"]["total_answers"] += 1 
                        save_metrics_db(st.session_state.db_metrics)
                        st.session_state.messages.append({"role": "assistant", "content": full_output})

                    except Exception as e:
                        st.session_state.db_metrics["global_stats"]["api_errors"] += 1
                        save_metrics_db(st.session_state.db_metrics)
                        if "429" in str(e) or "RESOURCE" in str(e): st.error("⚠️ Servidor saturado. Reintenta en 30s.")
                        else: st.error(f"Error: {e}")

    if st.session_state.messages and "assistant" in st.session_state.messages[-1]["role"]:
        with col_chat:
            st.markdown("---")
            c1, c2, _ = st.columns([1,1,10])
            with c1:
                if st.button("👍", key="like"):
                    st.session_state.db_metrics["global_stats"]["positive_feedback"] += 1
                    st.session_state.db_metrics["global_stats"]["total_feedback"] += 1
                    st.session_state.db_metrics["feedback_counts"]["Positivo"] += 1
                    save_metrics_db(st.session_state.db_metrics)
                    st.rerun()
            with c2:
                if st.button("👎", key="dislike"):
                    st.session_state.db_metrics["global_stats"]["total_feedback"] += 1 
                    st.session_state.db_metrics["feedback_counts"]["Negativo"] += 1
                    save_metrics_db(st.session_state.db_metrics)
                    st.rerun()
