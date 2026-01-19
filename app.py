import streamlit as st
import os
import hashlib
import chromadb
import docx  # para word xd
import google.generativeai as genai

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================
st.set_page_config(page_title="Chat PDF/Word con Gemini")

# Carga variables de entorno
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Modelo de embeddings local
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# Se Inicializa el Cliente de ChromaDB
client = chromadb.Client()

# ============================================================
# SESSION STATE
# ============================================================
if "collection" not in st.session_state:
    st.session_state.collection = None

if "file_processed" not in st.session_state:
    st.session_state.file_processed = False

if "file_hash" not in st.session_state:
    st.session_state.file_hash = None

if "messages" not in st.session_state:
    st.session_state.messages = []  # Para guardar el historial del chat

# ============================================================
# FUNCIONES
# ============================================================
def hash_file(file) -> str:
    return hashlib.sha256(file.getvalue()).hexdigest()

def extract_text_from_pdf(pdf_file):
    """Extrae texto de un PDF digital."""
    reader = PdfReader(pdf_file)
    text = ""
    for i, page in enumerate(reader.pages):
        content = page.extract_text()
        if content:
            text += f"\n[Página {i+1}]\n{content}"
    return text

def extract_text_from_docx(docx_file):
    """Extrae texto de un archivo Word (.docx)."""
    doc = docx.Document(docx_file)
    all_text = []
    for para in doc.paragraphs:
        all_text.append(para.text)
    return "\n".join(all_text)

def chunk_text(text):
    """Divide un texto largo en fragmentos (chunks)."""
    chunk_size = 500 
    overlap = 100
    chunks = []
    start = 0
    chunk_id = 0

    while start < len(text):
        chunk_text = text[start:start + chunk_size]
        chunks.append({
            "id": f"chunk_{chunk_id}",
            "content": chunk_text,
            "start_index": start,
            "size": len(chunk_text)
        })
        chunk_id += 1
        start += chunk_size - overlap
    return chunks

def create_chroma_collection(chunks):
    """Crea una colección nueva en ChromaDB."""
    try:
        client.delete_collection("doc_rag") # Cambié nombre para evitar conflictos
    except:
        pass

    collection = client.create_collection(name="doc_rag")
    texts = [c["content"] for c in chunks]
    embeddings = EMBEDDING_MODEL.encode(texts)

    collection.add(
        documents=texts,
        embeddings=embeddings.tolist(),
        ids=[c["id"] for c in chunks],
        metadatas=[
            {
                "chunk_index": i,
                "start_index": c["start_index"],
                "chunk_size": c["size"]
            }
            for i, c in enumerate(chunks)
        ]
    )
    return collection

def retrieve_context(collection, query, k=4):
    """Recupera los k chunks más similares."""
    query_embedding = EMBEDDING_MODEL.encode([query])
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=k
    )
    return results

def ask_gemini(context, question):
    """Llama a Gemini usando el contexto."""
    model = genai.GenerativeModel("models/gemini-2.5-flash-lite")
    prompt = f"""
Eres un asistente que responde SOLO con la información del contexto.
Si la respuesta no está en el contexto, di: "No se encuentra en el documento".

Contexto:
{context}

Pregunta:
{question}
"""
    response = model.generate_content(prompt)
    return response.text

# ============================================================
# INTERFAZ
# ============================================================

st.title("📄 Chat con Documentos (PDF/Word)")

# AHORA ACEPTA PDF Y DOCX
uploaded_file = st.file_uploader("Sube un archivo", type=["pdf", "docx"])

# 🔄 Detectar cambio de archivo y resetear estado
if uploaded_file:
    current_hash = hash_file(uploaded_file)

    if st.session_state.file_hash != current_hash:
        st.session_state.file_hash = current_hash
        st.session_state.file_processed = False
        st.session_state.collection = None
        st.session_state.messages = [] # Limpiar chat al cambiar archivo

# ------------------------------
# BOTÓN PROCESAR
# ------------------------------
if uploaded_file and not st.session_state.file_processed:
    if st.button("📥 Procesar Documento"):
        with st.spinner("Procesando documento..."):
            
            # DETECTAR TIPO DE ARCHIVO
            if uploaded_file.name.endswith('.pdf'):
                text = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.name.endswith('.docx'):
                text = extract_text_from_docx(uploaded_file)
            else:
                text = ""
                st.error("Formato no soportado")

            if text:
                chunks = chunk_text(text)
                st.session_state.collection = create_chroma_collection(chunks)
                st.session_state.file_processed = True
                st.success(f"Documento procesado ✅ ({len(chunks)} fragmentos)")

# ------------------------------
# SECCIÓN DE PREGUNTAS
# ------------------------------
if st.session_state.file_processed and st.session_state.collection:
    st.divider()
    
    # Mostrar historial de chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Input del usuario (Estilo chat moderno)
    if prompt := st.chat_input("Escribe tu pregunta sobre el documento..."):
        # 1. Mostrar pregunta usuario
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # 2. Procesar respuesta
        with st.spinner("Pensando..."):
            results = retrieve_context(st.session_state.collection, prompt)
            context_text = "\n\n".join(results["documents"][0])
            answer = ask_gemini(context_text, prompt)

        # 3. Mostrar respuesta bot
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.write(answer)
        
        # 4. Mostrar contexto (opcional)
        with st.expander("📚 Ver contexto usado"):
            for doc in results["documents"][0]:
                st.text(doc[:200] + "...")

# ==========================================
# PLUS: Barra lateral con herramientas
# ==========================================
with st.sidebar:
    st.markdown("---")
    st.header("🛠️ Herramientas Extra")
    
    # Botón 1: Descargar conversación
    if "messages" in st.session_state and st.session_state.messages:
        chat_str = ""
        for msg in st.session_state.messages:
            role = "🤖 Bot" if msg["role"] == "assistant" else "👤 Usuario"
            chat_str += f"{role}: {msg['content']}\n\n"
            
        st.download_button(
            label="💾 Descargar Chat (.txt)",
            data=chat_str,
            file_name="historial_chat.txt",
            mime="text/plain"
        )

    # Botón 2: Limpiar memoria (Reset)
    if st.button("🗑️ Borrar Historial"):
        st.session_state.messages = []
        st.rerun()
    
