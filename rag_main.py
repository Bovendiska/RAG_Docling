import streamlit as st
import os
import sys
import torch
import chromadb
import time

DEVICE = 'cuda' if torch.cuda.is_available() else "cpu"

# Menambahkan path root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="RAG Docling", layout="wide")

@st.cache_resource
def load_reranker():
    from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
    try:
        reranker = FlagEmbeddingReranker(
            top_n = 5,
            model = "BAAI/bge-reranker-base"
        )
        print("✅ Reranker berhasil di muat")
        return reranker
    except Exception as e:
        st.error(f"Gagal memuat reranker {e}")
        st.stop()


def load_rag_pipeline(reranker):
        
    try:
        from utils.db import DB_PATH, COLLECTION_NAME
        from utils.providers import client as ollama_client, CHAT_MODEL, EMBED_MODEL
        from llama_index.core import (
            VectorStoreIndex,
            StorageContext,
            Settings
        )
        try :
            db_client = chromadb.PersistentClient(path = DB_PATH)
            chroma_collection = db_client.get_or_create_collection(name = COLLECTION_NAME)
        except Exception as e:
            st.error(f"Gagal koneksi ke Chroma {e}")
            st.stop()
        from llama_index.vector_stores.chroma import ChromaVectorStore
        from llama_index.embeddings.ollama import OllamaEmbedding
        from llama_index.llms.ollama import Ollama
    except ImportError as e:
        st.error(f"Gagal mengimpor modul yang diperlukan: {e}")
        st.stop()
    vector_store = ChromaVectorStore(chroma_collection = chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    embed_model = OllamaEmbedding(
        model_name = EMBED_MODEL,
        base_url = "http://localhost:11434",
        ollama_additional_kwargs = {"device" : DEVICE}
    )
    Settings.embed_model = embed_model

    llm = Ollama(
        model = CHAT_MODEL,
        base_url = "http://localhost:11434",
        request_timeout = 120.0,
        temperature = 0.0
    )

    Settings.llm = llm

    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context = storage_context
    )


    query_engine = index.as_query_engine(
        similarity_top_k = 25,
        node_postprocessors = [reranker],
        llm = llm
    )

    query_engine.update_prompts(
        {"text_qa_template" : respon_rag()}
    )
    return query_engine

def respon_rag():
    from llama_index.core import PromptTemplate

    template = (
        "PERAN: Anda adalah Asisten AI FAKTUAL yang menjawab dalam Bahasa Indonesia.\n"
        "ATURAN MUTLAK:\n"
        "1. Jawab HANYA berdasarkan \"KONTEKS DOKUMEN\" di bawah. DILARANG KERAS memakai pengetahuan umum/hafalan.\n"
        "2. Gunakan istilah spesifik dari dokumen. Jika Konteks berbahasa Inggris, TERJEMAHKAN ke Bahasa Indonesia.\n"
        "3. Jika user meminta \"Jelaskan\" dan Konteks berisi PARAGRAF: Uraikan jawaban Anda.\n"
        "4. Jika user meminta \"Jelaskan\" dan Konteks HANYA berisi POIN-POIN (LIST): Katakan: \"Berdasarkan dokumen, tahapan tersebut mencakup poin-poin berikut:\" lalu SEBUTKAN POIN-POIN tersebut.\n"
        "5. Jika informasi benar-benar TIDAK ADA di Konteks, katakan: \"Maaf, informasi spesifik ini tidak ditemukan di dalam dokumen.\"\n\n"
        "KONTEKS DOKUMEN:\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "PERTANYAAN USER: {query_str}\n"
        "JAWABAN (dalam Bahasa Indonesia):"
    )
    return PromptTemplate(template)

st.title("🤖 Chatbot RAG")

reranker = load_reranker()

try:
    query_engine = load_rag_pipeline(reranker)
except Exception as e:
    st.error(f"Gagal memuat RAG pipeline : {e}")
    st.stop()
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Halo! Saya siap menjawab pertanyaan tentang dokumen Anda."}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ketik pertanyaan Anda..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    prompt_bahasa = f"""
    Pertanyaan user : {prompt}

    1. WAJIB MENJAWAB DENGAN BAHASA INDONESIA
    2. Jika konteks dokumen dalam bahasa inggris, ANDA WAJIB UNTUK MENERJEMAHKAN KE BAHASA INDONESIA
    3. Jawab hanya berdasarkan Konteks, JANGAN MENGARANG
    """
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("Mencari..."):
            
            try :
                time_s = time.time()
                response = query_engine.query(prompt_bahasa)

                final_response = str(response)
                message_placeholder.markdown(final_response)

                with st.expander("🔍 Debug : 5 konteks teratas..."):
                    context_nodes = response.source_nodes

                    for i, node in enumerate(context_nodes):
                        st.markdown(f"Node {i+1} (skor {node.score:.4f}) - File {node.metadata.get('filename','N/A')}")
                        st.text(node.text[:500] + ".....")
                time_e = time.time()
                st.info(f"Waktu response {time_e - time_s}")
            except Exception as e:
                final_response = f"Terjadi Kesalahan {e}"
                message_placeholder.markdown(final_response)
            
            st.session_state.messages.append({
                "role" : "assistant",
                "content" : final_response
            })
