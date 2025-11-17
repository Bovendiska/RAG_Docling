import torch
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.settings import Settings

from utils.db import DB_PATH, COLLECTION_NAME
from utils.providers import client as ollama_client, EMBED_MODEL

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

client = chromadb.PersistentClient(path=DB_PATH)
chroma_collection = client.get_or_create_collection(name = COLLECTION_NAME)

print("Memulai proses embedding dokumen...")

def embed_and_store_doc(nodes:list):

    vector_store = ChromaVectorStore(chroma_collection = chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store= vector_store)

    embed_model = OllamaEmbedding(
        model_name = EMBED_MODEL,
        base_url = "http://localhost:11434",
        ollama_additional_kwargs = {"device":DEVICE}
    )

    Settings.embed_model = embed_model
    Settings.chunk_size = 1500
    try:

        index= VectorStoreIndex(
            nodes= nodes,
            storage_context = storage_context,
            show_progress= True
           )

    except Exception as e:
        print(f"Gagal melakukan embedding untuk node: {e}")
    print("Proses embedding selesai dan data disimpan ke database Chroma.")
    print(f"Total {chroma_collection.count()} item dalam koleksi Chroma.")
    print(f"Total dokumen yang di-embed dan disimpan: {len(nodes)}")