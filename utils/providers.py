import os
import ollama
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASED_URL = os.getenv("OLLAMA_BASED_URL", "http://localhost:11434")

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3:instruct"

print("Menginisialisasi koneksi ke Ollama...")
try : 
    client = ollama.Client(host = OLLAMA_BASED_URL)
    client.list()
    print("Berhasil Terhubung ke Ollama")
except Exception as e:
    print(f"Gagal Terhubung ke Ollama {e}")
    client = None