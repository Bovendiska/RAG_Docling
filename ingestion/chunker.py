from llama_index.core.node_parser import SentenceSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core.schema import Document

print("Inisialisasi Chunker...")

def chunk_doc(documents: list[Document]) -> list:
    """
    Proses dokumen menjadi potongan-potongan kecil menggunakan SentenceSplitter.
    """

    print("Memulai proses Chunking...")

    chunker = SentenceSplitter(
        chunk_size = 1500,
        chunk_overlap = 150
    )

    nodes = chunker.get_nodes_from_documents(documents)

    print(f"Proses chunking selesai, Dokumen dipecah menjadi {len(nodes)} potongan.")

    return nodes