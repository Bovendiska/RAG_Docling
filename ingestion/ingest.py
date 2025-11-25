import sys
import os
import torch
import time
import hashlib
import json


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from docling.document_converter import DocumentConverter,PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions,AcceleratorDevice

from ingestion.chunker import chunk_doc
from ingestion.embedder import embed_and_store_doc
from llama_index.core.schema import Document

# Tentukan lokasi folder dokumen
DOCUMENTS_PATH = "./source_data"
STATE_FILE = "./ingestion/ingested_state.json"

print("Memuat: ingestion/ingest.py (Versi Final: GPU + OCR + Tabel + Incremental)")

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Gagal memuat state: {e}, membuat state baru.")
            return {}
    return {}

def save_state(state):
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=4)
        print("💾 State berhasil disimpan.")
    except Exception as e:
        print(f"⚠️ Gagal menyimpan state: {e}")

def calculate_file_hash(path):
    hash_md5 = hashlib.md5()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"⚠️ Gagal menghitung hash untuk {path}: {e}")
        return None

def load_data_docling(path):
    documents = []
    state = load_state()
    new_state = state.copy()
    files_processed_count = 0

    print("Menyiapkan Docling....")
    print("Konfigurasi menggunakan GPU....")

    pipeline_option = PdfPipelineOptions()
    pipeline_option.do_ocr = True
    pipeline_option.do_table_structure = True

    pipeline_option.accelerator_options = AcceleratorOptions(
        num_threads= 9,
        device = AcceleratorDevice.CUDA
    )

    format_option = {
        InputFormat.PDF : PdfFormatOption(pipeline_options=pipeline_option)
    }

    try :
        converter = DocumentConverter(format_options=format_option)
        print("✅ Docling Converter berhasil kalibrasi pakai CUDA")
    except Exception as e:
        print(f"❌ Gagal menggunakan CUDA saat kalibrasi karena {e}")
        converter = DocumentConverter()

    print(f"Memulai membaca dokumen dari : {path}")

    if not os.path.exists(path):
        print(f"❌ Folder {path} tidak ditemukan!")
        return [], state # Return empty list and current state

    for filename in os.listdir(path):
        filepath = os.path.join(path,filename)
        abs_filepath = os.path.abspath(filepath)
        
        # Skip directories
        if not os.path.isfile(filepath):
            continue

        # Incremental Check
        current_hash = calculate_file_hash(filepath)
        if current_hash:
            if filename in state and state[filename] == current_hash:
                print(f"⏭️ {filename} tidak berubah, lewati proses.")
                continue
            else:
                print(f"🔄️ {filename} baru atau berubah, akan diproses!")
        
        # Processing Logic
        processed_successfully = False
        if filename.lower().endswith('.pdf'):
            print(f"Proses dokumen {filename} pakai Docling")
            start_time = time.time()

            try :
                result = converter.convert(abs_filepath)
                markdown_text = result.document.export_to_markdown()

                # Fast Debuging
                print(f"Berhasil memproses, panjang teks: {len(markdown_text)} karakter")
                print(f"Preview tabel : \n{markdown_text[:10]}...")

                doc_obj = Document(
                    text = markdown_text,
                    metadata = {"filename": filename}
                )
                documents.append(doc_obj)
                processed_successfully = True
                end_t = time.time()

                print(f"Proses Selesai dalam waktu {start_time - end_t : .2f} detik")

            except Exception as e:
                print(f"Gagal proses karena {e}")
        
        elif filename.lower().endswith(".txt"):
            print(f"Proses file berekstensi .txt {filename}")
            start_time = time.time()
            try :
                with open(filepath, 'r', encoding = 'utf-8') as f:
                    text = f.read()
                    documents.append(
                        Document(
                            text=text,
                            metadata = {"filename" : filename}
                        )
                    )
                processed_successfully = True
                end_time = time.time()
                print(f"Proses selesai dalam waktu {start_time - end_time:.2f} detik")
            except Exception as e:
                print(f"Tidak bisa memproses file txt karena : {e}")
        
        # Update state only if successfully processed
        if processed_successfully and current_hash:
            new_state[filename] = current_hash
            files_processed_count += 1

    return documents, new_state

def main():
    print("Memulai proses Ingestion dengan Docling...")
    
    documents, new_state = load_data_docling(DOCUMENTS_PATH)

    if not documents:
        print("Tidak ada dokumen baru untuk diproses.")
        # Even if no documents, we might want to save state if we want to handle deletions later, 
        # but for now, if nothing processed, maybe nothing to save unless we want to sync removals.
        # Let's just save to be safe if we eventually handle deletions.
        # But wait, if documents is empty, it means either no files or all skipped.
        # If all skipped, new_state is same as state (or updated if we handled deletions, which we didn't yet).
        # Let's save the state if we processed anything or if we want to ensure consistency.
        # Actually, if we just skipped everything, saving is harmless.
        save_state(new_state) 
        return None
    
    print(f"\n Total dokumen BARU yang akan di proses {len(documents)}")

    nodes = chunk_doc(documents)
    embed_and_store_doc(nodes)
    
    # Save state only after successful embedding
    save_state(new_state)

    print("="*10 + "Selesai" + "="*10)

if __name__ == "__main__":
    main()