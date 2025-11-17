import sys
import os
import torch
import time


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from docling.document_converter import DocumentConverter,PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions,AcceleratorDevice

from ingestion.chunker import chunk_doc
from ingestion.embedder import embed_and_store_doc
from llama_index.core.schema import Document

# Tentukan lokasi folder dokumen
DOCUMENTS_PATH = "./source_data"

print("Memuat: ingestion/ingest.py (Versi Final: GPU + OCR + Tabel)")

def load_data_docling(path):
    documents = []
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

    for filename in os.listdir(path):
        filepath = os.path.join(path,filename)

        abs_filepath = os.path.abspath(filepath)

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
                end_time = time.time()
                print(f"Proses selesai dalam waktu {start_time - end_time:.2f} detik")
            except Exception as e:
                print(f"Tidak bisa memproses file txt karena : {e}")
    return documents

def main():
    print("Memulai proses Ingestion dengan Docling...")
    
    document= load_data_docling(DOCUMENTS_PATH)

    if not document:
        print("Dokumen tidak ditemukan")
        return None
    
    print(f"\n Total dokumen yang akan di proses {len(document)}")

    nodes = chunk_doc(document)
    embed_and_store_doc(nodes)

    print("="*10 + "Selesai" + "="*10)

if __name__ == "__main__":
    main()