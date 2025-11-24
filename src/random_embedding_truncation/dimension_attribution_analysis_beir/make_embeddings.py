import torch
import json, pathlib, sys, os
import faiss
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
import zlib
import gc

from random_embedding_truncation.truncator import Truncator

# Load embedding model (You can change to another model if needed)
# 임베딩 생성
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Which Device : {device}")
model_name = 'all-MiniLM-L6-v2'
st_model = SentenceTransformer(model_name, device=device)
is_e5 = "e5" in model_name.lower()

# Truncator 인스턴스 생성 (resize_scale=1.0으로 설정하면 truncation 없음)
truncator_model = Truncator(st_model, resize_scale=1.0, is_e5=is_e5)

def batch_encoding(emb_mem, off, nxt_off, corpus_batch):
    # Truncator의 encode_corpus 메서드 사용
    embeddings = truncator_model.encode_corpus(
        corpus_batch, 
        batch_size=128, 
        convert_to_numpy=True, 
        show_progress_bar=True, 
        num_workers=3
    )
    print(f"text len : {embeddings.shape[0]}, offset_size : {nxt_off - off}")

    if embeddings.shape[0] == (nxt_off - off):
        print("offset match")
        emb_mem[off:nxt_off] = embeddings
    else:
        print("offset mismatch")
        del emb_mem
        sys.exit(1)
    
def read_corpus(corpus_path, dataset_name):
    # Read JSON lines file and process
    # Truncator.encode_corpus가 사용할 수 있도록 list[dict] 형태로 반환
    corpus = []

    print("File Open Start")
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            corpus.append(data)

    print("File Read Complete")
    return corpus

def make_embeddings(dataset_name):
    # corpus file path
    corpus_filename = "corpus.jsonl"

    # 모델 이름을 파일 시스템에 안전한 형태로 변환 (예: "sentence-transformers/all-MiniLM-L6-v2" -> "sentence-transformers-all-MiniLM-L6-v2")
    model_name_safe = model_name.replace("/", "-")

    # dataset and embedding directory
    # cache_dir 구조에 맞춰서: {cache_dir}/{dataset_name}/ 형태로 저장
    dataset_dir = os.path.join("/media/dcceris/muvera_optimized/datasets", dataset_name)
    # cache_dir은 model_name을 포함한 경로 (예: /media/dcceris/muvera_optimized/embeddings/all-MiniLM-L6-v2)
    cache_dir = os.path.join("/media/dcceris/muvera_optimized/embeddings", model_name_safe)
    embedding_dir = os.path.join(cache_dir, dataset_name)
    os.makedirs(embedding_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)

    # corpus file path and embedding file path
    file_path = os.path.join(dataset_dir, corpus_filename)
    embedding_file = os.path.join(embedding_dir, f"{dataset_name}_{model_name_safe}_embeddings.dat")

    # Convert texts to embeddings
    print("start encode")
    if pathlib.Path(embedding_file).exists():
        print(f"✅ [INFO] 임베딩 파일이 존재합니다. 파일을 읽습니다: {embedding_file}")
        return embedding_file
    else:
        print(f"✅ [INFO] 임베딩 파일이 존재하지 않습니다. 새로 batch 생성합니다: {embedding_file}")

        # Read JSON lines file and process
        corpus = read_corpus(file_path, dataset_name)

        tot_doc = len(corpus)
        d = truncator_model.st.get_sentence_embedding_dimension()
        print(f"total doc: {tot_doc}, dimension: {d}")

        emb_mem = np.memmap(embedding_file, dtype="float32", mode="w+", shape=(tot_doc, d))
        batch_size = 20000
        offset = 0

        print(f"✅ [INFO] 임베딩 파일이 존재하지 않습니다. 새로 batch 생성합니다: {embedding_file}")
        while (offset < tot_doc):
            next_offset = min(offset + batch_size, tot_doc)
            print(f" offset : {offset}, offset + B : {next_offset - 1}, next_offset  : {next_offset}") 
            corpus_batch = corpus[offset:next_offset]
            batch_encoding(emb_mem, offset, next_offset, corpus_batch)
            offset = next_offset
        
        del emb_mem
        del corpus
        gc.collect()
        print("encode complete")
        print(f"✅ [INFO] 임베딩 파일이 생성되었습니다: {embedding_file}")

        return embedding_file

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("❌ 사용법: [arg0]python3 [arg1]dataset name [arg2] cluster size [arg3] make_index.py")
        sys.exit(1)

    dataset_name = sys.argv[1]



    try:
        # Make IVF index
        make_embeddings(dataset_name)

        gc.collect()

    except Exception as e:
        print(f"🚨 오류 발생: {e}")
        sys.exit(1)