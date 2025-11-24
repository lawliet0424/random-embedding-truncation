from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
import os
import json
import gc

import numpy as np
import sienna
from sentence_transformers.evaluation.NanoBEIREvaluator import NanoBEIREvaluator
from sentence_transformers.SentenceTransformer import SentenceTransformer
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.util import cos_sim, dot_score

from random_embedding_truncation.truncator import Truncator, task_name_to_instruct
from random_embedding_truncation.utils import read_toml

from random_embedding_truncation.dimension_attribution_analysis_beir.make_embeddings import make_embeddings

DATASET_NAMES = [
    #"climatefever",
    #"dbpedia",
    #"fever",
    #"fiqa2018",
    #"hotpotqa",
    #"msmarco",
    #"nfcorpus",
    #"nq",
    #"quoraretrieval",
    "scidocs",
    "arguana",
    #"scifact",
    #"touche2020",
]


@dataclass
class Config:
    model_name: str
    cache_dir: Path
    output_path: Path
    use_dot_product: bool
    is_e5_mistral: bool
    embedding_file_pattern: str | None = None

    @classmethod
    def from_config(cls) -> "Config":
        parser = ArgumentParser()
        parser.add_argument("--config", type=str)
        args = parser.parse_args()
        _config = read_toml(args.config)
        return cls(
            _config["model_name"],
            Path(_config["cache_dir"]),
            Path(_config["output_path"]),
            use_dot_product=_config.get("use_dot_product", False),
            is_e5_mistral="e5-mistral" in _config["model_name"],
            embedding_file_pattern=_config.get("embedding_file_pattern", None),
        )

##FDE 생성 관련 meta.json 파일 이용해서 실험하고자 하는 FDE에 대해 실험 수행
def performance_test_FDE(dim_size, precomputed_embeddings_dict, encoder, config.cache_dir, config.is_e5_mistral, config.output_path):
    for dim_to_drop in range(dim_size):
        dims_to_keep = list(range(dim_size))
        del dims_to_keep[dim_to_drop]
        
        # 각 dataset에 대해 별도의 Truncator 생성 (각 dataset마다 다른 임베딩 파일 사용)
        for dataset_name in DATASET_NAMES:
            precomputed_emb = precomputed_embeddings_dict.get(dataset_name)
            
            model = Truncator(
                encoder,
                resize_scale=1.0,
                cache_dir=config.cache_dir,
                indexes_to_keep=dims_to_keep,
                is_e5=config.is_e5_mistral,
                precomputed_embeddings=precomputed_emb,
            )
            # 각 dataset에 대해 인덱스 리셋 (순서 기반 매칭을 위해)
            model._corpus_start_index = 0
            
            evaluator = NanoBEIREvaluator(
                dataset_names=[dataset_name],
                score_functions={SimilarityFunction.COSINE.value: cos_sim}
                if not config.use_dot_product
                else {SimilarityFunction.DOT_PRODUCT.value: dot_score},
                query_prompts=task_name_to_instruct if config.is_e5_mistral else None,
                batch_size=8 if config.is_e5_mistral else 32,
            )
            result = evaluator(model)
            
            # 결과 저장
            key = f"{dataset_name}_dim_{dim_to_drop}"
            if key not in results:
                results[key] = {}
            results[key] = result
            sienna.save(results, config.output_path)
            print(f"✅ [INFO] {dataset_name}, dim_to_drop={dim_to_drop} 완료")

    print(f"✅ [INFO] 모든 결과 저장 완료: {config.output_path}")

##FDE 생성할 때, 생성 관련 meta.json 파일 이용
def performance_test_partition(dim_size, precomputed_embeddings_dict, encoder, config.cache_dir, config.is_e5_mistral, config.output_path):
    for dim_to_drop in range(dim_size):
        dims_to_keep = list(range(dim_size))
        del dims_to_keep[dim_to_drop]
        
        # 각 dataset에 대해 별도의 Truncator 생성 (각 dataset마다 다른 임베딩 파일 사용)
        for dataset_name in DATASET_NAMES:
            precomputed_emb = precomputed_embeddings_dict.get(dataset_name)
            
            model = Truncator(
                encoder,
                resize_scale=1.0,
                cache_dir=config.cache_dir,
                indexes_to_keep=dims_to_keep,
                is_e5=config.is_e5_mistral,
                precomputed_embeddings=precomputed_emb,
            )
            # 각 dataset에 대해 인덱스 리셋 (순서 기반 매칭을 위해)
            model._corpus_start_index = 0
            
            evaluator = NanoBEIREvaluator(
                dataset_names=[dataset_name],
                score_functions={SimilarityFunction.COSINE.value: cos_sim}
                if not config.use_dot_product
                else {SimilarityFunction.DOT_PRODUCT.value: dot_score},
                query_prompts=task_name_to_instruct if config.is_e5_mistral else None,
                batch_size=8 if config.is_e5_mistral else 32,
            )
            result = evaluator(model)
            
            # 결과 저장
            key = f"{dataset_name}_dim_{dim_to_drop}"
            if key not in results:
                results[key] = {}
            results[key] = result
            sienna.save(results, config.output_path)
            print(f"✅ [INFO] {dataset_name}, dim_to_drop={dim_to_drop} 완료")

if __name__ == "__main__":
    ##FDE 생성 관련 meta.json 파일 이용해서 실험하고자 하는 FDE에 대해 실험 수행
    method = args.ArgumentParser()

    config = Config.from_config()

    if not config.output_path.parent.exists():
        config.output_path.parent.mkdir(parents=True)

    encoder = SentenceTransformer(config.model_name, trust_remote_code=True)

    dim_size = encoder.get_sentence_embedding_dimension()
    assert isinstance(dim_size, int)

    if config.output_path.exists():
        results = sienna.load(config.output_path)
    else:
        results = {}

    # 각 dataset에 대한 precomputed embeddings 로드
    precomputed_embeddings_dict = {}
    # 모델 이름을 파일 시스템에 안전한 형태로 변환
    model_name_safe = config.model_name.replace("/", "_")

    # FDE가 존재하는지 확인 
    FDE_file_path = os.path.join("/media/dcceris/muvera_optimized/cache_muvera/scidocs/main_weight/scidocs__raphaelsty_neural-cherche-colbert__d128_r1_p1_seed42_fill1__555ea3b587")
    
    # 임베딩 파일 경로 패턴 설정 (cache_dir 사용)
    if config.embedding_file_pattern:
        embedding_file_pattern = config.embedding_file_pattern
    else:
        # 기본 패턴: cache_dir 안에서 찾기
        embedding_file_pattern = "{cache_dir}/{dataset_name}/{dataset_name}_{model_name_safe}_embeddings.dat"
    
    for dataset_name in DATASET_NAMES:
        embedding_file_path = embedding_file_pattern.format(
            cache_dir=config.cache_dir,
            dataset_name=dataset_name,
            model_name_safe=model_name_safe
        )
        embedding_file = Path(embedding_file_path)
        if embedding_file.exists():
            # 파일 크기로부터 문서 수 계산
            file_size = embedding_file.stat().st_size
            num_docs = file_size // (dim_size * 4)  # float32 = 4 bytes
            # 임베딩 파일을 memmap으로 로드
            # make_embeddings.py에서 (tot_doc, d) shape으로 저장했으므로 동일하게 로드
            emb_mem = np.memmap(embedding_file, dtype="float32", mode="r", shape=(num_docs, dim_size))
            precomputed_embeddings_dict[dataset_name] = emb_mem
            print(f"✅ [INFO] {dataset_name} 임베딩 파일 로드 완료: {embedding_file}, shape: {emb_mem.shape}")
        else:
            print(f"⚠️  [WARNING] {dataset_name} 임베딩 파일이 없습니다: {embedding_file}")
            print(f"   임베딩 파일을 생성합니다...")
            
            # make_embeddings 함수 호출
            created_embedding_file = make_embeddings(dataset_name)
            
            # 생성된 파일 경로 확인
            if created_embedding_file:
                embedding_file = Path(created_embedding_file)
            
            # 파일이 생성되었는지 확인
            if embedding_file.exists():
                # 생성된 임베딩 파일 로드
                file_size = embedding_file.stat().st_size
                num_docs = file_size // (dim_size * 4)  # float32 = 4 bytes
                emb_mem = np.memmap(embedding_file, dtype="float32", mode="r", shape=(num_docs, dim_size))
                precomputed_embeddings_dict[dataset_name] = emb_mem
                print(f"✅ [INFO] {dataset_name} 임베딩 파일 생성 및 로드 완료: {embedding_file}, shape: {emb_mem.shape}")
            else:
                print(f"   ⚠️  임베딩 파일 생성 실패. 모델로 직접 인코딩합니다.")
                precomputed_embeddings_dict[dataset_name] = None

    for dim_to_drop in range(dim_size):
        dims_to_keep = list(range(dim_size))
        del dims_to_keep[dim_to_drop]
        
        # 각 dataset에 대해 별도의 Truncator 생성 (각 dataset마다 다른 임베딩 파일 사용)
        for dataset_name in DATASET_NAMES:
            precomputed_emb = precomputed_embeddings_dict.get(dataset_name)
            
            model = Truncator(
                encoder,
                resize_scale=1.0,
                cache_dir=config.cache_dir,
                indexes_to_keep=dims_to_keep,
                is_e5=config.is_e5_mistral,
                precomputed_embeddings=precomputed_emb,
            )
            # 각 dataset에 대해 인덱스 리셋 (순서 기반 매칭을 위해)
            model._corpus_start_index = 0
            
            evaluator = NanoBEIREvaluator(
                dataset_names=[dataset_name],
                score_functions={SimilarityFunction.COSINE.value: cos_sim}
                if not config.use_dot_product
                else {SimilarityFunction.DOT_PRODUCT.value: dot_score},
                query_prompts=task_name_to_instruct if config.is_e5_mistral else None,
                batch_size=8 if config.is_e5_mistral else 32,
            )
            result = evaluator(model)
            
            # 결과 저장
            key = f"{dataset_name}_dim_{dim_to_drop}"
            if key not in results:
                results[key] = {}
            results[key] = result
            sienna.save(results, config.output_path)
            print(f"✅ [INFO] {dataset_name}, dim_to_drop={dim_to_drop} 완료")
    
    print(f"✅ [INFO] 모든 결과 저장 완료: {config.output_path}")
