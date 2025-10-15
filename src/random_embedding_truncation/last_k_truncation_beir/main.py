import logging
import os
from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from pathlib import Path

import sienna
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation.NanoBEIREvaluator import NanoBEIREvaluator
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.util import cos_sim, dot_score

from random_embedding_truncation.truncator import (
    Truncator,
    task_name_to_instruct,
)
from random_embedding_truncation.utils import (
    convert_dataset_name_for_nanobeir,
    read_toml,
)


@dataclass(frozen=True)
class Result:
    dataset_name: str
    model: str
    scale: float | int
    ndcg: dict[str, float] | None
    _map: dict[str, float] | None
    recall: dict[str, float] | None
    precision: dict[str, float] | None
    mrr: tuple[dict[str, float]] | None

    def get_save_path(self, output_dir: Path) -> Path:
        fname = f"{self.dataset_name}_{self.model.replace('/', '-')}_{self.scale}.json"
        save_path = output_dir / fname
        return save_path

    def save(self, output_dir: Path) -> Path:
        save_path = self.get_save_path(output_dir)
        sienna.save(asdict(self), str(save_path))
        return save_path

    def does_exist(self, output_dir: Path) -> bool:
        save_path = self.get_save_path(output_dir)
        return save_path.exists()


@dataclass(frozen=True)
class BEIREvaluator:
    dataset_name: str
    batch_size: int

    def run(self, model: Truncator, model_name: str, scale: float) -> Result:
        dataset_name = self.dataset_name
        model = DRES(model, batch_size=self.batch_size)
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(
            dataset_name
        )
        out_dir = os.path.join(Path(__file__).parent.absolute(), "datasets")
        data_path = util.download_and_unzip(url, out_dir)
        ## Provide the data_path where scifact has been downloaded and unzipped
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(
            split="test"
        )
        is_contriever = "contriever" in model_name
        ## Load the SBERT model and retrieve using cosine-similarity
        retriever = EvaluateRetrieval(
            model, score_function="dot" if is_contriever else "cos_sim"
        )
        results = retriever.retrieve(corpus, queries)

        ndcg, _map, recall, precision = retriever.evaluate(
            qrels, results, retriever.k_values
        )
        mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, "mrr")
        result = Result(
            dataset_name, model_name, scale, ndcg, _map, recall, precision, mrr
        )
        return result


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    config = read_toml(args.config)
    batch_size: int = config["batch_size"]
    model_name: str = config["model_name"]
    dataset_name: str = config["dataset_name"]
    result_output_dir: Path = Path(config["result_output_dir"])
    cache_dir: Path = Path(config["cache_dir"])

    if not result_output_dir.exists():
        result_output_dir.mkdir(parents=True, exist_ok=True)

    is_e5_mistral = "e5-mistral" in model_name
    is_e5_large = "e5-large" in model_name
    is_contriever = "contriever" in model_name

    if is_e5_mistral:
        query_prompt = task_name_to_instruct[dataset_name]
    else:
        query_prompt = None

    st_model = SentenceTransformer(model_name)

    for scale_idx in range(1, 11):
        scale = scale_idx / 10

        _result = Result(dataset_name, model_name, scale, None, None, None, None, None)

        pooler = Truncator(
            st_model,
            resize_scale=scale,
            is_e5=is_e5_large,
            cache_dir=cache_dir,
            indexes_to_keep=None,
            query_prompt=query_prompt,
        )

        if is_e5_mistral:
            dataset_name = dataset_name
            dataset_name = convert_dataset_name_for_nanobeir(dataset_name)
            logging.info("Using NanoBEIR for E5-Mistral")
            evaluator = NanoBEIREvaluator(
                dataset_names=[dataset_name],
                score_functions={SimilarityFunction.COSINE.value: cos_sim}
                if not is_contriever
                else {SimilarityFunction.DOT_PRODUCT.value: dot_score},
                query_prompts=task_name_to_instruct if is_e5_mistral else None,
                batch_size=8 if is_e5_mistral else 32,
            )
            result = evaluator(pooler)
            sienna.save(result, _result.get_save_path(result_output_dir))
        else:
            evaluator = BEIREvaluator(dataset_name, batch_size)
            result = evaluator.run(pooler, model_name, scale)
            result.save(result_output_dir)
