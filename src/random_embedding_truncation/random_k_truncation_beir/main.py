import logging
from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from pathlib import Path

import sienna
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation.NanoBEIREvaluator import NanoBEIREvaluator
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.util import cos_sim, dot_score

from random_embedding_truncation.truncator import Truncator, task_name_to_instruct
from random_embedding_truncation.last_k_truncation_beir.main import BEIREvaluator
from random_embedding_truncation.utils import make_mask, read_toml


@dataclass(frozen=True)
class Result:
    dataset_name: str
    model: str
    scale: float
    mask_idx: int
    ndcg: dict[str, float]
    _map: dict[str, float]
    recall: dict[str, float]
    precision: dict[str, float]
    mrr: tuple[dict[str, float]] | None

    def get_save_path(self, output_dir: Path) -> Path:
        fname = f"{self.dataset_name}_{self.model.replace('/', '-')}_{self.scale}_id-{self.mask_idx}.json"
        return output_dir / fname

    def save(self, output_dir: Path) -> Path:
        save_path = self.get_save_path(output_dir)
        sienna.save(asdict(self), str(save_path))
        return save_path

    def does_exist(self, output_dir: Path) -> bool:
        save_path = self.get_save_path(output_dir)
        return save_path.exists()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %I:%M:%S %p"
    )

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

    st_model = SentenceTransformer(model_name)

    is_e5_mistral = "e5-mistral" in model_name
    if is_e5_mistral:
        query_prompt = task_name_to_instruct[dataset_name]
    else:
        query_prompt = None

    is_contriever = "contriever" in model_name
    is_e5_large = "e5-large" in model_name

    for scale_idx in range(1, 11):
        for mask_idx in range(10):
            scale = scale_idx / 10
            logging.info(f"scale: {scale}. mask-idx: {mask_idx}")

            _result = Result(
                dataset_name, model_name, scale, mask_idx, {}, {}, {}, {}, None
            )
            indexes_to_keep = make_mask(
                scale, st_model.get_sentence_embedding_dimension()
            )
            pooler = Truncator(
                st_model,
                resize_scale=scale,
                is_e5=is_e5_large,
                cache_dir=cache_dir,
                indexes_to_keep=indexes_to_keep,
                query_prompt=query_prompt,
            )

            if is_e5_mistral:
                logging.info("Using NanoBEIR for E5-Mistral")
                dataset_name = dataset_name
                if dataset_name == "climate-fever":
                    dataset_name = "climatefever"
                if dataset_name == "dbpedia-entity":
                    dataset_name = "dbpedia"
                if dataset_name == "fiqa":
                    dataset_name = "fiqa2018"
                if dataset_name == "quora":
                    dataset_name = "quoraretrieval"
                if dataset_name == "webis-touche2020":
                    dataset_name = "touche2020"

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
                result = Result(
                    result.dataset_name,
                    result.model,
                    result.scale,
                    mask_idx,
                    result.ndcg,
                    result._map,
                    result.recall,
                    result.precision,
                    result.mrr,
                )
                result.save(result_output_dir)
