from argparse import ArgumentParser
from pathlib import Path

import mteb
from sentence_transformers import SentenceTransformer

from random_embedding_truncation.truncator import Truncator
from random_embedding_truncation.utils import read_toml

TASK_LIST_CLASSIFICATION = [
    "AmazonCounterfactualClassification",
    "AmazonPolarityClassification",
    "AmazonReviewsClassification",
    "Banking77Classification",
    "EmotionClassification",
    "ImdbClassification",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "MTOPDomainClassification",
    "MTOPIntentClassification",
    "ToxicConversationsCassification",
    "TweetSentimentExtractionClassification",
]

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    config = read_toml(args.config)
    batch_size: int = config["batch_size"]
    model_name: str = config["model_name"]
    result_output_dir: Path = Path(config["result_output_dir"])
    cache_dir: Path = Path(config["cache_dir"])
    tasks: list[str] = config.get("tasks", TASK_LIST_CLASSIFICATION)

    is_e5_mistral = "e5-mistral" in model_name

    st = SentenceTransformer(model_name)

    for scale_idx in range(1, 10):
        scale = scale_idx / 10
        for task in tasks:
            output_folder = (
                result_output_dir / f"{task}_{model_name.replace('/', '-')}_{scale}"
            )

            model = Truncator(st, resize_scale=scale, cache_dir=cache_dir / task)
            eval_splits = ["dev"] if task == "MSMARCO" else ["test"]
            evaluation = mteb.MTEB(tasks=[task], task_langs=["en"])
            evaluation.run(
                model,
                output_folder=str(output_folder),
                eval_splits=eval_splits,
                encode_kwargs={"batch_size": batch_size},
            )
