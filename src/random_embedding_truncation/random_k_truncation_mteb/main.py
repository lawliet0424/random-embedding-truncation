import logging
from argparse import ArgumentParser
from pathlib import Path

import mteb
from sentence_transformers import SentenceTransformer

from random_embedding_truncation.truncator import Truncator
from random_embedding_truncation.utils import make_mask, read_toml

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    config = read_toml(args.config)
    batch_size: int = config["batch_size"]
    model_name: str = config["model_name"]
    result_output_dir: Path = Path(config["result_output_dir"])
    cache_dir: Path = Path(config["cache_dir"])

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
        "ToxicConversationsClassification",
        "TweetSentimentExtractionClassification",
    ]

    st = SentenceTransformer(model_name)

    masks = [
        [make_mask(1.0, st.get_sentence_embedding_dimension()) for _ in range(10)]
        for _ in range(10)
    ]

    for scale_idx in range(1, 11):
        scale = scale_idx / 10

        for mask_idx in range(10):
            for task in TASK_LIST_CLASSIFICATION:
                output_folder = (
                    result_output_dir
                    / f"{task}_{model_name.replace('/', '-')}_{scale}_id-{mask_idx}"
                )
                logging.info(
                    f"{output_folder} DOES NOT exist yet, running experiments."
                )

                end = int(st.get_sentence_embedding_dimension() * scale)
                indexes_to_keep = masks[scale_idx - 1][mask_idx][:end]

                model = Truncator(
                    st,
                    resize_scale=scale,
                    indexes_to_keep=indexes_to_keep,
                    cache_dir=cache_dir / task,
                )
                eval_splits = ["dev"] if task == "MSMARCO" else ["test"]
                evaluation = mteb.MTEB(tasks=[task], task_langs=["en"])
                evaluation.run(
                    model, output_folder=output_folder, eval_splits=eval_splits
                )
