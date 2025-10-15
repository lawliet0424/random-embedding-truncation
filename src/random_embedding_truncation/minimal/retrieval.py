"""Full-sized original embeddings v.s. truncated 50% embeddings on a retrieval dataset"""

from pathlib import Path
import tempfile

from sentence_transformers.evaluation.NanoBEIREvaluator import NanoBEIREvaluator
from random_embedding_truncation.truncator import Truncator
from sentence_transformers import SentenceTransformer, SimilarityFunction
from sentence_transformers.util import cos_sim


def main():
    dataset_name = "msmarco"  # Select one from https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/evaluation/NanoBEIREvaluator.py#L22
    model_name = "sentence-transformers/sentence-t5-base"

    model = SentenceTransformer(model_name)

    evaluator = NanoBEIREvaluator(
        dataset_names=[dataset_name],
        batch_size=32,
        score_functions={SimilarityFunction.COSINE.value: cos_sim},
    )

    with (
        tempfile.TemporaryDirectory() as tmpdirname
    ):  # Cache to skip feed-forward computation for the halved embedding encoding
        print("Evaluation with full-sized embeddings.")

        # We do pass model to Truncator even for the full-sized embedding experiments.
        # This is just for efficiency by caching.
        # As we set 1.0 to `resize_scale` argument, it will **not** perform any truncation.
        truncator = Truncator(model, resize_scale=1.0, cache_dir=Path(tmpdirname))

        full_emb_result = evaluator(truncator)

        print("Evaluation with half-sized embeddings.")

        # This time, 0.5 is set to `resize_scale`. Meaning, that half (50%) of last dimension will be removed.
        truncator = Truncator(model, resize_scale=0.5, cache_dir=Path(tmpdirname))

        half_emb_result = evaluator(truncator)

    metric_key = "NanoBEIR_mean_cosine_ndcg@10"

    print(f"Used metric: {metric_key}")
    print(f"full: {full_emb_result[metric_key]:.2f}")
    print(f"half: {half_emb_result[metric_key]:.2f}")
    print(
        f"Relative performance: {((half_emb_result[metric_key] / full_emb_result[metric_key]) * 100):.1f}%"
    )


if __name__ == "__main__":
    main()
