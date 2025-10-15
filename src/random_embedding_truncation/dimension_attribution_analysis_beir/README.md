# Dimension Attribution Analysis for Retrieval


This directory is in charge of *dimension attribution analysis* (Sec. 4 in the paper) with retrieval datasets.
Its functionality is super simple. It just repeats retrieval evalution N times (N being number of dimensions in the original embeddings) with one dimension removed each time so we can measure the contribution of each dimension on downstream tasks.

## Run

You can start an experiment by simply running,

```sh
uv run  \
  python src/random_embedding_truncation/dimension_attribution_analysis_beir/main.py \
  --config ./src/random_embedding_truncation/dimension_attribution_analysis_beir/configs/t5-base.toml
```

This command will do the evalution for all the retrieval datasets we used in our paper, and save the results in the path defined in the configuration file  (`t5-base.toml` file in the example command).
