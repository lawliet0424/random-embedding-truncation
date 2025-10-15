# Dimension Attribution Analysis for Classification

This directory is in charge of *dimension attribution analysis* (Sec. 4 in the paper) with classification datasets.
Its functionality is super simple. It just repeats classification evalution N times (N being number of dimensions in the original embeddings) with one dimension removed each time so we can measure the contribution of each dimension on downstream tasks.

This would take more time than the retrieval counterpart, as we train classifier for each run.

## Run

You can start an experiment by simply running,

```sh
uv run  \
  python src/random_embedding_truncation/dimension_attribution_analysis_mteb/main.py \
  --config ./src/random_embedding_truncation/dimension_attribution_analysis_mteb/configs/all-mpnet-base-v2.toml
```

This command will do the evalution for all the classification datasets we used in our paper, and save the results in the path defined in the configuration file  (`all-mpnet-base-v2.toml` file in the example command). We have prepared more configuration files in the same directory. You can also write you own to test other models.
