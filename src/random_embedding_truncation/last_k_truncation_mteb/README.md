# Last K% Truncation for Classification

This directory is in charge of *Last K% Truncation* (Sec. 2 in the paper) with classification datasets.
Truncation level (K) changes from 100% to 10% with 10% step, and perform classification evaluation with each truncation level and save the performance.

This would take more time than the retrieval counterpart as we train classifier for each run.

## Run

You can start an experiment by simply running,

```sh
uv run  \
  python ./src/random_embedding_truncation/last_k_truncation_mteb/main.py \
  --config ./src/random_embedding_truncation/last_k_truncation_mteb/configs/all-mpnet-base-v2/all-mpnet-base-v2.toml
```

This command will do the evalution for the model specified in the configuration file on all classification datasets used in our paper, and save the results in the path defined in the configuration file  (`all-mpnet-base-v2.toml` file in the example command).
We have prepared more configuration files in the same directory. You can also write you own to test other models.
