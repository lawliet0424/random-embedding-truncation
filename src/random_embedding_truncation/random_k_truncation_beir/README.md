# Random K% Truncation for Retrieval

This directory is in charge of *Random K% Truncation* (Sec. 2 in the paper) with retrieval datasets.
Truncation level (K) changes from 100% to 10% with 10% step, and perform retrieval evaluation with each truncation level and save the performance.

## Run

You can start an experiment by simply running,

```sh
uv run  \
  python ./src/random_embedding_truncation/random_k_truncation_beir/main.py \
  --config ./src/random_embedding_truncation/random_k_truncation_beir/configs/sentence-t5/sentence-t5-arguana.toml
```

This command will do the evalution for the model and the dataset specified in the configuration file, and save the results in the path defined in the configuration file  (`all-mpnet-base-v2.toml` file in the example command).
We have prepared more configuration files in the same directory. You can also write you own to test other models.
