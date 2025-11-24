# Dimension Attribution Analysis for Retrieval

This directory is in charge of *dimension attribution analysis* (Sec. 4 in the paper) with retrieval datasets.
Its functionality is super simple. It repeats retrieval evaluation N times (N being the number of dimensions in the original embeddings) with one dimension removed each time, so we can measure the contribution of each dimension on downstream tasks.

## How It Works

For each dimension in the embedding space (0 to N-1):
1. Create a `Truncator` that keeps all dimensions except one (specified by `indexes_to_keep`)
2. For each dataset in `DATASET_NAMES`:
   - Evaluate the model using `NanoBEIREvaluator`
   - Save the results with key `{dataset_name}_dim_{dim_to_drop}`
3. Results are saved incrementally to the JSON file after each evaluation

The script automatically handles:
- **Precomputed embeddings**: If embedding files exist in `cache_dir`, they are loaded and used. If not, they are automatically generated using the model and saved for future use.
- **Embedding file pattern**: You can customize the path pattern via `embedding_file_pattern` in the config file.

## Run

### Basic Usage

You can start an experiment by simply running,

```sh
uv run  \
  python src/random_embedding_truncation/dimension_attribution_analysis_beir/main.py \
  --config ./src/random_embedding_truncation/dimension_attribution_analysis_beir/configs/all-MiniLM-L6-v2.toml
```

### Execution Flow

This command will:

1. **Load configuration** from the specified TOML file
2. **Load or generate embeddings** for each dataset:
   - If embedding files exist in `cache_dir`, they are loaded
   - If not, they are automatically generated using the model
3. **For each dimension** (0 to N-1, where N is the embedding dimension):
   - Create a `Truncator` that removes that dimension (`indexes_to_keep` contains all dimensions except the dropped one)
   - For each dataset in `DATASET_NAMES`:
     - Evaluate using `NanoBEIREvaluator`
     - Save results with key `{dataset_name}_dim_{dim_to_drop}` to the JSON file
4. **Results are saved incrementally** after each evaluation, so you can resume from where you left off if the process is interrupted

### Example Output

```
✅ [INFO] msmarco 임베딩 파일 로드 완료: /path/to/msmarco_embeddings.dat, shape: (8841823, 384)
✅ [INFO] scidocs 임베딩 파일 로드 완료: /path/to/scidocs_embeddings.dat, shape: (27503, 384)
✅ [INFO] msmarco, dim_to_drop=0 완료
✅ [INFO] scidocs, dim_to_drop=0 완료
✅ [INFO] msmarco, dim_to_drop=1 완료
✅ [INFO] scidocs, dim_to_drop=1 완료
...
✅ [INFO] 모든 결과 저장 완료: ./outputs/one_dim_drop/results/all-MiniLM-L6-v2.json
```

### Running with Different Models

To run with a different model, simply change the config file:

```sh
uv run  \
  python src/random_embedding_truncation/dimension_attribution_analysis_beir/main.py \
  --config ./src/random_embedding_truncation/dimension_attribution_analysis_beir/configs/t5-base.toml
```

## Configuration

Example config file (`all-MiniLM-L6-v2.toml`):

```toml
cache_dir = "/media/hyunji/muvera_optimized/embeddings/all-MiniLM-L6-v2"
output_path = "./outputs/one_dim_drop/results/all-MiniLM-L6-v2.json"
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_file_pattern = "{cache_dir}/{dataset_name}/{dataset_name}_{model_name_safe}_embeddings.dat"
```

### Config Options

- `cache_dir`: Directory for storing precomputed embeddings and model cache. Embedding files are stored as `{cache_dir}/{dataset_name}/{dataset_name}_{model_name_safe}_embeddings.dat`
- `output_path`: Path to save evaluation results (JSON format). Results are saved incrementally with keys like `{dataset_name}_dim_{dim_to_drop}`
- `model_name`: HuggingFace model name or path
- `embedding_file_pattern`: Optional pattern for embedding file paths. Available placeholders: `{cache_dir}`, `{dataset_name}`, `{model_name_safe}`. If not specified, defaults to `{cache_dir}/{dataset_name}/{dataset_name}_{model_name_safe}_embeddings.dat`

### Precomputed Embeddings

The script automatically handles precomputed embeddings:

- **If embedding files exist**: They are loaded from `cache_dir` and used directly
- **If embedding files don't exist**: They are automatically generated using the model and saved to `cache_dir` for future use

You can also manually generate embeddings using `make_embeddings.py`:

```sh
python src/random_embedding_truncation/dimension_attribution_analysis_beir/make_embeddings.py <dataset_name>
```

But this is optional since `main.py` will generate them automatically if needed.
