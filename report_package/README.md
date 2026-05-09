# TheMergeConflict Final Code README

Team: `TheMergeConflict`

Members:
- Jnanasree Konda, NetID: `jk9286`
- Rithwik Amajala, NetID: `sa9880`

Kaggle public leaderboard position: `6`

Public leaderboard score: `0.915`

Final submission file: `submission.csv`

This README describes how to reproduce the final code version only. The solution uses only the provided competition data and the allowed model `HuggingFaceTB/SmolVLM-500M-Instruct`.

## Code Files

- `TheMergeConflict_final_solution.ipynb`: final notebook for training the LoRA adapter, selecting the validation checkpoint, running deterministic answer-logit inference with test-time augmentation, averaging scores, and writing `submission.csv`.
- `requirements.txt`: Python package requirements for the notebook.
- `submission.csv`: final submitted prediction file.

## Expected Data Layout

Place the provided competition package next to this README or update the notebook config cell to point to its location:

```text
pixels-to-predictions/
  train.csv
  validation.csv          # or the provided validation file name
  test.csv
  images/                 # provided image files
```

Do not add external datasets, labels, synthetic examples, or external pretrained models. The notebook assumes the allowed SmolVLM checkpoint is available from the runtime cache or from the competition/offline environment.

## Local Virtual Environment Setup

Use this when running on a local VM or Chameleon Jupyter environment:

```bash
cd /path/to/report_package
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For Kaggle or Colab, use a fresh GPU runtime and run the notebook setup cell instead of creating a local virtual environment.

## Running the Final Notebook

1. Open `TheMergeConflict_final_solution.ipynb` in Jupyter, Kaggle, or Colab.
2. In the configuration cell, set `cfg.data_dir` to the folder containing `pixels-to-predictions/` if the default path is not correct.
3. Run all notebook cells from top to bottom in a fresh GPU runtime.
4. The notebook trains the parameter-efficient LoRA adapter under the 5M trainable-parameter limit.
5. The notebook selects the best checkpoint using validation accuracy.
6. The notebook runs deterministic answer-logit scoring with choice-permutation test-time augmentation.
7. The notebook averages compatible answer-score matrices produced by the final run.
8. The notebook writes the final file as `submission.csv`.

## Output

The final notebook produces:

```text
outputs/
  best_adapter/
  answer_scores.npy
  submission.csv
```

The submission must have exactly:

```text
id,answer
```

where `answer` is a 0-indexed integer corresponding to the selected option index.

## Final Validation

The last notebook cell validates the produced file. It checks:

- columns are exactly `id` and `answer`;
- there are 1008 prediction rows;
- ids are unique;
- answers are integer-valued;
- answers are 0-indexed and within the valid option range.

## Main Hyperparameters

- Base model: `HuggingFaceTB/SmolVLM-500M-Instruct`
- Adapter: LoRA
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- LoRA rank: `8`
- LoRA alpha: `32`
- LoRA dropout: `0.05`
- Trainable parameters: approximately `2.08M`, below the 5M competition limit
- Optimizer: AdamW
- Learning rate: `1e-4` for the main training run
- Micro batch size: `1`
- Gradient accumulation: `16`
- Image long side: `384` for the stable final setting
- Checkpoint selection: best validation accuracy

## Reproducibility Notes

- Use a fresh runtime before running the notebook to avoid stale CUDA, processor, or package-state issues.
- Keep `num_workers=0` in the notebook data loaders on Kaggle/Colab to avoid temporary-directory and multiprocessing failures.
- Keep answer decoding constrained to option logits; do not use free-form generated text for the final answer.
- The final package is intentionally notebook-only. No separate Python script files are required for reproduction.
