# LoRA Fine-Tuning of Decoder LLMs for Human-Preference Prediction

An end-to-end, tested pipeline that fine-tunes a decoder-only LLM with **LoRA** to predict which of two chatbot responses a human prefers. It is built on the **[LMSYS - Chatbot Arena Human Preference Predictions](https://www.kaggle.com/competitions/lmsys-chatbot-arena)** Kaggle competition (scored by multi-class log loss) and uses the public [55k-battle training set](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k) (Apache-2.0).

> **Status.** Data pipeline, training, evaluation, inference, baselines and tests are complete and tested. The non-LLM baselines below were measured on the real data. The fine-tuned model's numbers are added to the results table only after the GPU run is finished (see [Results](#results)); nothing in this README is estimated or copied from other solutions.

## The task

Each example is a user prompt and two responses (A and B) from different chatbots. The label is the voter's choice: **A**, **B** or **tie**. The model must output probabilities for the three outcomes, and the score is the log loss (lower is better; a constant uniform guess scores ln 3 = 1.0986).

Dataset facts, measured here: 57,477 battles; A 34.9%, B 34.2%, tie 30.9%; median input of 522 Qwen tokens per example (prompt + both responses), and 17.6% of examples exceed 1,024 tokens, which is why truncation is handled carefully (below).

## Approach

| Design choice | What it does | Why |
|---|---|---|
| Decoder LLM + 3-way classification head | `AutoModelForSequenceClassification` reads the last token of `[prompt, response A, response B, verdict marker]` | A classification head gives 3-class probabilities directly, which is what the log-loss metric needs |
| LoRA on all attention + MLP projections | Rank 16, alpha 32, dropout 0.05, new head kept trainable, trainable params cast to fp32 | Fine-tunes about 1 to 2% of the weights, so a 1.5B model fits on a free Kaggle T4; fp32 adapters keep fp16 training stable |
| Budget-aware truncation | Prompt gets at most 25% of the token budget (its most recent part is kept); the two responses split the rest and spare space goes to the longer one; the verdict marker is never cut | Naive truncation would drop the verdict token the head reads, or let one long answer crowd out the other |
| **A/B swap augmentation** | Randomly swap responses A and B (and flip the label) during training | The label is symmetric: swapping the order must swap the answer. This removes position bias and doubles effective data |
| **Swap test-time augmentation** | Average the prediction for (A,B) with the swapped prediction for (B,A) | Cheap ensembling that also reduces position bias; reported separately so its effect is measurable |
| Group-aware split | Rows with the same prompt never appear in both train and validation | Avoids leakage from repeated prompts that would flatter the score |
| Checkpoint selection on log loss | `load_best_model_at_end` on a validation subset, final scoring on the full held-out split | Matches the competition metric |

This repository is sized for a free GPU and does not claim to reach leaderboard-level scores.

## Results

Validation split: 5,693 battles (10%, grouped by prompt, seed 42). Metric: 3-class log loss (lower is better).

| Model | Log loss | Accuracy | Notes |
|---|---|---|---|
| Class prior (constant) | 1.0973 | 0.349 | measured |
| Length/style features + logistic regression | 1.0538 | 0.455 | measured; 10 hand-crafted response features, A/B-symmetrised |
| Length/style features + gradient-boosted trees | 1.0386 | 0.464 | measured |
| **LoRA fine-tuned LLM (this repo)** | **pending** | **pending** | filled in after the Kaggle GPU run; `outputs/<run>/metrics.json` records the exact configuration |

The baselines show how much signal sits in simple length and formatting cues (human voters favour longer, structured answers), so the fine-tuned model should be judged against 1.0386, not only against the constant prior.

Reproduce the baselines (about 1.5 minutes on a laptop CPU):

```bash
python scripts/run_baselines.py --data data/train.csv --out results/baselines.json
```

## Quick start

```bash
git clone https://github.com/SkyWalkersCity/llm-preference-finetuning.git
cd llm-preference-finetuning
pip install -e ".[dev]"

# data (public Hugging Face copy of the competition training set)
mkdir -p data && curl -L -o data/train.csv \
  https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k/resolve/main/train.csv

python -m pytest -q                                      # unit tests + end-to-end smoke test
python scripts/run_baselines.py --data data/train.csv --out results/baselines.json

# fine-tune (needs a GPU; see notebooks/kaggle_train.ipynb for the Kaggle version)
python -m lmsys_pref.train --data data/train.csv --preset qwen1.5b --output outputs/qwen1.5b \
    --max_train_samples 20000 --epochs 1

# predict on a Kaggle-format test file (id, prompt, response_a, response_b)
python -m lmsys_pref.infer --adapter outputs/qwen1.5b/adapter --test test.csv --out submission.csv
```

Presets (`--preset`): `qwen0.5b`, `qwen1.5b` (default), `qwen3b`, and `gemma2-9b-4bit` (QLoRA, needs a larger GPU and `bitsandbytes`). Any option can be overridden on the command line, for example `--max_len 768` or `--lora_r 32`.

### Running on Kaggle

Open `notebooks/kaggle_train.ipynb` on Kaggle with a GPU accelerator and Internet enabled. It runs a short smoke test, the baselines, the real fine-tune and prints the metrics. Training on a T4 uses fp16 automatically (bf16 where supported).

## Repository layout

```
src/lmsys_pref/
  data.py        parsing of the JSON-in-CSV format, budget-aware prompt packing, A/B swap augmentation, group split
  model.py       classification model + LoRA wrapper (optional 4-bit QLoRA)
  train.py       Hugging Face Trainer entry point, presets, final plain + swap-TTA evaluation
  infer.py       adapter loading and Kaggle-format submission with swap TTA
  metrics.py     log loss, accuracy, expected calibration error
  baselines.py   class prior and length/style baselines
scripts/run_baselines.py
notebooks/kaggle_train.ipynb
tests/           15 tests, including an end-to-end train + infer smoke test on a tiny random Qwen2 model
results/baselines.json
```

## Testing

`python -m pytest -q` runs 15 tests: JSON/NaN parsing, label validation, leakage-free splitting, truncation never exceeding `max_len` with the verdict marker intact, swap augmentation correctness (swapped input equals manually swapped data, labels flip), metric parity with scikit-learn, baseline sanity, and a real end-to-end run (LoRA training, checkpointing, evaluation, adapter reload, submission format) on a tiny randomly initialised Qwen2 model. The tokenizer test fixture downloads the Qwen2.5 tokenizer (a few MB) and the tests skip if the hub is unreachable.

## Limitations

- The fine-tune is sized for a free Kaggle GPU (about 1.5B parameters, a subset of the training data, a single epoch), so a larger model trained on more data and hardware would score better.
- Validation uses a held-out 10% of the public training set, not the competition's private test set, so numbers are comparable only to the baselines in this README, not to the leaderboard.
- Ties are noisy by nature (the "tie" label merges "both good" and "both bad"), which caps achievable accuracy.
- Examples longer than `max_len` are truncated; about 17.6% of examples exceed 1,024 tokens.

## AI assistance

This project was developed with assistance from Claude (Anthropic), an AI assistant, which helped write and review code, tests and documentation. All measured results in this README come from runs of the code in this repository.

## Data and citation

Data: [`lmarena-ai/arena-human-preference-55k`](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k), Apache-2.0. Please cite the Chatbot Arena paper:

```
@misc{chiang2024chatbot,
  title={Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference},
  author={Wei-Lin Chiang and Lianmin Zheng and Ying Sheng and Anastasios Nikolas Angelopoulos and Tianle Li and Dacheng Li and Hao Zhang and Banghua Zhu and Michael Jordan and Joseph E. Gonzalez and Ion Stoica},
  year={2024}, eprint={2403.04132}, archivePrefix={arXiv}, primaryClass={cs.AI}
}
```

## License

MIT, see [LICENSE](LICENSE).
