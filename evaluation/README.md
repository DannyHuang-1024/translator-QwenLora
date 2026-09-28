# Evaluation and report artifacts

This directory evaluates the base Qwen3 model and the LoRA adapter on the
same held-out translation examples. It writes all raw predictions and report
artifacts to a directory on Google Drive so the experiment can be reproduced
after a Colab runtime ends.

## Colab command

Mount Drive in a Notebook Python cell first, then run from the repository:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
cd /content/translator-QwenLora
python -m pip install "sacrebleu>=2.4" "matplotlib>=3.8"

PYTHONPATH=src python evaluation/evaluate.py \
  --adapter /content/drive/MyDrive/qwen3-translation/checkpoints/qwen3-0.6b-en-zh-lora/final \
  --output-dir /content/drive/MyDrive/qwen3-translation/evaluation/qwen3-0.6b-en-zh-lora \
  --limit 200 \
  --batch-size 4
```

Use `--limit 0` to evaluate every example in the selected split. The first
run should use 100-200 examples to check the pipeline. The same command with
`--limit 1000` is suitable for the course report if runtime permits.

## Saved files

```text
predictions.jsonl       source, reference, base output, LoRA output, lengths
sentence_metrics.csv    per-example BLEU/chrF and paired differences
summary.json             corpus metrics, win/tie/loss counts, run metadata
metrics_table.tex        LaTeX table for the main automatic evaluation
examples_table.tex       LaTeX table with representative translations
training_curves.png      train/eval loss and learning-rate curves
model_metrics.png        base vs LoRA BLEU and chrF
sentence_delta.png       distribution of per-example chrF improvements
win_tie_loss.png         number of sentences improved, tied, or degraded
length_comparison.png   output length comparison and length ratios
length_bucket_delta.png mean chrF change grouped by reference length
*.pdf                   vector versions of the figures for LaTeX
```

The corpus BLEU is computed on Chinese character-segmented text because
standard English tokenization is not appropriate for Chinese. chrF is also
reported because it is less sensitive to Chinese word segmentation. Automatic
metrics should be accompanied by the saved qualitative examples in the report.
