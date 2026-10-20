# Financial Sentiment Analysis with Domain-Adapted Transformers

A comparison of a finance-specific transformer, a general-domain transformer and classical baselines on three-class financial sentiment classification, with both transformers fine-tuned under identical downstream settings.

## Overview

This project tests whether a finance-specific transformer improves sentence-level sentiment classification over a general-domain transformer and a traditional classifier. The focus is on methodology rather than benchmark performance: the same data splits, sequence length, optimiser, class-weighting scheme and random seed apply to both transformers. The starting checkpoints differ in more than their pre-training corpus: `yiyanghkust/finbert-tone` was pre-trained on financial text and then fine-tuned for sentiment on about 10,000 labelled analyst-report sentences, and the comparison therefore measures the combined effect of financial pre-training and prior sentiment training, not domain pre-training alone. Four models are compared, from a trivial floor to a finance-adapted transformer, and the analysis pays particular attention to the minority classes that carry most of the downstream signal in financial text, where missing a negative sentence matters more than a clean overall score.

## Data

* **Source:** Financial PhraseBank (Malo et al., 2014), loaded from the HuggingFace `datasets` library under `takala/financial_phrasebank`, `sentences_allagree` configuration (only sentences with unanimous annotator agreement).
* **Size:** 2,264 sentences, around 68,000 tokens.
* **Target:** three-class sentiment, negative, neutral and positive.
* **Class balance:** 61.4% neutral (1,391), 25.2% positive (570), 13.4% negative (303), so accuracy is misleading and macro-F1 is the primary metric.

The dataset is not stored in this repository. It loads automatically through the `datasets` library when the notebook runs, and no manual download is needed. Financial PhraseBank is distributed under the CC BY-NC-SA 3.0 licence; the MIT licence in this repository covers the code only.

## Approach

1. **Preprocessing (classical branch only).** Regex cleaning, lowercasing, negation handling that prefixes `NOT_` to the token after a negation cue, stopword removal with negation cues kept, and lemmatisation. Negation handling runs before stopword removal so that cues such as "not" and "never" are not stripped before they can reverse the following token. Transformers receive raw text, because their subword tokenisers depend on the original surface forms.
2. **Stratified split.** 80/10/10 over integer indices with seed 42, preserving class proportions (1,811 train, 226 validation, 227 test). The validation partition is used for early stopping and best-checkpoint selection, and supplied 12 of the 24 examples in the qualitative error analysis; it is never used for the reported test metrics.
3. **Baselines.** A Majority Class predictor (scikit-learn `DummyClassifier`) sets the floor, and a TF-IDF logistic regression with unigram and bigram features and sublinear term-frequency scaling bounds what surface lexical features alone can achieve.
4. **General-domain transformer.** `bert-base-uncased` fine-tuned through the HuggingFace Trainer, with learning rate 2e-5, batch size 16, up to 5 epochs with early stopping on validation macro-F1 (patience 2), weight decay 0.01, and a maximum sequence length of 64 tokens chosen by a 95% coverage rule.
5. **Finance-adapted transformer.** `yiyanghkust/finbert-tone` fine-tuned under the identical hyperparameters above. Its existing three-class sentiment head is loaded and retrained on the PhraseBank labels.
6. **Class-weighted loss.** A custom `WeightedTrainer` applies balanced class weights to cross-entropy, preferred over oversampling because resampling distorts the per-batch label distribution.
7. **Evaluation.** Macro-F1 as the primary metric, with macro precision, recall, weighted F1 and per-class F1 reported, plus a manual error analysis of 24 misclassified sentences grouped into recurring failure types (12 from the test set, which held only 12 distinct transformer errors, and 12 from the validation set).

## Results

| Model | Macro-F1 | Negative-class F1 |
|---|---|---|
| Majority Class | 0.2532 | 0.00 |
| TF-IDF + Logistic Regression | 0.7948 | 0.78 |
| BERT (bert-base-uncased) | 0.9302 | 0.87 |
| FinBERT (finbert-tone) | 0.9422 | 0.92 |

Moving from surface features to contextual representation adds about 13.5 macro-F1 points (TF-IDF to BERT). The finance-specific checkpoint adds a further 1.2 points on the test set (BERT to FinBERT), concentrated on the minority negative class, the category that matters most for risk monitoring. The gain is small and should be read as indicative: both transformers misclassified 9 of the 227 test sentences, FinBERT's advantage comes from missing 3 of the 31 negative sentences against BERT's 5, and during training BERT reached the higher best validation macro-F1 (0.967 against 0.955). The manual error analysis found recurring structural difficulties (contrastive clauses, negation and forward-looking statements) rather than vocabulary gaps.

## Limitations

* The two transformer checkpoints have different training histories (see Overview); the comparison therefore does not isolate the effect of domain pre-training.
* During best-checkpoint restoration in the recorded run (Transformers 5.0.0), the logs report 50 LayerNorm tensors missing from the checkpoint, saved under legacy gamma/beta names. The restored models therefore cannot be assumed to match the selected checkpoint exactly; the reported scores are those of the models actually evaluated.
* The test set holds 227 sentences (31 negative), and a difference of one or two sentences moves per-class scores noticeably.

## Tech stack

Python, PyTorch, HuggingFace Transformers, HuggingFace datasets, scikit-learn, nltk, NumPy, matplotlib, seaborn, Google Colab Pro.

## Files

* `financial-sentiment-analysis.ipynb` runs the full workflow: preprocessing, baselines, both transformer fine-tuning runs, evaluation, confusion matrices, and the manual error analysis.

The notebook was developed in Google Colab Pro on a single T4 GPU and mounts Google Drive for outputs. The fine-tuned model weights (about 780 MB) are not included in this repository. The recorded run used Python 3.12, PyTorch 2.10, Transformers 5.0.0, Datasets 3.6.0, scikit-learn 1.6.1 and NLTK 3.9.1. The dataset loads through a Hugging Face loading script, which Datasets 4.0 and later no longer support.

## Reproducibility

Random seeds are fixed at 42 across Python, NumPy, PyTorch, CUDA and the HuggingFace `set_seed` utility. This fixes the stratified split, the class weights and the training run. GPU fine-tuning can carry small run-to-run variation, and a rerun may differ slightly from the reported figures.

## Notes

This project was developed during the INM434 Natural Language Processing module of the MSc Data Science programme at City St George's, University of London. It applies transformer fine-tuning, domain adaptation, class-imbalance handling through weighted loss, NLP preprocessing, and manual error analysis, skills that transfer directly to financial text analytics and risk-signal extraction in financial services.
