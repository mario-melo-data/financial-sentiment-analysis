# Financial Sentiment Analysis with Domain-Adapted Transformers

A controlled comparison of a finance-adapted transformer, a general-domain transformer, and a classical baseline on three-class financial sentiment classification, designed so that any performance gap is attributable to the pre-training corpus alone.

## Overview

This project tests whether pre-training a transformer on financial text improves sentence-level sentiment classification over a general-domain transformer and a traditional classifier. The focus is on methodology rather than benchmark performance: every non-target factor is held constant, so that the same data splits, sequence length, optimiser, class-weighting scheme and random seed apply to both transformers, leaving the pre-training corpus as the only variable. Four models are compared, from a trivial floor to a finance-adapted transformer, and the analysis pays particular attention to the minority classes that carry most of the downstream signal in financial text, where missing a negative sentence matters more than a clean overall score.

## Data

* **Source:** Financial PhraseBank (Malo et al., 2014), loaded from the HuggingFace `datasets` library under `takala/financial_phrasebank`, `sentences_allagree` configuration (only sentences with unanimous annotator agreement).
* **Size:** 2,264 sentences, around 68,000 tokens.
* **Target:** three-class sentiment, negative, neutral and positive.
* **Class balance:** 61.4% neutral (1,391), 25.2% positive (570), 13.4% negative (303), so accuracy is misleading and macro-F1 is the primary metric.

The dataset is not stored in this repository. It loads automatically through the `datasets` library when the notebook runs, so no manual download is needed.

## Approach

1. **Preprocessing (classical branch only).** Regex cleaning, lowercasing, negation handling that prefixes `NOT_` to the token after a negation cue, stopword removal with negation cues kept, and lemmatisation. Negation handling runs before stopword removal so that cues such as "not" and "never" are not stripped before they can reverse the following token. Transformers receive raw text, because their subword tokenisers depend on the original surface forms.
2. **Stratified split.** 80/10/10 over integer indices with seed 42, preserving class proportions (1,811 train, 227 test). The validation partition is used only for early stopping and is never scored as test.
3. **Baselines.** A Majority Class predictor (scikit-learn `DummyClassifier`) sets the floor, and a TF-IDF logistic regression with unigram and bigram features and sublinear term-frequency scaling bounds what surface lexical features alone can achieve.
4. **General-domain transformer.** `bert-base-uncased` fine-tuned through the HuggingFace Trainer, with learning rate 2e-5, batch size 16, up to 5 epochs with early stopping on validation macro-F1 (patience 2), weight decay 0.01, and a maximum sequence length of 64 tokens chosen by a 95% coverage rule.
5. **Finance-adapted transformer.** `yiyanghkust/finbert-tone` fine-tuned under the identical hyperparameters above, so the pre-training corpus is the only difference between the two.
6. **Class-weighted loss.** A custom `WeightedTrainer` applies balanced class weights to cross-entropy, preferred over oversampling because resampling distorts the per-batch label distribution.
7. **Evaluation.** Macro-F1 as the primary metric, with macro precision, recall, weighted F1 and per-class F1 reported, plus a manual error analysis of 24 misclassified sentences grouped into recurring failure types.

## Results

| Model | Macro-F1 | Negative-class F1 |
|---|---|---|
| Majority Class | 0.2532 | 0.00 |
| TF-IDF + Logistic Regression | 0.7948 | 0.78 |
| BERT (bert-base-uncased) | 0.9302 | 0.87 |
| FinBERT (finbert-tone) | 0.9422 | 0.92 |

Moving from surface features to contextual representation adds about 13.5 macro-F1 points (TF-IDF to BERT). Finance-specific pre-training adds a further 1.2 points (BERT to FinBERT), and that gain sits almost entirely on the minority negative class, the category that matters most for risk monitoring. The hypothesis holds in direction but the gain is modest: on this unanimous-agreement subset, strong general pre-training already performs well, and around half the residual errors are structural (contrastive clauses, negation, forward-looking statements) rather than vocabulary-driven, which domain adaptation cannot fix.

## Tech stack

Python, PyTorch, HuggingFace Transformers, HuggingFace datasets, scikit-learn, nltk, NumPy, matplotlib, seaborn.

## Files

* `financial-sentiment-analysis.ipynb` runs the full workflow: preprocessing, baselines, both transformer fine-tuning runs, evaluation, confusion matrices, and the manual error analysis.

To reproduce the results, run the notebook in a GPU runtime (the transformers fine-tune on a single T4). The dataset loads automatically from HuggingFace.

## Reproducibility

Random seeds are fixed at 42 across Python, NumPy, PyTorch, CUDA and the HuggingFace `set_seed` utility. This fixes the stratified split, the class weights and the training run. GPU fine-tuning can carry small run-to-run variation, but with seeds fixed and the test partition held out until final evaluation, the reported figures are stable.

## Notes

This project was developed during the INM434 Natural Language Processing module of the MSc Data Science programme at City St George's, University of London. It applies transformer fine-tuning, domain adaptation, class-imbalance handling through weighted loss, NLP preprocessing, and manual error analysis, skills that transfer directly to financial text analytics and risk-signal extraction in financial services.
