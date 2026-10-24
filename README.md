# AIT 626 PA2 — Naive Bayes sentiment classification

Author: Sankalpa Banjade. Team members and assignment deadline: to be confirmed.

This repository starts the tweet sentiment-classification assignment. The supplied
archive separates positive and negative documents into training and test folders.
The classifier and final report are not implemented yet; the full assignment
description is still needed to confirm required preprocessing, model variants,
command-line interface and output format.

## Start here

1. Keep the original `tweet.zip` beside this README (excluded from Git).
2. Run `python scripts/audit_dataset.py --archive tweet.zip`.
3. Read `docs/WORK_PLAN.md` and add the full Canvas assignment requirements.
4. Build preprocessing and the classifier using the course specification.

The audit uses only Python's standard library. It prints document counts, empty
files, and exact byte-for-byte train/test overlaps without extracting the archive.
It does not train a classifier or alter the provided split.

## Planned algorithm

1. Load documents and labels from the provided training folders.
2. Apply the required preprocessing identically to train and test documents.
3. Build the vocabulary and class/token statistics using training data only.
4. Estimate class priors and smoothed word likelihoods.
5. Score test documents in log space and select the highest-scoring class.
6. Produce the required confusion matrix and precision, recall, accuracy and F1.

Count-based and binary-feature variants, stemming, smoothing and handling unknown
words will be finalized against the complete assignment prompt.

## Files and attribution

- `scripts/audit_dataset.py`: dataset inspection utility.
- `docs/WORK_PLAN.md`: implementation and submission checklist.
- `docs/DATA_AUDIT.json`: computed archive inventory.
- Local instructor references: `Programming assignment NB_Tips.pdf` and
  `Programming Assignment Rubric.pdf` (excluded from Git).

AI assistance: Codex helped create the repository structure, dataset audit and
work plan. Team members must review and explain their work and follow current
course attribution requirements. No external solution code has been copied.
