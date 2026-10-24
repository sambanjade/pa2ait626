# AIT 626 PA2: Tweet sentiment classification with Naive Bayes

Sankalpa Banjade | Updated October 24, 2026

Manually implemented multinomial Naive Bayes for positive/negative tweet
classification. The experiment compares count and binary-occurrence features,
each with and without Porter stemming. It selects a configuration using only
training-derived validation data, refits on full training data, and evaluates
the instructor's original test split.

## Run

Use Python 3.10 or newer. From this repository folder:

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/audit_dataset.py --archive tweet.zip --output docs/DATA_AUDIT.json
python sentiment_nb.py experiment --data tweet.zip --output results
python sentiment_nb.py predict --model results/model.json --text "Great service!"
```

Obtain the instructor's `tweet.zip` and place it beside this README. It is
excluded from Git and the upload package. An extracted directory containing
`train/{negative,positive}` and `test/{negative,positive}` also works:

```sh
python sentiment_nb.py experiment --data path/to/tweet --output results
```

No NLTK corpus downloads, API keys, or network access are needed after installing
the dependencies. Prediction emits JSON with the original text, predicted label,
and negative/positive log scores. Scores are uncalibrated model scores.
Actual output for the `Great service!` command above with the saved model:

```json
{"text": "Great service!", "label": "positive", "log_scores": {"negative": -18.11125900875921, "positive": -15.412693650536635}}
```

## Algorithm and input/output example

1. Load documents without changing labels or the official train/test split.
2. Keep visible HTML text, remove URLs, convert emoji to names, retain common
   emoticons and individual `!`/`?` tokens, and separate other punctuation.
3. Lowercase words except all-capital alphabetic tokens longer than one; optionally
   apply Porter stemming to non-acronym alphabetic words. Retain negation and stop words.
4. Build a shared vocabulary exclusively from fitting data. Count token frequency
   or one occurrence per document for binary mode.
5. Estimate empirical class priors and add-one smoothed likelihoods; score in log
   space. Ignore unseen words. Empty input uses priors; exact ties select negative.
6. Select by validation macro-F1 using seed 626. Keep identical texts together;
   conflicting-label groups stay entirely in fitting. Refit on all training data.
7. Automatically save metrics, predictions, model, errors and an experiment report.

Hand-calculable example, also verified by the tests:

```python
from sentiment_nb import Config, Document, NaiveBayes
model = NaiveBayes(Config()).fit([
    Document("a", "negative", "bad bad"),
    Document("b", "positive", "good"),
])
print(model.predict("good"))
# positive
```

Here `P(bad|negative)=3/4`, `P(good|positive)=2/3`, and each class prior is `1/2`.
Binary occurrence clips token frequency during both training and prediction;
it does not include absent-word terms from a Bernoulli model.

## Outputs and data quality

See [the experiment report](results/REPORT.md) for measured results, all four
variants, the selected confusion matrix and per-class precision/recall/F1.

- `results/selection.json`: validation results saved before test evaluation.
- `results/metrics.json`: full metrics, baseline, class counts and data-quality counts.
- `results/predictions_0.csv` through `predictions_3.csv`: one row per test document;
  order is count/no stem, count/stem, binary/no stem, binary/stem.
- `results/model.json`: selected model for the prediction command.
- `results/errors.json`: selected-model errors with original text for local review.
- `docs/DATA_AUDIT.json`: archive checksum, counts, duplicates and conflicting labels.

The dataset has 4,181 training and 4,182 test documents. Two training text groups
have contradictory labels. All original examples and labels are retained.
Exact train/test overlaps are disclosed; a separate sensitivity analysis removes
overlapping test texts only from its additional metric. Official evaluation
still uses every supplied test document.

Runtime fields vary by machine. Generated models, prediction CSVs and error text
remain local and are excluded from Git; rerun the experiment to recreate them.
The summary report, metrics and selection are suitable for version control.

## Requirements and attribution

Read [the rubric review](docs/RUBRIC_REVIEW.md) and
[requirements and assumptions](docs/REQUIREMENTS_AND_ASSUMPTIONS.md).
The local NB tips and generic rubric were reviewed. The main assignment
description is absent from this folder, so exact CLI/output/submission requirements,
team identity and deadline remain unverified.

Sources: instructor-provided NB tips and rubric; NLTK PorterStemmer,
Beautiful Soup and emoji library APIs. The Bayes learner is implemented here,
without a library classifier. Codex wrote and revised the implementation, tests,
documentation and experiment runner. The student/team must review and understand
the work and verify applicable course AI and attribution requirements before submission.
