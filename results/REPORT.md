# AIT 626 PA2 Naive Bayes experiment report

Sankalpa Banjade | October 24, 2026

This experiment implements tweet sentiment classification using count and binary-occurrence Naive Bayes, with and without Porter stemming. All Bayes probabilities and log scores are computed by the submitted code. This report records actual runs on the supplied archive.

## Data and selection

Training class counts: {'negative': 3000, 'positive': 1181}. Test class counts: {'negative': 3000, 'positive': 1182}.

A deterministic, class-stratified split of exact-text groups (seed 626) gives 3349 fitting documents and 832 validation documents. The four configurations use add-one smoothing. Validation macro-F1 selects the configuration; selection.json is saved before any test evaluation. Each configuration is refit on all training documents.

The training data contains 2 identical-text groups (4 documents) with conflicting labels. These groups stay entirely in the fitting partition. No labels or supplied documents are removed or changed.

There are 56 test documents whose exact text occurs in training. The official split is preserved. A separately labeled sensitivity analysis excludes those test rows; these are not substitute official scores and do not detect near duplicates.

## Preprocessing and algorithm

Visible HTML text is retained; tags and URLs are removed. Emoji become English names; common emoticons and individual exclamation/question marks are retained. Words are lowercased except all-capital alphabetic tokens of length greater than one, treated as acronym candidates. Negation and stop words remain. Optional Porter stemming applies to non-acronym alphabetic words.

For class c, P(c)=N_c/N and P(w|c)=(count(w,c)+1)/(total_count(c)+|V|), using a shared training-only vocabulary V. The score is log P(c) plus the sum of token frequencies times log P(w|c). Binary mode counts each token once in both fitting and prediction; it is binary-occurrence multinomial NB, not Bernoulli NB. Unknown words are ignored. Empty or entirely unknown input falls back to priors. Exact ties choose negative.

## Measured results

| Features | Porter stemming | Validation macro-F1 | Test accuracy | Test macro-F1 | Test macro-F1 without exact overlaps |
|---|---|---:|---:|---:|---:|
| count | False | 0.8705 | 0.8957 | 0.8617 | 0.8584 |
| count | True | 0.8659 | 0.8989 | 0.8664 | 0.8632 |
| binary | False | 0.8664 | 0.8972 | 0.8623 | 0.8590 |
| binary | True | 0.8650 | 0.8977 | 0.8632 | 0.8599 |

Selected configuration: **{'representation': 'count', 'stem': False, 'alpha': 1.0}**. The always-negative baseline has accuracy 0.7174 and macro-F1 0.4177.

### Selected model confusion matrix

Rows are actual labels; columns are predictions.

| Actual / predicted | Negative | Positive |
|---|---:|---:|
| negative | 2910 | 90 |
| positive | 346 | 836 |

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| negative | 0.8937 | 0.9700 | 0.9303 | 3000 |
| positive | 0.9028 | 0.7073 | 0.7932 | 1182 |

## Interpretation and limitations

The selected configuration is the best on this one internal validation split; its test ranking does not retroactively change the selection. Macro-F1 weights both classes equally, while accuracy is influenced by the larger negative class. Stemmed and unstemmed variants and count/binary features may react differently to sparse vocabulary and repeated words. The conditional-independence assumption cannot model word order, negation scope or sarcasm. An acronym heuristic can also treat emphatic uppercase words as distinct features. These limitations and duplicate leakage restrict conclusions beyond the supplied dataset.

Per-document predictions, log scores and errors.json are included for inspection. Scores are Naive Bayes model scores, not calibrated confidence estimates. No test-driven tuning was performed.

## Reproduction and requirements

Run `python -m pip install -r requirements.txt`, then `python sentiment_nb.py experiment --data tweet.zip --output results`. The experiment uses a fixed seed. Runtime measurements vary with hardware.

The local NB tips and generic rubric informed this implementation. The main assignment prompt and exact required invocation/output format still need to be checked before submission. A runnable implementation alone does not establish complete rubric compliance.

## Sources and AI assistance

- Instructor-provided Programming assignment NB_Tips.pdf and Programming Assignment Rubric.pdf.
- NLTK PorterStemmer: https://www.nltk.org/api/nltk.stem.porter.html
- emoji demojize API: https://emoji-python.readthedocs.io/en/stable/
- Beautiful Soup: https://www.crummy.com/software/BeautifulSoup/bs4/doc/

OpenAI Codex wrote the implementation, tests, experiment runner and report, and executed the experiments. The student/team must review, explain and verify the work and reconcile the full assignment instructions and applicable course attribution rules before submission.
