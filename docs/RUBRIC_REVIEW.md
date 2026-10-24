# PA2 rubric and instruction review

Reviewed October 24, 2026. This is an evidence checklist, not a predicted grade.

## Available instructions

Read both pages of `Programming assignment NB_Tips.pdf` and the complete
`Programming Assignment Rubric.pdf` in this folder. A search of `ait626/PAs`
found no separate main PA2 assignment description. The tips expressly defer the
full preprocessing requirements to that missing description. The local course
outline is a secondary summary and does not replace the assignment.

## Rubric mapping

| Criterion | Evidence and status |
|---|---|
| Opening comment: task (5 points) | Module docstring describes positive/negative tweet classification and inputs/outputs. |
| Opening comment: actual examples and usage (5 points) | Module docstring has experiment/predict commands and a hand-computed example; README includes runnable input/output. |
| Opening comment: stepwise algorithm and identity (10 points) | Module docstring lists six steps, student name, date and assistance attribution. |
| Detailed comments (20 points) | Docstrings/comments cover preprocessing, smoothing, shared vocabulary, binary occurrence, ties, group validation and test isolation. |
| Correct implementation (40 points) | Manual learner, probability checks, metrics checks, serialization, split-integrity and full-pipeline tests. Full supplied-data run produces all four model variants. |
| Automatically produced correct output (20 points) | Experiment command writes confusion matrices, per-class precision/recall/F1, accuracy, macro-F1, four prediction CSVs, model and report. Exact assignment-specific format remains unverified. |
| On-time bonus (10 points) | Deadline and submission time not known; no claim. |
| Extra-functionality bonus (up to 20 points) | Audit, validation-only selection, overlap sensitivity analysis and model persistence are implemented; eligibility is the instructor's decision. |

The rubric has two preliminary yes/no gates. Runtime is verified for the
documented invocation, but the exact required invocation/output is unavailable.
The other gate asks about exclusive team contributions and restricted sources.
The local course outline describes AI assistance with disclosure. Substantial
Codex assistance is disclosed here; the student/team must reconcile the applicable
instructions with the instructor. This review cannot certify that gate.

## NB tips mapping

| Tip | Implementation |
|---|---|
| Import separate positive/negative documents | ZIP or extracted-folder loader preserves original partitions. |
| BeautifulSoup HTML removal | Keeps visible HTML text, removes markup. |
| emoji.demojize | Emoji become English-name tokens. |
| Punctuation, case/acronyms, links | Retains individual ! and ?, common emoticons; separates other punctuation; removes URLs; retains uppercase acronym candidates. These are documented choices. |
| Stemmer flag | Both Porter stemming and no-stemming variants run. |
| Count and binary vectorizers | Manual frequency counts versus per-document unique-token counts. |
| Priors and word likelihoods | Empirical priors; shared vocabulary; add-one smoothing; log-space scoring. Smoothing is an explicit implementation choice. |
| Prediction | Identical preprocessing at fit/inference; prior fallback for empty or unknown-only text. |
| Confusion matrix and performance | Labeled axes, precision, recall, F1, accuracy; additional macro-F1 and majority baseline. |

The tips' zero default for unseen class words does not specify a complete
smoothing procedure. This implementation uses add-one smoothing to avoid
zero-probability products; verify that choice against the main prompt.

## Data and verification

- Supplied archive: 4,181 train / 4,182 test; 3,000 negative training documents
  and 1,181 positive training documents.
- Two exact training text groups (four documents) have conflicting labels. Both
  labels are retained and each whole group stays in fitting during selection.
- 56 test documents have exact text matches in training. Official metrics use
  all test documents; separately labeled metrics exclude those 56 rows.
- Nine tests cover hand-computed probabilities, binary clipping, unknown/empty
  input, model serialization, preprocessing, metrics, duplicate split integrity,
  conflicting labels, and ZIP/folder full-pipeline behavior with all test texts overlapping.
- See `results/REPORT.md`, `results/metrics.json` and `results/selection.json`
  for actual measured results. Selected model: count features without stemming;
  test accuracy 0.8957436633, macro-F1 0.8617378929. Verified again using the exact
  pinned dependencies: beautifulsoup4 4.14.3, emoji 2.15.0 and NLTK 3.9.2 on
  Python 3.13. Tests, all four experiments and the saved-model predict command passed.

## Upload and review packages

The GitHub upload ZIP contains code, tests, README, audit, review documents and
the summary results. Extract it and upload its contents into the existing
`sambanjade/pa2ait626` repository, retaining the `docs`, `scripts`, `tests` and
`results` folders. Upload the extracted files, rather than just the ZIP itself.

The separate review ZIP additionally includes the saved model, prediction CSVs
and local error excerpts. Both exclude `.git`, caches, environment files,
dependencies, the instructor PDFs and the tweet archive. These packages are
prepared for review; neither is asserted to meet an unseen Canvas packaging rule.

## Remaining student/team checks

1. Obtain/reconcile the full PA2 description and exact required command/output.
2. Confirm team identity, deadline and applicable AI/source rules.
3. Review the code and results; complete the required submission steps on Canvas.
