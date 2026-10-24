# PA2 requirements and implementation choices

The supplied local files are `Programming assignment NB_Tips.pdf`, the generic `Programming Assignment Rubric.pdf`, and `tweet.zip`. A second workspace search on October 22 did not locate the main sentiment-assignment description. The tips explicitly defer preprocessing requirements to it. This implementation is complete against the choices below; submission compliance remains conditional on checking the full prompt.

| Item | Implemented choice | Evidence or confirmation needed |
|---|---|---|
| Input | Provided train/test folders, ZIP or extracted directory | Supplied archive |
| HTML and emoji | BeautifulSoup visible text, emoji.demojize | Tips |
| Case | Lowercase except all-capital alphabetic tokens longer than one | Tips discuss acronym exception; exact definition absent |
| Punctuation | Keep individual ! and ?, common emoticons; separate other punctuation | Tips discuss repeated punctuation; exact list absent |
| URLs | Remove | Suggested by tips; full prompt may differ |
| Stemming | Compare Porter versus no stemming | Tips allow Porter or Snowball |
| Features | Compare counts and binary occurrence | Both described in tips |
| Learner | Manual multinomial NB; alpha=1, shared training-only vocabulary, log scores | Smoothing/unknown-word rules not specified in available prompt |
| Binary interpretation | Count each word at most once per document at fit and inference | Tips describe occurrence counts; not full Bernoulli NB |
| Metrics | Labeled confusion matrix, per-class precision/recall/F1, accuracy and macro-F1 | Tips specify first five; macro-F1 supports imbalanced data |
| Selection | Training-only stratified exact-text group holdout, seed 626 | Added reproducible evaluation choice |
| Output | Console summary, JSON metrics/model, prediction CSVs, report | Required filenames/CLI/output format not known |
| Identity | Sankalpa Banjade; other team identities not assumed | Confirm PA2 team separately from project team |

The generic rubric includes an exclusive-team-contribution condition, whereas the saved course outline describes AI assistance with disclosure. This work explicitly discloses substantial Codex assistance; confirm the applicable current assignment instructions before treating it as an eligible final submission. No compliance claim is made on the team's behalf.

No claims are made about Canvas submission, deadline, GitHub publication or work completed by teammates.
