# PA2 work plan

## Requirements still needed

- Full Canvas assignment description and deadline.
- Team number, member names and GitHub usernames if collaboration is required.
- Exact preprocessing rules, required model variants, smoothing settings and
  expected run command/output format.
- Required packaging, report format and bonus options.

The tips and generic rubric do not define these details completely.

## Implementation sequence

1. **Data audit:** preserve the provided train/test split, count classes, inspect
   empty files, encodings and duplicate documents. Record any leakage rather than
   silently deleting data or changing the instructor's evaluation split.
2. **Preprocessing:** implement the specified handling of HTML, URLs, emoji,
   punctuation, case/acronyms and optional stemming. Use a single function for both
   training and inference. Record choices and examples.
3. **Features:** derive training-only vocabulary; implement count and binary
   representations if both are required. Distinguish binary word occurrence from
   a full Bernoulli Naive Bayes model, which also scores word absence.
4. **Model:** estimate priors, apply the required smoothing, score in log space,
   define unseen-word and empty-document behavior, and make ties deterministic.
5. **Evaluation:** confusion matrix with labeled axes; class precision/recall/F1,
   accuracy and any required aggregate metrics. Make tuning decisions on a
   training-derived validation split; reserve the supplied test set for evaluation.
6. **Experiments:** compare required representation and stemming variants on
   identical data. Save actual metrics and representative errors.
7. **Documentation:** opening program comment describing the task, actual
   input/output examples, usage, algorithm steps and team identity. Explain
   substantive algorithm choices throughout the code.
8. **Verification and submission:** check a hand-calculable toy example, unknown
   words, empty text and class balance; run the exact required command from a
   clean checkout; package only the required deliverables.

## Rubric mapping

- Opening explanation, examples, usage and stepwise algorithm: 20 points.
- Detailed code comments: 20 points.
- Correct implementation: 40 points.
- Correct automatically produced output: 20 points.
- The rubric lists on-time and extra-functionality bonuses; confirm applicable
  terms in the full assignment.

## Completion criteria

- [x] Repository scaffold and reproducible dataset audit.
- [ ] GitHub remote created and initial commit pushed.
- [ ] Full assignment requirements captured.
- [ ] Preprocessing and model implemented.
- [ ] Required experiments executed and explained.
- [ ] Team review, attribution and final submission package.
