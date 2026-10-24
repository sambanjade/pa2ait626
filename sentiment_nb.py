"""AIT 626 PA2: tweet sentiment classification with manually implemented Naive Bayes.

Prepared for Sankalpa Banjade, October 24, 2026, with disclosed Codex assistance.
Input: the provided tweet.zip archive, or an extracted tweet directory containing
train/{negative,positive} and test/{negative,positive} text files.
Output: labeled metrics, per-document predictions, a saved model and an experiment
report. Example: `python sentiment_nb.py experiment --data tweet.zip`.
Prediction: `python sentiment_nb.py predict --model results/model.json --text "Great!"`.
Actual toy example: fitting negative="bad bad", positive="good" with alpha=1
gives P(bad|negative)=3/4, P(good|positive)=2/3; predicting "good" returns
`positive`. The CLI prints JSON containing text, label and both log scores.

Algorithm:
1. Read the instructor's supplied train/test partitions without changing them.
2. Remove HTML and URLs, demojize emoji, preserve sentiment punctuation, normalize
   case except all-capital acronyms, and optionally apply Porter stemming.
3. Count training tokens (or one occurrence per document in binary mode).
4. Estimate empirical class priors and add-alpha smoothed word probabilities
   over ONE shared vocabulary derived exclusively from training examples.
5. Add log prior and log token likelihoods; choose the largest class score.
6. Choose a feature/stemming configuration using training-derived validation,
   refit on full training data, and evaluate the supplied test split.

Sources: local NB_Tips and Programming Assignment Rubric; NLTK PorterStemmer,
BeautifulSoup and emoji public library APIs. The Bayes learner is written here,
not delegated to a library classifier. The full assignment prompt is not local;
docs/REQUIREMENTS_AND_ASSUMPTIONS.md records choices needing reconciliation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
import time
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    import emoji
    from bs4 import BeautifulSoup
    from nltk.stem import PorterStemmer
except ImportError as exc:
    raise SystemExit("Install requirements first: python -m pip install -r requirements.txt") from exc

LABELS = ("negative", "positive")


@dataclass(frozen=True)
class Document:
    identifier: str
    label: str
    text: str


@dataclass(frozen=True)
class Config:
    representation: str = "count"
    stem: bool = False
    alpha: float = 1.0


def tokenize(text: str, stem: bool = False) -> list[str]:
    """One identical preprocessing path for fitting and prediction.

    This treats HTML as markup, but keeps its visible text. Demojized names and
    common emoticons remain features. ! and ? are split individually so repeated
    punctuation contributes repeated evidence in count mode. Other punctuation
    separates words. Negation and stop words are deliberately retained.
    """
    visible = BeautifulSoup(text, "html.parser").get_text(" ")
    visible = re.sub(r"https?://\S+|www\.\S+", " ", visible, flags=re.I)
    visible = emoji.demojize(visible, language="en", delimiters=(" :", ": "))
    tokens = re.findall(r":[a-zA-Z0-9_&+\-]+:|[:;=][\-^]?[)(DPp/]|[\w]+(?:['’][\w]+)*|[!?]", visible)
    porter = PorterStemmer() if stem else None
    result = []
    for token in tokens:
        acronym = len(token) > 1 and token.isalpha() and token.isupper()
        normalized = token if acronym else token.lower()
        # Do not stem acronyms, emoticons, emoji names, or punctuation.
        if porter and normalized.isalpha() and not acronym:
            normalized = porter.stem(normalized)
        result.append(normalized)
    return result


def load_documents(data: Path) -> dict[str, list[Document]]:
    """Load folders or ZIP members directly; never extract untrusted paths."""
    result = {"train": [], "test": []}
    if data.is_file():
        with zipfile.ZipFile(data) as archive:
            members = [(name, archive.read(name)) for name in sorted(archive.namelist())
                       if name.lower().endswith(".txt") and "__MACOSX" not in name]
    elif data.is_dir():
        members = [(p.relative_to(data).as_posix(), p.read_bytes())
                   for p in sorted(data.rglob("*.txt"))]
    else:
        raise ValueError(f"Dataset not found: {data}")
    for name, raw in members:
        parts = Path(name.replace("\\", "/")).parts
        matches = [(split, label) for split in result for label in LABELS
                   if any(parts[i:i+2] == (split, label) for i in range(len(parts)-1))]
        if len(matches) != 1:
            continue
        split, label = matches[0]
        result[split].append(Document(name, label, raw.decode("utf-8-sig")))
    for split, rows in result.items():
        if set(row.label for row in rows) != set(LABELS):
            raise ValueError(f"{split} must contain both positive and negative documents")
    return result


class NaiveBayes:
    """Multinomial NB with count or binary occurrence features.

    Binary mode clips frequency to one in BOTH training and inference. It is not
    Bernoulli NB: absent words do not contribute likelihood terms.
    """

    def __init__(self, config: Config):
        if config.representation not in ("count", "binary"):
            raise ValueError("representation must be count or binary")
        if not math.isfinite(config.alpha) or config.alpha <= 0:
            raise ValueError("alpha must be a finite positive number")
        self.config = config

    def features(self, text: str) -> Counter:
        tokens = tokenize(text, self.config.stem)
        # Sorted binary features keep floating-point summation reproducible
        # across processes with different Python hash seeds.
        return Counter(sorted(set(tokens)) if self.config.representation == "binary" else tokens)

    def fit(self, rows: list[Document]) -> NaiveBayes:
        counts = {label: Counter() for label in LABELS}
        docs = Counter(row.label for row in rows)
        if set(docs) != set(LABELS):
            raise ValueError("Training requires both sentiment classes")
        for row in rows:
            counts[row.label].update(self.features(row.text))
        self.vocabulary = set().union(*(set(c) for c in counts.values()))
        if not self.vocabulary:
            raise ValueError("Training vocabulary is empty after preprocessing")
        self.log_prior = {label: math.log(docs[label] / len(rows)) for label in LABELS}
        self.log_likelihood = {}
        for label in LABELS:
            denominator = sum(counts[label].values()) + self.config.alpha * len(self.vocabulary)
            self.log_likelihood[label] = {
                word: math.log((counts[label][word] + self.config.alpha) / denominator)
                for word in sorted(self.vocabulary)
            }
        return self

    def scores(self, text: str) -> dict[str, float]:
        features = self.features(text)
        return {label: self.log_prior[label] + sum(
            count * self.log_likelihood[label][word]
            for word, count in features.items() if word in self.vocabulary
        ) for label in LABELS}

    def predict(self, text: str) -> str:
        scores = self.scores(text)
        # Fixed LABELS order makes exact ties deterministic: negative first.
        return max(LABELS, key=scores.__getitem__)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"format_version": 1, "config": asdict(self.config),
                                  "log_prior": self.log_prior,
                                  "log_likelihood": self.log_likelihood}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> NaiveBayes:
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("format_version") != 1:
            raise ValueError("Unsupported model format")
        model = cls(Config(**saved["config"]))
        model.log_prior = saved["log_prior"]
        model.log_likelihood = saved["log_likelihood"]
        model.vocabulary = set(model.log_likelihood[LABELS[0]])
        return model


def evaluate(actual: list[str], predicted: list[str]) -> dict:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("Nonempty actual/predicted lists must have equal lengths")
    if (set(actual) | set(predicted)) - set(LABELS):
        raise ValueError("Unexpected class label")
    matrix = [[sum(a == row and p == col for a, p in zip(actual, predicted))
               for col in LABELS] for row in LABELS]
    per_class = {}
    for i, label in enumerate(LABELS):
        tp = matrix[i][i]
        support = sum(matrix[i])
        called = sum(row[i] for row in matrix)
        precision = tp / called if called else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = dict(precision=precision, recall=recall, f1=f1, support=support)
    return dict(labels=list(LABELS), confusion_matrix=matrix,
                matrix_axes="rows=actual, columns=predicted",
                accuracy=sum(matrix[i][i] for i in range(2)) / len(actual),
                macro_f1=sum(v["f1"] for v in per_class.values()) / 2,
                per_class=per_class)


def validation_split(rows: list[Document], seed: int = 626) -> tuple[list[Document], list[Document]]:
    """Hold out ~20% of exact-text groups within each class, using train only.

    A duplicate text must not occur on both sides of this internal split. If
    identical text has conflicting labels, keep the whole group in fitting.
    This retains all supplied labels without leaking that text into validation.
    The instructor's original train/test split remains untouched.
    """
    groups = {}
    for row in rows:
        digest = hashlib.sha256(row.text.encode("utf-8")).hexdigest()
        groups.setdefault(digest, []).append(row)
    rng, fit, validation = random.Random(seed), [], []
    pure_groups = {}
    for key, group in groups.items():
        if len({r.label for r in group}) > 1:
            fit.extend(group)
        else:
            pure_groups[key] = group
    for label in LABELS:
        keys = sorted(k for k, group in pure_groups.items() if group[0].label == label)
        if len(keys) < 2:
            raise ValueError("Need at least two unambiguous distinct training texts per class")
        rng.shuffle(keys)
        cutoff = max(1, min(len(keys)-1, round(0.2 * len(keys))))
        for i, key in enumerate(keys):
            (validation if i < cutoff else fit).extend(groups[key])
    return fit, validation


def write_predictions(path: Path, rows: list[Document], model: NaiveBayes) -> list[str]:
    predictions = []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["document", "actual", "predicted", "log_score_negative", "log_score_positive"])
        for row in rows:
            scores = model.scores(row.text)
            predicted = max(LABELS, key=scores.__getitem__)
            predictions.append(predicted)
            writer.writerow([row.identifier, row.label, predicted, scores["negative"], scores["positive"]])
    return predictions


def experiment(data: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    rows = load_documents(data)
    fit, dev = validation_split(rows["train"])
    configurations = [Config(mode, stem) for mode in ("count", "binary") for stem in (False, True)]
    trials = []
    for config in configurations:
        start = time.perf_counter()
        model = NaiveBayes(config).fit(fit)
        metrics = evaluate([r.label for r in dev], [model.predict(r.text) for r in dev])
        trials.append(dict(config=asdict(config), validation=metrics,
                           validation_seconds=time.perf_counter()-start))
    selected = max(range(len(trials)), key=lambda i: trials[i]["validation"]["macro_f1"])
    # Selection is locked before inspecting any test predictions or test metrics.
    (output / "selection.json").write_text(json.dumps({"selection_rule": "highest validation macro-F1; first listed breaks ties",
        "selected_index": selected, "trials": trials}, indent=2), encoding="utf-8")
    train_hashes = {hashlib.sha256(r.text.encode("utf-8")).hexdigest() for r in rows["train"]}
    overlap = [hashlib.sha256(r.text.encode("utf-8")).hexdigest() in train_hashes for r in rows["test"]]
    training_groups = {}
    for row in rows["train"]:
        training_groups.setdefault(row.text, []).append(row)
    conflicting_groups = [group for group in training_groups.values()
                          if len({r.label for r in group}) > 1]
    for i, (config, trial) in enumerate(zip(configurations, trials)):
        start = time.perf_counter()
        model = NaiveBayes(config).fit(rows["train"])
        predicted = write_predictions(output / f"predictions_{i}.csv", rows["test"], model)
        trial["test"] = evaluate([r.label for r in rows["test"]], predicted)
        unique_rows = [(r, p) for r, p, duplicate in zip(rows["test"], predicted, overlap) if not duplicate]
        trial["nonoverlap_test"] = (evaluate([r.label for r, p in unique_rows], [p for r, p in unique_rows])
                                    if unique_rows else None)
        trial["full_train_test_seconds"] = time.perf_counter()-start
        trial["vocabulary_size"] = len(model.vocabulary)
        if i == selected:
            model.save(output / "model.json")
            # Error excerpts make manual review possible without another run.
            errors = [dict(document=r.identifier, actual=r.label, predicted=p, text=r.text)
                      for r, p in zip(rows["test"], predicted) if r.label != p]
            (output / "errors.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
    counts = {split: dict(Counter(r.label for r in group)) for split, group in rows.items()}
    majority = max(LABELS, key=lambda label: counts["train"][label])
    result = dict(seed=626, data=str(data), counts=counts, fit_documents=len(fit),
                  validation_documents=len(dev), overlap_test_documents=sum(overlap),
                  conflicting_training_groups=len(conflicting_groups),
                  conflicting_training_documents=sum(map(len, conflicting_groups)),
                  selected_index=selected, trials=trials,
                  majority_baseline=evaluate([r.label for r in rows["test"]], [majority]*len(rows["test"])))
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(result, output)
    print(f"Selected on validation: {configurations[selected]}")
    print(json.dumps(trials[selected]["test"], indent=2))
    return result


def write_report(result: dict, output: Path) -> None:
    selected = result["trials"][result["selected_index"]]
    lines = ["# AIT 626 PA2 Naive Bayes experiment report", "",
        "Sankalpa Banjade | October 24, 2026", "",
        "This experiment implements tweet sentiment classification using count and binary-occurrence "
        "Naive Bayes, with and without Porter stemming. All Bayes probabilities and log scores are computed "
        "by the submitted code. This report records actual runs on the supplied archive.", "",
        "## Data and selection", "",
        f"Training class counts: {result['counts']['train']}. Test class counts: {result['counts']['test']}.", "",
        f"A deterministic, class-stratified split of exact-text groups (seed 626) gives "
        f"{result['fit_documents']} fitting documents and {result['validation_documents']} validation documents. "
        "The four configurations use add-one smoothing. Validation macro-F1 selects the configuration; "
        "selection.json is saved before any test evaluation. Each configuration is refit on all training documents.", "",
        f"The training data contains {result['conflicting_training_groups']} identical-text groups "
        f"({result['conflicting_training_documents']} documents) with conflicting labels. These groups "
        "stay entirely in the fitting partition. No labels or supplied documents are removed or changed.", "",
        f"There are {result['overlap_test_documents']} test documents whose exact text occurs in training. "
        "The official split is preserved. A separately labeled sensitivity analysis excludes those test rows; "
        "these are not substitute official scores and do not detect near duplicates.", "",
        "## Preprocessing and algorithm", "",
        "Visible HTML text is retained; tags and URLs are removed. Emoji become English names; common "
        "emoticons and individual exclamation/question marks are retained. Words are lowercased except "
        "all-capital alphabetic tokens of length greater than one, treated as acronym candidates. "
        "Negation and stop words remain. Optional Porter stemming applies to non-acronym alphabetic words.", "",
        "For class c, P(c)=N_c/N and P(w|c)=(count(w,c)+1)/(total_count(c)+|V|), using a "
        "shared training-only vocabulary V. The score is log P(c) plus the sum of token frequencies "
        "times log P(w|c). Binary mode counts each token once in both fitting and prediction; "
        "it is binary-occurrence multinomial NB, not Bernoulli NB. Unknown words are ignored. "
        "Empty or entirely unknown input falls back to priors. Exact ties choose negative.", "",
        "## Measured results", "",
        "| Features | Porter stemming | Validation macro-F1 | Test accuracy | Test macro-F1 | Test macro-F1 without exact overlaps |",
        "|---|---|---:|---:|---:|---:|"]
    for trial in result["trials"]:
        c = trial["config"]
        nonoverlap = trial["nonoverlap_test"]
        nonoverlap_f1 = f"{nonoverlap['macro_f1']:.4f}" if nonoverlap else "N/A (no documents)"
        lines.append(f"| {c['representation']} | {c['stem']} | {trial['validation']['macro_f1']:.4f} | "
                     f"{trial['test']['accuracy']:.4f} | {trial['test']['macro_f1']:.4f} | {nonoverlap_f1} |")
    lines += ["", f"Selected configuration: **{selected['config']}**. "
              f"The always-{max(result['counts']['train'], key=result['counts']['train'].get)} baseline has "
              f"accuracy {result['majority_baseline']['accuracy']:.4f} and macro-F1 {result['majority_baseline']['macro_f1']:.4f}.", "",
              "### Selected model confusion matrix", "", "Rows are actual labels; columns are predictions.", "",
              "| Actual / predicted | Negative | Positive |", "|---|---:|---:|"]
    for label, row in zip(LABELS, selected["test"]["confusion_matrix"]):
        lines.append(f"| {label} | {row[0]} | {row[1]} |")
    lines += ["", "| Class | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"]
    for label, m in selected["test"]["per_class"].items():
        lines.append(f"| {label} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['support']} |")
    lines += ["", "## Interpretation and limitations", "",
        "The selected configuration is the best on this one internal validation split; its test ranking "
        "does not retroactively change the selection. Macro-F1 weights both classes equally, while "
        "accuracy is influenced by the larger negative class. Stemmed and unstemmed variants and "
        "count/binary features may react differently to sparse vocabulary and repeated words. "
        "The conditional-independence assumption cannot model word order, negation scope or sarcasm. "
        "An acronym heuristic can also treat emphatic uppercase words as distinct features. "
        "These limitations and duplicate leakage restrict conclusions beyond the supplied dataset.", "",
        "Per-document predictions, log scores and errors.json are included for inspection. "
        "Scores are Naive Bayes model scores, not calibrated confidence estimates. No test-driven tuning was performed.", "",
        "## Reproduction and requirements", "",
        "Run `python -m pip install -r requirements.txt`, then "
        "`python sentiment_nb.py experiment --data tweet.zip --output results`. "
        "The experiment uses a fixed seed. Runtime measurements vary with hardware.", "",
        "The local NB tips and generic rubric informed this implementation. The main assignment prompt "
        "and exact required invocation/output format still need to be checked before submission. "
        "A runnable implementation alone does not establish complete rubric compliance.", "",
        "## Sources and AI assistance", "",
        "- Instructor-provided Programming assignment NB_Tips.pdf and Programming Assignment Rubric.pdf.",
        "- NLTK PorterStemmer: https://www.nltk.org/api/nltk.stem.porter.html",
        "- emoji demojize API: https://emoji-python.readthedocs.io/en/stable/",
        "- Beautiful Soup: https://www.crummy.com/software/BeautifulSoup/bs4/doc/", "",
        "OpenAI Codex wrote the implementation, tests, experiment runner and report, and executed the experiments. "
        "The student/team must review, explain and verify the work and reconcile the full assignment instructions "
        "and applicable course attribution rules before submission.", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("experiment", help="Run four variants and evaluate")
    run.add_argument("--data", type=Path, default=Path("tweet.zip"))
    run.add_argument("--output", type=Path, default=Path("results"))
    pred = commands.add_parser("predict", help="Classify one tweet using a saved model")
    pred.add_argument("--model", type=Path, required=True)
    pred.add_argument("--text", required=True)
    args = parser.parse_args()
    if args.command == "experiment":
        experiment(args.data, args.output)
    else:
        model = NaiveBayes.load(args.model)
        scores = model.scores(args.text)
        print(json.dumps({"text": args.text, "label": model.predict(args.text), "log_scores": scores}, ensure_ascii=False))


if __name__ == "__main__":
    main()
