"""Hand-calculable probability checks, split integrity and inference edge cases."""
import math
import sys
import tempfile
import unittest
import zipfile
import contextlib
import io
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sentiment_nb import Config, Document, NaiveBayes, evaluate, tokenize, validation_split, experiment, load_documents


class BayesTests(unittest.TestCase):
    def setUp(self):
        self.rows = [Document('a', 'negative', 'bad bad'), Document('b', 'positive', 'good')]

    def test_hand_computed_count_probabilities(self):
        model = NaiveBayes(Config()).fit(self.rows)
        # V={bad,good}; negative counts 2,0 -> 3/4,1/4; positive -> 1/3,2/3.
        self.assertAlmostEqual(math.exp(model.log_likelihood['negative']['bad']), 3/4)
        self.assertAlmostEqual(math.exp(model.log_likelihood['positive']['good']), 2/3)
        self.assertAlmostEqual(model.scores('bad bad')['negative'], math.log(.5)+2*math.log(.75))

    def test_binary_clipping_training_and_prediction(self):
        model = NaiveBayes(Config('binary')).fit(self.rows)
        self.assertAlmostEqual(math.exp(model.log_likelihood['negative']['bad']), 2/3)
        self.assertEqual(model.scores('bad'), model.scores('bad bad bad'))

    def test_unknown_empty_and_serialization(self):
        model = NaiveBayes(Config()).fit(self.rows)
        self.assertEqual(model.predict(''), 'negative')
        self.assertEqual(model.scores('unseen'), model.log_prior)
        self.assertNotIn('unseen', model.vocabulary)
        with tempfile.TemporaryDirectory() as temporary:
            p = Path(temporary)/'model.json'
            model.save(p)
            self.assertEqual(NaiveBayes.load(p).scores('good bad!'), model.scores('good bad!'))

    def test_preprocessing(self):
        self.assertEqual(tokenize('<b>Happy</b>!!! NLP https://x.example 😂'),
                         ['happy','!','!','!','NLP',':face_with_tears_of_joy:'])
        self.assertEqual(tokenize('running not NLP', True), ['run','not','NLP'])

    def test_metrics_against_hand_counts(self):
        m = evaluate(['negative','negative','positive','positive'],
                     ['negative','positive','positive','positive'])
        self.assertEqual(m['confusion_matrix'], [[1,1],[0,2]])
        self.assertAlmostEqual(m['accuracy'], .75)
        self.assertAlmostEqual(m['macro_f1'], ((2/3)+.8)/2)

    def test_duplicate_groups_do_not_cross_validation(self):
        rows = [Document(str(i), label, text) for label in ('negative','positive')
                for i,text in enumerate([label+' a',label+' b',label+' c',label+' a'])]
        fit, validation = validation_split(rows)
        self.assertFalse({r.text for r in fit} & {r.text for r in validation})
        self.assertEqual(len(fit)+len(validation),len(rows))
        self.assertEqual(set(r.label for r in validation),{'negative','positive'})

    def test_reject_bad_smoothing(self):
        for alpha in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                NaiveBayes(Config(alpha=alpha))

    def test_conflicting_label_groups_stay_in_fitting(self):
        rows = [Document(f'{label}{i}', label, f'{label} {i}')
                for label in ('negative', 'positive') for i in range(5)]
        conflicts = [Document('conflict-neg', 'negative', 'same tweet'),
                     Document('conflict-pos', 'positive', 'same tweet')]
        fit, dev = validation_split(rows + conflicts)
        self.assertTrue(all(row in fit for row in conflicts))
        self.assertFalse({r.text for r in fit} & {r.text for r in dev})
        self.assertEqual(sorted(r.identifier for r in fit + dev),
                         sorted(r.identifier for r in rows + conflicts))
        self.assertEqual((fit, dev), validation_split(rows + conflicts))

    def test_experiment_zip_and_folder_with_all_test_texts_overlapping(self):
        # Exercise the real CLI pipeline's data/model/report paths on a tiny
        # archive, including the legitimate empty sensitivity-analysis subset.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / 'tweets.zip'
            with zipfile.ZipFile(archive, 'w') as bundle:
                for label, word in [('negative', 'bad'), ('positive', 'great')]:
                    for i in range(5):
                        bundle.writestr(f'tweet/train/{label}/{i}.txt', f'{word} service {i}')
                    bundle.writestr(f'tweet/test/{label}/0.txt', f'{word} service 0')
            with contextlib.redirect_stdout(io.StringIO()):
                result = experiment(archive, root / 'results')
            self.assertEqual(len(result['trials']), 4)
            self.assertEqual(result['overlap_test_documents'], 2)
            self.assertTrue(all(t['nonoverlap_test'] is None for t in result['trials']))
            self.assertTrue((root / 'results/REPORT.md').is_file())
            self.assertEqual(NaiveBayes.load(root / 'results/model.json').predict('great'), 'positive')
            # Extract only these generated, controlled member paths for loader parity.
            with zipfile.ZipFile(archive) as bundle:
                bundle.extractall(root / 'extracted')
            self.assertEqual(load_documents(archive), load_documents(root / 'extracted'))


if __name__ == '__main__':
    unittest.main()
