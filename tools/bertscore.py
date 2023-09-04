from bert_score import BERTScorer
from pathlib import Path
from torchmetrics import Metric
# from torchmetrics.text import BERTScore
from transformers import AutoModel, AutoTokenizer
import os
import pandas as pd
import time
import torch


class BERTScore(Metric):
    """
    BERTScore based on: https://www.imageclef.org/2023/medical/caption.
    """

    def __init__(
            self, split, ckpt_dir, mbatch_size, exp_dir, num_workers,
    ):
        """
        Argument/s:
            split - dataset split.
            ckpt_dir - path to the checkpoint directory.
            mbatch_size - mini-batch size for CheXbert.
            exp_dir - experiment directory where outputs will be saved.
            num_workers - the number of workers for BERTScore.
        """
        super().__init__(dist_sync_on_step=False)

        self.split = split
        self.ckpt_dir = ckpt_dir
        self.mbatch_size = mbatch_size
        self.exp_dir = exp_dir
        self.num_workers = num_workers

        self.add_state('captions', default=[])

        self.save_dir = os.path.join(self.exp_dir, 'bertscore')
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def mini_batch(iterable, mbatch_size=1):
        length = len(iterable)
        for i in range(0, length, mbatch_size):
            yield iterable[i:min(i + mbatch_size, length)]

    def update(self, predictions, labels, ids):
        """
        Argument/s:
            predictions - the predictions must be in the following format:

                [
                    '...',
                    '...',
                ]
            labels - the labels must be in the following format:

                [
                    ['...'],
                    ['...'],
                ]
            ids - list of study identifiers.
        """

        assert isinstance(predictions, list), '"predictions" must be a list of strings.'
        assert all(isinstance(i, str) for i in predictions), 'Each element of "predictions" must be a string.'
        assert isinstance(labels, list), '"labels" must be a list of lists, where each sub-list has a multiple strings.'
        assert all(isinstance(i, list) for i in labels), 'Each element of "labels" must be a list of strings.'
        assert all(isinstance(j, str) for i in labels for j in i), 'each sub-list must have one or more strings.'

        for (x, y, z) in zip(predictions, labels, ids):
            self.captions.append({'prediction': x, 'label': y, 'ids': z})

    def compute(self, epoch):

        # BertScore:
        bert_scorer = BERTScorer(
            model_type=os.path.join(self.ckpt_dir, 'microsoft', 'deberta-xlarge-mnli'),
            num_layers=19,
            batch_size=self.mbatch_size,
            nthreads=self.num_workers,
            all_layers=False,
            idf=False,
            lang='en',
            device=self.device,
            rescale_with_baseline=False,
        )

        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        y_hat = [j['prediction'] for j in self.captions]
        y = [j['label'] for j in self.captions]
        ids = [j['ids'] for j in self.captions]

        # Following COCO, the labels are contained in a nested list:
        for j in y:
            assert len(j) == 1
        y = [j[0] for j in y]

        with torch.no_grad():
            bert_scores, hash_code = bert_scorer.score(y_hat, y, batch_size=self.mbatch_size, return_hash=True)
        print(hash_code)

        precision = bert_scores[0].tolist()
        recall = bert_scores[1].tolist()
        f1 = bert_scores[2].tolist()

        rows = []
        for x, s_1, s_2, s_3 in zip(ids, f1, precision, recall):
            rows.append({'ids': x, 'f1': s_1, 'precision': s_2, 'recall': s_3})

        # Gather if DDP
        if torch.distributed.is_initialized():
            rows_gathered = [None] * torch.distributed.get_world_size()

            torch.distributed.all_gather_object(rows_gathered, rows)

            rows = [j for i in rows_gathered for j in i]

        bert_scores = pd.DataFrame(rows)

        # Drop duplicates caused by DDP
        bert_scores = bert_scores.drop_duplicates(subset=['ids'])

        # Save the example and class scores
        def save_scores():
            bert_scores.to_csv(
                os.path.join(
                    self.save_dir,
                    f'{self.split}_epoch-{epoch}_scores_{time.strftime("%d-%m-%Y_%H-%M-%S")}.csv',
                ),
                index=False,
            )

        if not torch.distributed.is_initialized():
            save_scores()
        elif torch.distributed.get_rank() == 0:
            save_scores()

        return {
            'bertscore_f1': bert_scores.f1.mean(),
            'bertscore_precision': bert_scores.precision.mean(),
            'bertscore_recall': bert_scores.recall.mean(),
        }
