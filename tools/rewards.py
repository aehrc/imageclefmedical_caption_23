import os

import torch
from bert_score import BERTScorer


class BERTScoreReward:

    def __init__(self, ckpt_dir, device, num_workers, mbatch_size):

        self.mbatch_size = mbatch_size

        # BertScore:
        self.bert_scorer = BERTScorer(
            model_type=os.path.join(ckpt_dir, 'microsoft', 'deberta-xlarge-mnli'),
            num_layers=19,
            batch_size=mbatch_size,
            nthreads=num_workers,
            all_layers=False,
            idf=False,
            lang='en',
            device=device,
            rescale_with_baseline=False,
        )
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    def __call__(self, predictions, labels):
        return self.reward(predictions, labels)

    def reward(self, predictions, labels):

        with torch.no_grad() and torch.autocast(device_type='cuda', dtype=torch.float32):

            bert_scores = self.bert_scorer.score(predictions, labels, batch_size=self.mbatch_size)
            f1 = bert_scores[2].tolist()
            f1 = torch.tensor([f1] if isinstance(f1, float) else f1)

        return f1
