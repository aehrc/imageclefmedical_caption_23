import os
import time
from pathlib import Path

import pandas as pd
import torch
from torchmetrics import Metric


class CaptionLogger(Metric):

    is_differentiable = False
    full_state_update = False

    def __init__(
            self,
            exp_dir: str,
            split: str,
            dist_sync_on_step: bool = False,
    ):
        """
        exp_dir - experiment directory to save the captions and individual scores.
        split - train, val, or test split.
        dist_sync_on_step - sync the workers at each step.
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.exp_dir = exp_dir
        self.split = split

        # No dist_reduce_fx, manually sync over devices
        self.add_state('captions', default=[])

        self.save_dir = os.path.join(self.exp_dir, 'generated_captions')

        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

    def update(self, captions, ids):
        """
        Argument/s:
            captions - the captions section must be in the following format:

                [
                    '...',
                    '...',
                ]
            ids - list of identifiers.
        """

        assert isinstance(captions, list), '"captions" must be a list of strings.'
        assert all(isinstance(i, str) for i in captions), 'Each element of "captions" must be a string.'

        for (i, j) in zip(captions, ids):
            self.captions.append({'captions': i, 'ids': j})

    def compute(self, epoch):
        """
        https://torchmetrics.readthedocs.io/en/stable/pages/implement.html#torchmetrics.Metric.compute
        """

        if torch.distributed.is_initialized():  # If DDP
            captions_gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(captions_gathered, self.captions)
            self.captions = [j for i in captions_gathered for j in i]

        return self.log(epoch)

    def log(self, epoch):

        def save():

            df = pd.DataFrame(self.captions).drop_duplicates(subset='ids')

            df.to_csv(
                os.path.join(self.save_dir, f'{self.split}_epoch-{epoch}_{time.strftime("%d-%m-%Y_%H-%M-%S")}.csv'),
                index=False,
            )

        if not torch.distributed.is_initialized():
            save()
        elif torch.distributed.get_rank() == 0:
            save()
