import torch
import transformers

from modules.lightning_modules.no_ca import NoCrossAttention
from tools.rewards import BERTScoreReward


class SCSTNoCrossAttentionBERTScore(NoCrossAttention):

    def __init__(
        self,
        trial,
        scst_sample_top_p: float = 1.0,
        scst_sample_top_k: float = 50,
        scst_sample_temperature: float = 1.0,
        **kwargs,
    ):
        """
        Argument/s:
            trial - trial number for the model.
            scst_sample_top_p - only the most probable tokens with probabilities that add up to top_p or higher are
                considered during sampling.
            scst_sample_top_k - only the top-k ranked tokens are considered during sampling.
            scst_sample_temperature - the sharpness of the softmax probability distribution during sampling.
            kwargs - keyword arguments.
        """
        super(SCSTNoCrossAttentionBERTScore, self).__init__(**kwargs)

        self.trial = trial
        self.scst_sample_top_p = scst_sample_top_p
        self.scst_sample_top_k = scst_sample_top_k
        self.scst_sample_temperature = scst_sample_temperature

    def on_fit_start(self):
        """
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-fit-start.
        """

        # Reward:
        self.reward = BERTScoreReward(self.ckpt_zoo_dir, self.device, self.num_workers, self.mbatch_size)

    def training_step(self, batch, batch_idx):
        """
        Training step (the training loss needs to be returned).

        Argument/s:
            batch - mini-batch from the training set DataLoader.
            batch_idx - batch idx of each example in the mini-batch.

        Returns:
            loss - training loss for the mini-batch.
        """

        # SCST step
        loss = self.scst_step(batch, batch_idx)

        # Logging
        self.log_dict({'scst_loss': loss}, on_step=True, on_epoch=True, batch_size=batch['images'].size()[0])

        # Update and log scores for each validation metric.
        return loss

    def scst_step(self, batch, batch_idx):
        """
        Self-critical sequence training step (the training loss needs to be returned).

        Argument/s:
            batch - mini-batch from the training set DataLoader.
            batch_idx - batch idx of each example in the mini-batch.

        Returns:
            loss - training loss for the mini-batch.
        """

        # Encoder outputs
        encoder_outputs = self.encoder_decoder.encoder(batch['images'])

        # Samples
        logits, sampled_token_ids, sample_str = self.sample(encoder_outputs)

        # Sample reward
        reward = self.reward(sample_str, batch['captions']).to(self.device)  # batch contains the labels.

        # Baseline reward
        baseline_ids = self.encoder_decoder.generate(
            encoder_outputs=encoder_outputs,
            max_length=self.decoder_max_len,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            num_beams=1,
            return_dict_in_generate=True,
            use_cache=True,
        )['sequences']

        # Baseline string:
        baseline_str = self.tokenizer.batch_decode(baseline_ids, skip_special_tokens=True)

        # Baseline reward:
        baseline = self.reward(baseline_str, batch['captions']).to(self.device)  # batch contains the labels.
        reward = reward - baseline

        # Loss
        loss = self.reinforce_loss(logits, sampled_token_ids, reward)

        # Update and log scores for each metric
        self.log_dict(
            {'reward': torch.mean(reward), 'baseline': torch.mean(baseline)},
            on_step=True,
            on_epoch=True,
            batch_size=batch['images'].size()[0],
        )

        return loss

    def sample(self, encoder_outputs):
        """
        Generate the sample caption for SCST.

        Argument/s:
            encoder_outputs - outputs from the encoder.

        Returns:
            logits - logits from the output of the language model head.
            sampled_token_ids - sampled token indices.
            sample_str - the sampled captions.
        """

        # Logits warper:
        logits_warper = transformers.LogitsProcessorList(
            [
                transformers.TemperatureLogitsWarper(self.scst_sample_temperature),
                transformers.TopPLogitsWarper(self.scst_sample_top_p),
                transformers.TopKLogitsWarper(self.scst_sample_top_k),
            ]
        )

        # Stopping criteria:
        stopping_criteria = transformers.StoppingCriteriaList([
            transformers.MaxLengthCriteria(max_length=self.decoder_max_len)],
        )

        # Sample
        bos_ids = torch.ones(
            (encoder_outputs[0].size()[0], 1), dtype=torch.long, device=self.device
        ) * self.tokenizer.bos_token_id

        sample = self.encoder_decoder.sample(
            input_ids=bos_ids,
            encoder_outputs=encoder_outputs,
            logits_warper=logits_warper,
            stopping_criteria=stopping_criteria,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            mask_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            do_sample=True,
            use_cache=True,
            output_scores=True,
        )

        # Logits:
        logits = torch.stack(sample['scores'], dim=-1)

        # Sample string:
        sample_str = self.tokenizer.batch_decode(sample['sequences'], skip_special_tokens=True)

        # Sampled token IDs:
        sampled_token_ids = sample['sequences'][:, 1:]

        # Sequence length:
        mask = sampled_token_ids == self.tokenizer.pad_token_id
        seq_len = torch.sum(torch.logical_not(mask), dim=-1).float()

        # Log sequence length:
        self.log_dict({'seq_len': torch.mean(seq_len)}, on_step=True, on_epoch=True, batch_size=seq_len.size()[0])

        return logits, sampled_token_ids, sample_str

    def reinforce_loss(self, logits: torch.Tensor, sampled_token_ids: torch.Tensor,
                       reward: torch.Tensor) -> torch.Tensor:
        """
        Loss for the REINFORCE algorith from https://doi.org/10.1007/BF00992696. It is detailed for
        gradient descent in https://doi.org/10.1109/cvpr.2017.131.

        PyTorch implementation:
            https://pytorch.org/docs/stable/distributions.html#score-function

        Argument/s
            logits - logits from the language model head.
            sampled_token_ids - sampled token indices.
            reward - reward for each batch element.

        Returns:
            REINFORCE loss for gradient descent.
        """

        # Negative log-likelihood of each sampled token
        loss = torch.nn.functional.nll_loss(
            input=torch.nn.functional.log_softmax(logits, dim=1),
            target=sampled_token_ids,
            ignore_index=self.tokenizer.pad_token_id,
            reduction='none',
        )

        # Negative sequence log-likelihood
        loss = loss.sum(dim=-1)

        # Reward
        loss = loss * reward

        # Mean over mini-batch elements
        loss = loss.mean()

        return loss
