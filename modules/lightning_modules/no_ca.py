import os
import pathlib
from typing import Optional

import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from lightning.pytorch import LightningModule
from torch.utils.data import DataLoader
from torchvision import transforms

from modules.transformers.medicap.modelling_medicap import (
    CvtWithProjectionHead, CvtWithProjectionHeadConfig,
    MedICapEncoderDecoderModel)
from tools.bertscore import BERTScore
from tools.caption_logger import CaptionLogger
from tools.coco import COCONLGMetricsMIMICCXR
from tools.dataset import ImageCLEFSubset, ImageCLEFTestSet


class NoCrossAttention(LightningModule):
    def __init__(
            self,
            warm_start_modules: bool,
            exp_dir_trial: str,
            dataset_dir: str,
            ckpt_zoo_dir: str,
            mbatch_size: int,
            decoder_max_len: int,
            lr: float,
            num_test_beams: int,
            strategy: Optional[str] = None,
            prefetch_factor: int = 5,
            num_workers: int = 0,
            **kwargs,
    ):
        LightningModule.__init__(self)
        self.save_hyperparameters()

        self.warm_start_modules = warm_start_modules
        self.exp_dir_trial = exp_dir_trial
        self.dataset_dir = dataset_dir
        self.ckpt_zoo_dir = ckpt_zoo_dir
        self.mbatch_size = mbatch_size
        self.decoder_max_len = decoder_max_len
        self.lr = lr
        self.num_test_beams = num_test_beams
        self.strategy = strategy
        self.prefetch_factor = prefetch_factor
        self.num_workers = num_workers

        """
        Evaluation metrics
        
        These need to be defined correctly in order for them to be placed on the correct device:
        https://torchmetrics.readthedocs.io/en/stable/pages/lightning.html#torchmetrics-in-pytorch-lightning
        """
        self.val_metrics, self.test_metrics = [], []

        # COCO NLG metrics:
        self.val_nlg_metrics = COCONLGMetricsMIMICCXR(
            split='val',
            metrics=['bleu', 'cider', 'rouge'],
            exp_dir=self.exp_dir_trial,
            accumulate_over_dicoms=False,
        )
        self.test_nlg_metrics = COCONLGMetricsMIMICCXR(
            split='test',
            metrics=['bleu', 'cider', 'rouge', 'meteor'],
            exp_dir=self.exp_dir_trial,
            accumulate_over_dicoms=False,
        )
        
        # BERTScore:
        self.val_bertscore = BERTScore(
            ckpt_dir=self.ckpt_zoo_dir,
            mbatch_size=self.mbatch_size,
            exp_dir=self.exp_dir_trial,
            split='val',
            num_workers=self.num_workers,
        )
        self.val_bertscore = BERTScore(
            ckpt_dir=self.ckpt_zoo_dir,
            mbatch_size=self.mbatch_size,
            exp_dir=self.exp_dir_trial,
            split='test',
            num_workers=self.num_workers,
        )

        # Caption logging:
        self.val_caption_logger = CaptionLogger(exp_dir=self.exp_dir_trial, split='val_captions')
        self.test_caption_logger = CaptionLogger(exp_dir=self.exp_dir_trial, split='test_captions')

        # Create directory for saliency maps
        self.saliency_map_dir = os.path.join(self.exp_dir_trial, 'saliency_maps')
        pathlib.Path(self.saliency_map_dir).mkdir(parents=True, exist_ok=True)

        # Checkpoint name:
        ckpt_name = 'aehrc/medicap'

        # Tokenizer:
        self.tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(ckpt_name)
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        # Image preprocessing:
        image_processor = transformers.AutoFeatureExtractor.from_pretrained(ckpt_name)

        # Encoder-to-decoder model:
        if self.warm_start_modules:
            config = transformers.GPT2Config.from_pretrained('distilgpt2')
            config.add_cross_attention = False  # No cross attention.
            config.is_decoder = True

            # Do not want this as an attribute of self as it will be an attribute of self.encoder_decoder
            decoder = transformers.GPT2LMHeadModel.from_pretrained('distilgpt2', config=config)

            # Resize GPT2 embedding to include padding and beginning of sentence token:
            decoder.resize_token_embeddings(config.vocab_size + 2)

            config = CvtWithProjectionHeadConfig.from_pretrained(
                'microsoft/cvt-21-384-22k',
                projection_size=768,
            )
            encoder = CvtWithProjectionHead.from_pretrained(
                'microsoft/cvt-21-384-22k',
                config=config,
            )
            self.encoder_decoder = MedICapEncoderDecoderModel(encoder=encoder, decoder=decoder)
        else:
            config = transformers.VisionEncoderDecoderConfig.from_pretrained(ckpt_name)
            self.encoder_decoder = MedICapEncoderDecoderModel(config=config)

        # Image transformations:
        self.train_transforms = transforms.Compose(
            [
                transforms.Resize(size=image_processor.size['shortest_edge']),
                transforms.RandomCrop(
                    size=[
                        image_processor.size['shortest_edge'],
                        image_processor.size['shortest_edge'],
                    ],
                    pad_if_needed=True,
                ),
                transforms.RandomRotation(degrees=5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=image_processor.image_mean,
                    std=image_processor.image_std,
                ),
            ]
        )

        self.test_transforms = transforms.Compose(
            [
                transforms.Resize(size=image_processor.size['shortest_edge']),
                transforms.CenterCrop(size=[
                    image_processor.size['shortest_edge'],
                    image_processor.size['shortest_edge'],
                ]
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=image_processor.image_mean,
                    std=image_processor.image_std,
                ),
            ]
        )

    def setup(self, stage=None):
        """
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#setup
        """
        if stage == 'fit' or stage is None:
            csv_path = os.path.join(
                self.dataset_dir, 
                'imageclef', 
                'imageclefmed_caption_2023', 
                'ImageCLEFmedical_Caption_2023_caption_prediction_train_labels.csv',
            )
            df = pd.read_csv(csv_path, sep='\t')
            images_dir = os.path.join(self.dataset_dir, 'imageclef', 'imageclefmed_caption_2023', 'train')
            self.train_set = ImageCLEFSubset(df=df, transforms=self.train_transforms, images_dir=images_dir)
            print(f'No. of training examples: {self.train_set.__len__()}.')

        if stage == 'fit' or stage == 'validate' or stage is None:
            csv_path = os.path.join(
                self.dataset_dir, 
                'imageclef', 
                'imageclefmed_caption_2023', 
                'ImageCLEFmedical_Caption_2023_caption_prediction_valid_labels.csv',
            )
            df = pd.read_csv(csv_path, sep='\t')
            images_dir = os.path.join(self.dataset_dir, 'imageclef', 'imageclefmed_caption_2023', 'valid')
            self.val_set = ImageCLEFSubset(df=df, transforms=self.test_transforms, images_dir=images_dir)
            print(f'No. of validation examples: {self.val_set.__len__()}.')

        if stage == 'test' or stage is None:
            images_dir = os.path.join(self.dataset_dir, 'imageclef', 'imageclefmed_caption_2023', 'test')
            self.test_set = ImageCLEFTestSet(transforms=self.test_transforms, images_dir=images_dir)
            print('No. of test examples: {}.'.format(self.test_set.__len__()))

    def train_dataloader(self, shuffle=True):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#train-dataloader
        """
        return DataLoader(
            self.train_set,
            batch_size=self.mbatch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            prefetch_factor=self.prefetch_factor,
        )

    def val_dataloader(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#val-dataloader
        """
        return DataLoader(
            self.val_set,
            batch_size=self.mbatch_size,
            num_workers=self.num_workers,
            shuffle=False,
            prefetch_factor=self.prefetch_factor,
        )

    def test_dataloader(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#test-dataloader
        """
        return DataLoader(
            self.test_set,
            batch_size=self.mbatch_size,
            num_workers=self.num_workers,
            shuffle=False,
            prefetch_factor=self.prefetch_factor,
        )

    def configure_optimizers(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers
        """
        optimiser = {'optimizer': torch.optim.AdamW(self.parameters(), lr=self.lr)}
        return optimiser

    def forward(self, images, decoder_input_ids, decoder_attention_mask):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#forward
        """
        # Teacher forcing: labels are given as input
        outputs = self.encoder_decoder(
            pixel_values=images,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            return_dict=True,
        )
        return outputs.logits

    def training_step(self, batch, batch_idx):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#training-step
        """

        # Tokenize captions:
        tokenized = self.encoder_decoder.tokenize_captions_teacher_forcing(batch['captions'], self.tokenizer, self.decoder_max_len)

        # Inference
        y_hat = self(batch['images'], tokenized['decoder_input_ids'], tokenized['decoder_attention_mask'])

        # Add padding to account for non-text positions in prompt:
        tokenized['label_ids'] = F.pad(
            tokenized['label_ids'],
            (y_hat.shape[1] - tokenized['label_ids'].shape[1], 0, 0, 0),
            'constant',
            self.tokenizer.pad_token_id,
        )

        # Loss
        loss = F.cross_entropy(
            y_hat.permute([0, 2, 1]), tokenized['label_ids'], ignore_index=self.tokenizer.pad_token_id,
        )

        # Logging
        self.log_dict({'train_loss': loss}, on_step=True, on_epoch=True, batch_size=batch['images'].size()[0])

        # Update and log scores for each validation metric.
        return loss

    def validation_step(self, batch, batch_idx):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#validation-step
        """

        # Greedy search:
        output_ids = self.encoder_decoder.generate(
            pixel_values=batch['images'],
            max_length=self.decoder_max_len,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            num_beams=1,
            return_dict_in_generate=True,
            use_cache=True,
            no_repeat_ngram_size=3,
        )['sequences']

        # Decode captions:
        generated_captions = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        # Log captions:
        self.val_caption_logger.update(generated_captions, ids=batch['ids'])

        # Evaluate:
        self.val_nlg_metrics.update(generated_captions, [[i] for i in batch['captions']], study_ids=batch['ids'])
        self.val_bertscore.update(generated_captions, [[i] for i in batch['captions']], ids=batch['ids'])

    def on_validation_epoch_end(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#on-validation-epoch-end
        """

        # Save captions:
        self.val_caption_logger.compute(self.current_epoch)
        self.val_caption_logger.reset()

        scores = {}

        output = self.val_nlg_metrics.compute(self.current_epoch)
        scores.update(output)
        self.val_nlg_metrics.reset()

        output = self.val_bertscore.compute(self.current_epoch)
        scores.update(output)
        self.val_bertscore.reset()

        self.log_dict({f'val_{k}': v for k, v in scores.items()}, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#test-step
        """

        # Beam search:
        output_ids = self.encoder_decoder.generate(
            pixel_values=batch['images'],
            max_length=self.decoder_max_len,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            num_beams=self.num_test_beams,
            return_dict_in_generate=True,
            use_cache=True,
            no_repeat_ngram_size=3,
        )['sequences']

        # Decode captions:
        generated_captions = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        # Log captions:
        self.test_caption_logger.update(generated_captions, ids=batch['ids'])

    def on_test_epoch_end(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#on-test-epoch-end
        """
        
        # Save captions:
        self.test_caption_logger.compute(self.current_epoch)
        self.test_caption_logger.reset()