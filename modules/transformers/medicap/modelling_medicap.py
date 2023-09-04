import os
from typing import Any, Optional, Tuple, Union

import torch
import transformers
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedTokenizerFast, VisionEncoderDecoderModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.vision_encoder_decoder.configuration_vision_encoder_decoder import \
    VisionEncoderDecoderConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class CvtWithProjectionHeadConfig(transformers.CvtConfig):
    def __init__(self, projection_size: int = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.projection_size = projection_size


class ModelOutputWithProjectionEmbedding(transformers.modeling_outputs.ModelOutput):
    last_hidden_state: torch.FloatTensor


class CvtProjectionHead(torch.nn.Module):

    def __init__(self, config) -> None:
        super().__init__()

        # https://github.com/huggingface/transformers/blob/68287689f2f0d8b7063c400230b3766987abf18d/src/transformers/models/cvt/modeling_cvt.py#L657
        self.layer_norm = torch.nn.LayerNorm(config.embed_dim[-1], eps=config.layer_norm_eps)

        # No bias as following layer normalisation with bias:
        self.projection = torch.nn.Linear(config.embed_dim[-1], config.projection_size, bias=False)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = self.projection(x)
        return x


class CvtWithProjectionHead(transformers.CvtPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.cvt = transformers.CvtModel(config, add_pooling_layer=False)
        self.projection_head = CvtProjectionHead(config)

        # Initialize weights and apply final processing:
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ModelOutputWithProjectionEmbedding]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.cvt(
            pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        projection = self.projection_head(
            torch.permute(torch.flatten(outputs.last_hidden_state, 2), [0, 2, 1]),
        )

        if not return_dict:
            return projection

        return ModelOutputWithProjectionEmbedding(
            last_hidden_state=projection,
        )
    

class MedICapEncoderDecoderModel(VisionEncoderDecoderModel):

    config_class = VisionEncoderDecoderConfig
    base_model_prefix = "vision_encoder_decoder"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def __init__(        
        self,
        config: Optional[PretrainedConfig] = None,
        encoder: Optional[PreTrainedModel] = None,
        decoder: Optional[PreTrainedModel] = None,
    ):

        if decoder:
            assert not decoder.config.add_cross_attention, '"add_cross_attention" must be False for the given decoder'
            assert decoder.config.is_decoder, '"is_decoder" must be True for the given decoder'

        if config is None and (encoder is None or decoder is None):
            raise ValueError("Either a configuration or an encoder and a decoder has to be provided.")
        if config is None:
            config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config)
        else:
            if not isinstance(config, self.config_class):
                raise ValueError(f"Config: {config} has to be of type {self.config_class}")

        config.tie_word_embeddings = False

        # initialize with config
        PreTrainedModel.__init__(self, config)

        # Encoder:
        if encoder is None:
            encoder = CvtWithProjectionHead(config=config.encoder)

        # Decoder:
        if decoder is None:
            decoder = transformers.GPT2LMHeadModel(config=config.decoder)

        self.encoder = encoder
        self.decoder = decoder

        if self.encoder.config.to_dict() != self.config.encoder.to_dict():
            logger.warning(
                f"Config of the encoder: {self.encoder.__class__} is overwritten by shared encoder config:"
                f" {self.config.encoder}"
            )
        if self.decoder.config.to_dict() != self.config.decoder.to_dict():
            logger.warning(
                f"Config of the decoder: {self.decoder.__class__} is overwritten by shared decoder config:"
                f" {self.config.decoder}"
            )
            
        self.encoder.config = self.config.encoder
        self.decoder.config = self.config.decoder

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        if decoder_inputs_embeds is None:
            decoder_inputs_embeds = self.decoder.transformer.wte(decoder_input_ids)

        if encoder_outputs is None:
            if pixel_values is None:
                raise ValueError("You have to specify pixel_values")

            encoder_outputs = self.encoder(
                pixel_values,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )  # CvT does not support output_attentions.
            decoder_inputs_embeds = torch.cat([encoder_outputs[0], decoder_inputs_embeds], dim=1)
            if decoder_attention_mask is not None:
                decoder_attention_mask = torch.cat(
                    [
                        torch.ones(encoder_outputs[0].shape[:-1], dtype=decoder_attention_mask.dtype, device=self.device), 
                        decoder_attention_mask
                    ], 
                    dim=1,
                )            

        decoder_outputs = self.decoder(
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        # Loss:
        loss = None
        if labels is not None:
            logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.reshape(-1))

        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs

        return Seq2SeqLMOutput(
            loss=loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        """
        Modification of: 
            https://github.com/huggingface/transformers/blob/main/src/transformers/models/encoder_decoder/modeling_encoder_decoder.py#L660

        This can help with managing input_embeds and input_ids: 
            https://github.com/huggingface/transformers/issues/6535
        """
        input_dict = {'use_cache': use_cache, 'encoder_outputs': encoder_outputs, 'attention_mask': attention_mask}
    
        if past_key_values is None:
            decoder_inputs = self.decoder.prepare_inputs_for_generation(
                input_ids, inputs_embeds=encoder_outputs[0], past_key_values=past_key_values,
            )
            input_dict['decoder_inputs_embeds'] = decoder_inputs['inputs_embeds']
        else:
            decoder_inputs = self.decoder.prepare_inputs_for_generation(
                input_ids, past_key_values=past_key_values,
            )
            input_dict['decoder_input_ids'] = decoder_inputs['input_ids']
        input_dict['past_key_values'] = decoder_inputs['past_key_values']
        input_dict['decoder_attention_mask'] = decoder_inputs['attention_mask'] if 'attention_mask' in decoder_inputs else None

        return input_dict

    def tokenize_captions_teacher_forcing(
        self, 
        captions: str, 
        tokenizer: PreTrainedTokenizerFast, 
        max_len: int,
    ):
        """
        Tokenizes the captions and creates the inputs and targets for teacher forcing.

        Argument/s:
            captions - the captions.
            tokenizer - Hugging Face tokenizer.
            max_len - maximum number of tokens.

        Returns:
            batch_dict = {
                decoder_input_ids - the token identifiers for the input of the decoder.
                decoder_attention_mask - the attention mask for the decoder_input_ids.
                decoder_token_type_ids - the token type identifiers for the decoder_input_ids.
                label_ids - the label token identifiers for the decoder.
            }
        """

        # Prepare the caption for the tokenizer by placing the special tokens:
        caption = [f'{tokenizer.bos_token}{i}{tokenizer.eos_token}' for i in captions]

        # Tokenize the caption:
        tokenized = tokenizer(
            caption,
            padding='longest',
            truncation=True,
            max_length=max_len + 1,  # +1 to account for the shift between input and target.
            return_tensors='pt',
            return_token_type_ids=False,
            add_special_tokens=False,  # Done in prepare_sections_for_tokenizer()
        ).to(self.device)

        # Modify for language modelling:
        batch_dict = {

            # Labels for the decoder (shifted right by one for autoregression):
            'label_ids': tokenized['input_ids'][:, 1:].detach().clone(),

            # Remove last token identifier to match the sequence length of the labels:
            'decoder_input_ids': tokenized['input_ids'][:, :-1],

            # Attention mask for the decoder_input_ids (remove first token so that the eos_token_id is not considered):
            'decoder_attention_mask': tokenized['attention_mask'][:, 1:],
        }

        return batch_dict