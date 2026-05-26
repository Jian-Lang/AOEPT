import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from transformers import CLIPModel
from transformers.modeling_attn_mask_utils import (
    _create_4d_causal_attention_mask,
    _prepare_4d_attention_mask,
)

from core.model.Base.base_model import Base


class CLIP(Base):
    def __init__(
        self,
        seq_len: int = 77,
        model_id: str = "openai/clip-vit-base-patch16",
        cls_num: int = 2,
        embed_dim: int = 512,
        pretrain: str = "cls",
        **kargs,
    ):
        super().__init__()
        self.arch = "CLIP"

        # Load pretrained backbone model with resized position embeddings
        self.model = self.get_pretrained_backbone(self.arch, model_id=model_id, seq_len=seq_len)

        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.pretrain = pretrain

        # Initialize classifier (expects concatenated features: embed_dim * 2)
        self.classifier = self.get_classifier(self.arch, cls_num, self.embed_dim)

        # Configure trainable parameters for backbone and classifier
        self.init_base_trainable_para(self.arch, self.model, self.classifier)

        # Feature collection for NMI analysis
        self.cfg = kargs.get("cfg")
        self.statis = self.cfg.get("statis", None) if self.cfg else None
        self.collect_token = self.statis == "collect_token"

        # Token pooling method: "cls", "mean", or "max"
        self.token_pooling = kargs.get("token_pooling", "mean")
        if self.token_pooling not in ["cls", "mean", "max"]:
            raise ValueError(f"token_pooling must be 'cls', 'mean', or 'max', got '{self.token_pooling}'")

        self.token_collector = {}

        self.print_trainable_parameters()

    def print_trainable_parameters(self):
        """Print the number of trainable parameters in the model."""
        trainable_params = 0
        all_params = 0
        for name, param in self.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        logger.info(
            f"Trainable params: {trainable_params:,} || All params: {all_params:,} || Trainable %: {100 * trainable_params / all_params:.2f}%"
        )

    def _get_current_epoch(self) -> int:
        """Return the current epoch if set by the trainer, else 0."""
        return getattr(self, "current_epoch", 0)

    def forward(self, **inputs):
        missing_masks = inputs.pop("missing_masks")
        sample_ids = inputs.pop("ids", None)

        if not self.collect_token:
            # Fast path - use existing implementation
            vision_outputs = self.model.vision_model(pixel_values=inputs["pixel_values"])
            text_outputs = self.model.text_model(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )

            # Get pooled features
            image_embeds = self.model.visual_projection(vision_outputs.pooler_output)
            text_embeds = self.model.text_projection(text_outputs.pooler_output)

            # Combine image and text embeddings by concatenation
            combined_embeds = torch.cat([image_embeds, text_embeds], dim=-1)

            # Classification
            logits = self.classifier(combined_embeds)
            probs = F.softmax(logits, dim=-1)

            return {
                "logits": logits,
                "probs": probs,
            }
        else:
            # Layer-wise path for feature/token collection
            return self._forward_with_collection(sample_ids=sample_ids, missing_masks=missing_masks, **inputs)

    def _forward_with_collection(self, sample_ids, missing_masks, **inputs):
        """Forward pass with layer-wise feature/token collection for CLIP."""
        current_epoch = self._get_current_epoch()
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        pixel_values = inputs["pixel_values"]
        position_ids = inputs.get("position_ids", None)

        # Get embeddings (before encoder layers)
        text_embeds = self.model.text_model.embeddings(input_ids=input_ids, position_ids=position_ids)
        image_embeds = self.model.vision_model.embeddings(pixel_values)
        image_embeds = self.model.vision_model.pre_layrnorm(image_embeds)

        # Get hidden states
        text_hidden_states = text_embeds
        image_hidden_states = image_embeds
        batch_size = text_hidden_states.shape[0]

        # Determine number of layers
        num_text_layers = len(self.model.text_model.encoder.layers)
        num_vision_layers = len(self.model.vision_model.encoder.layers)
        max_layers = max(num_text_layers, num_vision_layers)

        # Interleaved processing
        for i in range(max_layers):
            # Collect tokens before processing layer i
            if sample_ids is not None:
                if self.collect_token:
                    from core.utils.stats_utils import get_clip_token
                    # Apply missing masks: zero out tokens for missing modalities
                    text_tokens_masked = text_hidden_states.clone()
                    vision_tokens_masked = image_hidden_states.clone()

                    # Zero out text tokens where text modality is missing
                    # missing_masks shape: [batch_size, 2], where [:, 0] is text, [:, 1] is vision
                    text_mask = missing_masks[:, 0].float().view(-1, 1, 1)
                    text_tokens_masked = text_tokens_masked * (1 - text_mask)

                    # Zero out vision tokens where vision modality is missing
                    vision_mask = missing_masks[:, 1].float().view(-1, 1, 1)
                    vision_tokens_masked = vision_tokens_masked * (1 - vision_mask)

                    get_clip_token(
                        epoch=current_epoch,
                        layer_idx=i,
                        text_hidden_states=text_tokens_masked,
                        vision_hidden_states=vision_tokens_masked,
                        input_ids=input_ids,
                        eos_token_id=self.model.text_model.config.eos_token_id,
                        sample_ids=sample_ids,
                        token_collector=self.token_collector,
                        attention_mask=attention_mask,
                        pooling_method=self.token_pooling,
                        missing_masks=missing_masks,
                    )

            # Process text encoder layer
            if i < num_text_layers:
                layer_module = self.model.text_model.encoder.layers[i]
                
                # Prepare causal attention mask
                causal_attention_mask = _create_4d_causal_attention_mask(
                    (batch_size, text_hidden_states.shape[1]),
                    text_hidden_states.dtype,
                    device=text_hidden_states.device,
                )
                attention_mask_4d = _prepare_4d_attention_mask(attention_mask, text_hidden_states.dtype)

                layer_outputs = layer_module(
                    text_hidden_states,
                    attention_mask=attention_mask_4d,
                    causal_attention_mask=causal_attention_mask,
                    output_attentions=False,
                )
                text_hidden_states = layer_outputs[0]

            # Process vision encoder layer
            if i < num_vision_layers:
                layer_module = self.model.vision_model.encoder.layers[i]
                
                layer_outputs = layer_module(
                    image_hidden_states,
                    attention_mask=None,
                    causal_attention_mask=None,
                    output_attentions=False,
                )
                image_hidden_states = layer_outputs[0]

        # Apply final normalizations
        text_hidden_states = self.model.text_model.final_layer_norm(text_hidden_states)

        # Text pooling: get features at EOS token position
        text_pooled = text_hidden_states[
            torch.arange(text_hidden_states.shape[0], device=text_hidden_states.device),
            (input_ids.to(dtype=torch.int, device=text_hidden_states.device)
             == self.model.text_model.config.eos_token_id).int().argmax(dim=-1),
        ]

        # Vision pooling: get CLS token (first token)
        image_pooled = image_hidden_states[:, 0, :]
        image_pooled = self.model.vision_model.post_layernorm(image_pooled)

        # Apply projections
        text_pooled = self.model.text_projection(text_pooled)
        image_pooled = self.model.visual_projection(image_pooled)

        # Combine features by concatenation
        combined_embeds = torch.cat([image_pooled, text_pooled], dim=-1)

        # Classification
        logits = self.classifier(combined_embeds)
        probs = F.softmax(logits, dim=-1)

        return {
            "logits": logits,
            "probs": probs,
            "token_collector": self.token_collector if self.collect_token else None,
        }
