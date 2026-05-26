import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from transformers import ViltModel

from core.model.Base.base_model import Base


class ViLT(Base):
    def __init__(
        self,
        seq_len: int = 40,
        model_id: str = "dandelin/vilt-b32-mlm",
        embed_dim: int = 768,
        cls_num: int = 2,
        pretrain: str = "cls",
        **kargs,
    ):
        super().__init__()
        self.arch = "ViLT"
        self.cfg = kargs.get("cfg")
        self.statis = self.cfg.get("statis", "")
        self.pretrain = pretrain

        # Feature collection for NMI analysis
        self.collect_token = self.statis == "collect_token"

        # Token pooling method: "cls", "mean", or "max"
        self.token_pooling = kargs.get("token_pooling", "mean")
        if self.token_pooling not in ["cls", "mean", "max"]:
            raise ValueError(f"token_pooling must be 'cls', 'mean', or 'max', got '{self.token_pooling}'")

        self.token_collector = {}

        # Load pretrained backbone model with resized position embeddings
        self.model = self.get_pretrained_backbone(self.arch, model_id=model_id, seq_len=seq_len)

        # Store seq_len and embed_dim
        self.target_seq_len = seq_len
        self.embed_dim = embed_dim
        self.text_embedding_length = seq_len

        # Initialize classifier
        self.classifier = self.get_classifier(self.arch, cls_num, self.embed_dim)

        # Configure trainable parameters for backbone and classifier
        self.init_base_trainable_para(self.arch, self.model, self.classifier)

        self.print_trainable_parameters()

    def ensure_position_embeddings_resized(self):
        """Ensure position embeddings match target seq_len after checkpoint loading."""
        if self.target_seq_len != self.model.config.max_position_embeddings:
            self._resize_vilt_position_embeddings(self.model, self.target_seq_len)

    def print_trainable_parameters(self):
        """Print the number of trainable parameters in the model."""
        trainable_params = 0
        all_params = 0
        for name, param in self.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"Trainable params: {trainable_params:,} || All params: {all_params:,} || Trainable %: {100 * trainable_params / all_params:.2f}%"
        )

    def _get_current_epoch(self) -> int:
        """Return the current epoch if set by the trainer, else 0."""
        return getattr(self, "current_epoch", 0)

    def _prepare_vilt_attention_mask(self, attention_mask, dtype, device):
        """Convert 2D padding mask to the broadcastable additive mask ViLT expects."""
        extended_mask = attention_mask[:, None, None, :]
        extended_mask = extended_mask.to(device=device, dtype=dtype)
        extended_mask = (1.0 - extended_mask) * torch.finfo(dtype).min
        return extended_mask

    def embed_vilt(self, **inputs):
        """Embed function for ViLT architecture"""
        output, attention_mask = self.model.embeddings(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
            token_type_ids=inputs.get("token_type_ids"),
            inputs_embeds=None,
            image_embeds=None,
            pixel_values=inputs.get("pixel_values"),
            pixel_mask=inputs.get("pixel_mask"),
            image_token_type_idx=inputs.get("image_token_type_idx", 1),
        )
        return output, attention_mask

    def forward(self, **inputs):
        sample_ids = inputs.pop("ids", None)
        missing_masks = inputs.pop("missing_masks")
        input_ids = inputs.get("input_ids")

        # 1. Embeddings
        embeddings, attention_mask = self.embed_vilt(**inputs)

        # 2. Masking
        # missing_masks: [B, 2] (0: text, 1: vision)
        text_missing = missing_masks[:, 0]
        image_missing = missing_masks[:, 1]
        
        text_len = input_ids.shape[1]

        attention_mask = attention_mask.clone()
        if text_missing.any():
            # Mask text tokens (1 to text_len). Index 0 is CLS (preserved).
            attention_mask[text_missing.bool(), 1:text_len] = 0
        if image_missing.any():
            # Mask image tokens (text_len onwards).
            attention_mask[image_missing.bool(), text_len:] = 0
        
        # Apply mask to embeddings
        embeddings = embeddings * attention_mask.unsqueeze(-1)

        # 3. Prepare extended mask for Encoder
        # This ensures the encoder layers also ignore the missing modalities via self-attention
        attention_mask_expanded = self._prepare_vilt_attention_mask(
            attention_mask, dtype=embeddings.dtype, device=embeddings.device
        )
        
        # 4. Encoder Loop (with optional collection)
        current_epoch = self._get_current_epoch()
        
        # Setup for collection if needed
        combined_token_type_ids = None
        if self.collect_token and sample_ids is not None:
             batch_size, combined_seq_len = embeddings.shape[:2]
             vision_seq_len = combined_seq_len - text_len
             
             tt_ids = inputs.get("token_type_ids")
             if tt_ids is None:
                 tt_ids = torch.zeros((batch_size, text_len), dtype=torch.long, device=embeddings.device)
             else:
                 tt_ids = tt_ids.to(embeddings.device)
                 
             vision_token_types = torch.ones(batch_size, vision_seq_len, dtype=torch.long, device=embeddings.device)
             combined_token_type_ids = torch.cat([tt_ids, vision_token_types], dim=1)

        hidden_states = embeddings
        for i, layer_module in enumerate(self.model.encoder.layer):
            if sample_ids is not None:
                if self.collect_token:
                    from core.utils.stats_utils import get_vilt_token
                    
                    # Note: hidden_states are already masked by input masking above.
                    # We pass them as bases for collection.
                    
                    text_hidden_states = hidden_states.clone()
                    vision_hidden_states = hidden_states.clone()
                    
                    text_mask = missing_masks[:, 0].float().view(-1, 1, 1)
                    text_hidden_states = text_hidden_states * (1 - text_mask)
                    
                    vision_mask = missing_masks[:, 1].float().view(-1, 1, 1)
                    vision_hidden_states = vision_hidden_states * (1 - vision_mask)
                    
                    get_vilt_token(
                        epoch=current_epoch,
                        layer_idx=i,
                        text_hidden_states=text_hidden_states,
                        vision_hidden_states=vision_hidden_states,
                        token_type_ids=combined_token_type_ids,
                        attention_mask=attention_mask,
                        sample_ids=sample_ids,
                        token_collector=self.token_collector,
                        missing_masks=missing_masks,
                        combined_hidden_states=hidden_states,
                        pooling_method=self.token_pooling,
                    )

            # Process layer
            layer_outputs = layer_module(hidden_states, attention_mask=attention_mask_expanded)
            hidden_states = layer_outputs[0]

        # 5. Pool & Classify
        hidden_states = self.model.layernorm(hidden_states)
        pooled_output = self.model.pooler(hidden_states)

        logits = self.classifier(pooled_output)
        probs = F.softmax(logits, dim=-1)

        return {
            "logits": logits,
            "probs": probs,
            "token_collector": self.token_collector if self.collect_token else None,
        }
