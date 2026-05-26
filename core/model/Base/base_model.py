import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, ViltModel


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


class Base(nn.Module):
    def __init__(self):
        super().__init__()

    def get_pretrained_backbone(
        self,
        model: str,
        model_id: str | None = None,
        seq_len: int | None = None,
    ):
        """
        Get pretrained backbone model.

        Args:
            model: Model architecture name ('CLIP' or 'ViLT')
            model_id: HuggingFace model ID (optional, uses defaults if not provided)
            seq_len: Target sequence length for text (optional, resizes position embeddings if provided)

        Returns:
            Pretrained model with all parameters frozen
        """
        if model == "ViLT":
            if model_id is None:
                model_id = "dandelin/vilt-b32-mlm"  # dandelin/vilt-b32-finetuned-vqa
            backbone = ViltModel.from_pretrained(model_id).requires_grad_(False)
            if seq_len is not None and seq_len != backbone.config.max_position_embeddings:
                self._resize_vilt_position_embeddings(backbone, seq_len)
            return backbone
        elif model == "CLIP":
            if model_id is None:
                model_id = "openai/clip-vit-base-patch16"
            backbone = CLIPModel.from_pretrained(model_id).requires_grad_(False)
            if seq_len is not None and seq_len != backbone.text_model.config.max_position_embeddings:
                self._resize_clip_position_embeddings(backbone, seq_len)
            return backbone
        else:
            raise ValueError(f"Unknown model: {model}")

    def _resize_clip_position_embeddings(self, model, seq_len: int):
        """Resize CLIP text position embeddings to match the target sequence length."""
        text_embed_dim = model.text_model.config.hidden_size
        ori_max_len = model.text_model.config.max_position_embeddings
        ori_embedding_weight = model.text_model.embeddings.position_embedding.weight

        interpolated_weight = (
            F.interpolate(
                ori_embedding_weight.view(1, 1, ori_max_len, text_embed_dim),
                size=(seq_len, text_embed_dim),
                mode="bilinear",
            )
            .squeeze(0)
            .squeeze(0)
        )

        model.text_model.embeddings.position_ids = torch.arange(seq_len).unsqueeze(0)
        model.text_model.embeddings.position_embedding = nn.Embedding.from_pretrained(
            interpolated_weight, freeze=True
        )
        model.text_model.config.max_position_embeddings = seq_len

    def _resize_vilt_position_embeddings(self, model, seq_len: int):
        """Resize ViLT position embeddings to match the target sequence length."""
        embed_dim = model.config.hidden_size
        ori_max_len = model.config.max_position_embeddings
        ori_embedding_weight = model.embeddings.text_embeddings.position_embeddings.weight

        interpolated_weight = (
            F.interpolate(
                ori_embedding_weight.view(1, 1, ori_max_len, embed_dim),
                size=(seq_len, embed_dim),
                mode="bilinear",
            )
            .squeeze(0)
            .squeeze(0)
        )

        model.embeddings.text_embeddings.position_ids = torch.arange(seq_len).unsqueeze(0)
        model.embeddings.text_embeddings.position_embeddings = nn.Embedding.from_pretrained(
            interpolated_weight, freeze=True
        )
        # Also resize token_type_ids buffer to match new sequence length
        model.embeddings.text_embeddings.register_buffer(
            "token_type_ids",
            torch.zeros((1, seq_len), dtype=torch.long),
            persistent=False,
        )
        model.config.max_position_embeddings = seq_len

    def init_base_trainable_para(self, arch: str, model, classifier, pretrain: bool = False):
        """
        Initialize trainable parameters for backbone-specific components and classifier.

        Args:
            arch: Architecture name ('CLIP' or 'ViLT')
            model: The backbone model
            classifier: The classifier module
            pretrain: Unused (kept for signature compatibility)
        """
        # Enable classifier
        for param in classifier.parameters():
            param.requires_grad = True

        # Enable backbone-specific components
        if arch == "CLIP":
            # Enable projection layers
            for param in model.text_projection.parameters():
                param.requires_grad = True
            for param in model.visual_projection.parameters():
                param.requires_grad = True
            # Enable final layer norms
            for param in model.text_model.final_layer_norm.parameters():
                param.requires_grad = True
            for param in model.vision_model.post_layernorm.parameters():
                param.requires_grad = True
        elif arch == "ViLT":
            # Enable pooler
            for param in model.pooler.parameters():
                param.requires_grad = True

    def get_classifier(self, backbone: str, cls_num: int, embed_dim: int = 768):
        """
        Get appropriate classifier for the given backbone.

        Args:
            backbone: Backbone architecture name ('CLIP' or 'ViLT')
            cls_num: Number of classes
            embed_dim: Embedding dimension (default: 768 for ViLT, 512 for CLIP)

        Returns:
            Classifier module
        """
        if backbone == "ViLT":
            # ViLT classifier (MAPs baseline)
            return nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.LayerNorm(embed_dim * 2),
                nn.GELU(),
                nn.Linear(embed_dim * 2, cls_num),
            )
        elif backbone == "CLIP":
            # CLIP classifier (DCP - Deep Correlated Prompting)
            # Uses concatenated features (embed_dim * 2)
            hidden_size = embed_dim * 2
            classifier = nn.Linear(hidden_size, cls_num)
            # Apply init_weights to classifier (following DCP reference)
            classifier.apply(init_weights)
            return classifier
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
