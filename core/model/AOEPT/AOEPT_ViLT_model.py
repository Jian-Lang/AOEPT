import copy
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from transformers import ViltModel

from core.model.Base.base_model import Base
from core.model.AOEPT.module import AdaptivePromptGate


class AOEPT(Base):
    """
    AOEPT: Unified Multimodal Prompting with Adaptive Gating for ViLT

    Adapts the AOEPT architecture (Deep Prompting + Cross-Modality Gating)
    to the single-stream ViLT backbone.

    Distinct Text and Vision prompts are maintained and injected (prepended)
    to the shared sequence. Gating is driven by extracting "Text" and "Vision"
    features from the shared sequence (based on positional slicing).
    """

    def __init__(
        self,
        cls_num: int,
        init_from_token: str | None = None,
        prompt_strategy: str = "attention",  # 'attention', 'init', 'mlp'
        N: int = 32,  # Init token length
        L: int = 16,  # Prompt length (L < N)
        seq_len: int = 40,  # Max text length (matches MAPs)
        prompt_depth: int = 6,  # Number of layers to prompt
        attn_num_heads: int = 4,  # Only used when prompt_strategy == 'attention'
        reduction_ratio: int = 8,  # Bottleneck ratio for gating MLP
        loss_alpha: float = 0.05,  # Weight of auxiliary loss
        **kargs,
    ):
        super().__init__()
        self.arch = "ViLT"
        self.cfg = kargs.get("cfg", {})

        # AOEPT Settings
        self.prompt_strategy = prompt_strategy
        self.N = N
        self.L = L
        self.prompt_depth = prompt_depth
        self.attn_num_heads = attn_num_heads
        self.reduction_ratio = reduction_ratio
        self.init_from_token = init_from_token
        self.loss_alpha = loss_alpha
        self.seq_len = seq_len

        # Shared Prompt Settings
        self.L_shared = kargs.get("L_shared", self.L)
        self.use_shared_prompt = kargs.get("use_shared_prompt", True)

        # Load backbone
        # ViLT has a single hidden_size (usually 768)
        self.model = self.get_pretrained_backbone(self.arch, seq_len=seq_len)
        self.embed_dim = self.model.config.hidden_size  # 768
        self.vision_embed_dim = self.embed_dim  # Same for ViLT

        # Initialize classifier
        self.classifier = self.get_classifier(self.arch, cls_num, self.embed_dim)

        self.name = "AOEPT"

        # 1. Init Layer-wise and Modality-wise Tokens (Length N)
        self._init_tokens()

        # 2. Implement Prompt Strategy
        self._init_prompt_strategy()

        # 3. Gating Mechanism
        self._init_gating()

        # 4. Contrastive Learning Settings
        self.logit_scale = nn.Parameter(torch.ones([]) * 4.6052)
        # ViLT doesn't have separate projections, so we create small ones for contrastive loss
        # matching AOEPT_CLIP dimension logic (project to common space)
        # Or we can just use identity if dims are same, but projections help alignment.

        # Initialize trainable parameters
        self.init_trainable_para()

    def _init_tokens(self):
        """
        Load tokens from file and pool to length N.
        Stores:
            self.text_init_tokens: [prompt_depth, N, D]
            self.vision_init_tokens: [prompt_depth, N, D]
        """
        if self.init_from_token is None:
            logger.warning("No init_from_token provided. initializing random N tokens.")
            self.text_init_tokens = nn.Parameter(
                torch.randn(self.prompt_depth, self.N, self.embed_dim) * 0.02
            )
            self.vision_init_tokens = nn.Parameter(
                torch.randn(self.prompt_depth, self.N, self.vision_embed_dim) * 0.02
            )
            return

        # Load token file
        data = torch.load(self.init_from_token, map_location=self.model.device)

        # Text
        if "text_token" in data:
            text_token = data["text_token"]  # [Samples, Layers, D]
            text_non_zero = text_token.abs().sum(dim=(1, 2)) > 0
            text_token = text_token[text_non_zero]

            # Shuffle along samples
            idx = torch.randperm(text_token.shape[0])
            text_token = text_token[idx]

            if "cluster" in self.init_from_token:
                self.N = text_token.shape[0]
                text_pooled = text_token.float().permute(1, 0, 2)
            else:
                # Pool to length N
                text_pooled = F.adaptive_avg_pool1d(
                    text_token.float().permute(1, 2, 0), output_size=self.N
                ).permute(0, 2, 1)

            self.text_init_tokens = nn.Parameter(text_pooled[: self.prompt_depth].clone().detach())
        else:
            logger.warning("text_token not found in init file. Initializing random text tokens (missing_rate=1.0?).")
            self.text_init_tokens = nn.Parameter(
                torch.randn(self.prompt_depth, self.N, self.embed_dim) * 0.02
            )

        # Vision
        if "vision_token" in data:
            vision_token = data["vision_token"]
            vision_non_zero = vision_token.abs().sum(dim=(1, 2)) > 0
            vision_token = vision_token[vision_non_zero]

            # Shuffle along samples
            idx = torch.randperm(vision_token.shape[0])
            vision_token = vision_token[idx]

            if "cluster" in self.init_from_token:
                vision_pooled = vision_token.float().permute(1, 0, 2)
            else:
                vision_pooled = F.adaptive_avg_pool1d(
                    vision_token.float().permute(1, 2, 0), output_size=self.N
                ).permute(0, 2, 1)

            self.vision_init_tokens = nn.Parameter(vision_pooled[: self.prompt_depth].clone().detach())
        else:
            logger.warning("vision_token not found in init file. Initializing random vision tokens (missing_rate=1.0?).")
            self.vision_init_tokens = nn.Parameter(
                torch.randn(self.prompt_depth, self.N, self.vision_embed_dim) * 0.02
            )

    def _init_prompt_strategy(self):
        """
        Initialize parameters based on prompt strategy.
        Strategies: 'attention', 'init', 'mlp'
        """
        if self.prompt_strategy == "attention":
            # Text
            self.text_query = nn.Parameter(torch.randn(self.prompt_depth, self.L, self.embed_dim) * 0.02)
            # Vision
            self.vision_query = nn.Parameter(
                torch.randn(self.prompt_depth, self.L, self.vision_embed_dim) * 0.02
            )

            self.text_layer_norm = nn.LayerNorm(self.embed_dim)
            self.vision_layer_norm = nn.LayerNorm(self.vision_embed_dim)

            # Shared Query Parameters (only for attention strategy)
            if self.use_shared_prompt:
                self.shared_query = nn.Parameter(
                    torch.randn(self.prompt_depth, self.L_shared, self.embed_dim) * 0.02
                )
                self.shared_layer_norm = nn.LayerNorm(self.embed_dim)

        elif self.prompt_strategy == "init":
            # Text
            text_pooled = F.adaptive_avg_pool1d(
                self.text_init_tokens.permute(0, 2, 1), output_size=self.L
            ).permute(0, 2, 1)
            self.text_proxy_prompts = nn.Parameter(text_pooled.clone().detach())

            # Vision
            vision_pooled = F.adaptive_avg_pool1d(
                self.vision_init_tokens.permute(0, 2, 1), output_size=self.L
            ).permute(0, 2, 1)
            self.vision_proxy_prompts = nn.Parameter(vision_pooled.clone().detach())

        elif self.prompt_strategy == "mlp":
            # Text
            self.text_mlp_input = nn.Parameter(self.text_init_tokens.clone().detach())

            text_hidden_dim = self.embed_dim // self.reduction_ratio
            self.text_mlp = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.embed_dim, text_hidden_dim),
                        nn.GELU(),
                        nn.Linear(text_hidden_dim, self.embed_dim),
                    )
                    for _ in range(self.prompt_depth)
                ]
            )

            # Vision
            self.vision_mlp_input = nn.Parameter(self.vision_init_tokens.clone().detach())

            vision_hidden_dim = self.vision_embed_dim // self.reduction_ratio
            self.vision_mlp = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.vision_embed_dim, vision_hidden_dim),
                        nn.ReLU(),
                        nn.Linear(vision_hidden_dim, self.vision_embed_dim),
                    )
                    for _ in range(self.prompt_depth)
                ]
            )

        else:
            raise ValueError(f"Unknown prompt strategy: {self.prompt_strategy}")

    @staticmethod
    def _param_free_dot_attn(q: torch.Tensor, k: torch.Tensor, num_heads: int) -> torch.Tensor:
        batch_size, q_len, embed_dim = q.shape
        _, k_len, _ = k.shape
        head_dim = embed_dim // num_heads

        q = q.view(batch_size, q_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, k_len, num_heads, head_dim).transpose(1, 2)
        v = k

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        attn = attn_logits.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, q_len, embed_dim)
        return out

    def _init_gating(self):
        """
        Instance-aware gating with zero-initialization.
        Text Gated by Vision features.
        Vision Gated by Text features.
        """
        # Text Gating (Using Vision Features)
        self.text_gate_mlp = AdaptivePromptGate(
            input_dim=self.vision_embed_dim,
            hidden_dim=self.embed_dim,
            reduction_ratio=self.reduction_ratio,
        )

        # Vision Gating (Using Text Features)
        self.vision_gate_mlp = AdaptivePromptGate(
            input_dim=self.embed_dim,
            hidden_dim=self.vision_embed_dim,
            reduction_ratio=self.reduction_ratio,
        )

        # Shared Prompt Gating (Using Fusion of Text + Vision Features)
        if self.use_shared_prompt:
            self.shared_gate_mlp = AdaptivePromptGate(
                input_dim=self.embed_dim,  # Fusion features dimension
                hidden_dim=self.embed_dim,
                reduction_ratio=self.reduction_ratio,
            )

    def init_trainable_para(self):
        # Freeze backbone
        for param in self.model.parameters():
            param.requires_grad = False

        # Enable Strategy Parameters
        if self.prompt_strategy == "attention":
            self.text_query.requires_grad = True
            self.vision_query.requires_grad = True
            self.text_init_tokens.requires_grad = False
            self.vision_init_tokens.requires_grad = False
            for p in self.text_layer_norm.parameters():
                p.requires_grad = True
            for p in self.vision_layer_norm.parameters():
                p.requires_grad = True

            # NEW: Shared query
            if self.use_shared_prompt:
                self.shared_query.requires_grad = True
                for p in self.shared_layer_norm.parameters():
                    p.requires_grad = True

        elif self.prompt_strategy == "init":
            self.text_proxy_prompts.requires_grad = True
            self.vision_proxy_prompts.requires_grad = True
            self.text_init_tokens.requires_grad = False
            self.vision_init_tokens.requires_grad = False

        elif self.prompt_strategy == "mlp":
            self.text_mlp_input.requires_grad = False
            self.vision_mlp_input.requires_grad = False
            for p in self.text_mlp.parameters():
                p.requires_grad = True
            for p in self.vision_mlp.parameters():
                p.requires_grad = True

        # Enable Gating
        for p in self.text_gate_mlp.parameters():
            p.requires_grad = True
        for p in self.vision_gate_mlp.parameters():
            p.requires_grad = True

        # NEW: Shared gating
        if self.use_shared_prompt:
            for p in self.shared_gate_mlp.parameters():
                p.requires_grad = True

        # Enable Contrastive
        if hasattr(self, "logit_scale"):
            self.logit_scale.requires_grad = True

        # Enable Classifier & Base
        self.init_base_trainable_para(self.arch, self.model, self.classifier)

    def _get_proxy_prompts(self, batch_size):
        if self.prompt_strategy == "attention":
            # Text
            text_prompts = []
            for i in range(self.prompt_depth):
                q = self.text_query[i].unsqueeze(0).expand(batch_size, -1, -1)
                k = self.text_init_tokens[i].unsqueeze(0).expand(batch_size, -1, -1)
                out = self._param_free_dot_attn(q, k, num_heads=self.attn_num_heads)
                out = self.text_layer_norm(out + q)
                text_prompts.append(out)
            text_prompts = torch.stack(text_prompts)

            # Vision
            vision_prompts = []
            for i in range(self.prompt_depth):
                q = self.vision_query[i].unsqueeze(0).expand(batch_size, -1, -1)
                k = self.vision_init_tokens[i].unsqueeze(0).expand(batch_size, -1, -1)
                out = self._param_free_dot_attn(q, k, num_heads=self.attn_num_heads)
                out = self.vision_layer_norm(out + q)
                vision_prompts.append(out)
            vision_prompts = torch.stack(vision_prompts)

            # Shared Prompts (attend over joint text+vision memory)
            if self.use_shared_prompt:
                shared_prompts = []
                for i in range(self.prompt_depth):
                    q = self.shared_query[i].unsqueeze(0).expand(batch_size, -1, -1)  # [B, L_shared, D]

                    # Concatenate text and vision memory banks
                    text_memory = (
                        self.text_init_tokens[i].unsqueeze(0).expand(batch_size, -1, -1)
                    )  # [B, N, D]
                    vision_memory = (
                        self.vision_init_tokens[i].unsqueeze(0).expand(batch_size, -1, -1)
                    )  # [B, N, D]
                    joint_memory = torch.cat([text_memory, vision_memory], dim=1)  # [B, 2N, D]

                    # Attend over joint memory
                    out = self._param_free_dot_attn(q, joint_memory, num_heads=self.attn_num_heads)
                    out = self.shared_layer_norm(out + q)  # Residual + LayerNorm
                    shared_prompts.append(out)

                shared_prompts = torch.stack(shared_prompts)  # [prompt_depth, B, L_shared, D]

        elif self.prompt_strategy == "init":
            text_prompts = self.text_proxy_prompts.unsqueeze(1).expand(-1, batch_size, -1, -1)
            vision_prompts = self.vision_proxy_prompts.unsqueeze(1).expand(-1, batch_size, -1, -1)

        elif self.prompt_strategy == "mlp":
            text_prompts = []
            for i in range(self.prompt_depth):
                inp = self.text_mlp_input[i].unsqueeze(0).expand(batch_size, -1, -1)
                out = self.text_mlp[i](inp)
                # Pool N -> L
                out = F.adaptive_avg_pool1d(out.permute(0, 2, 1), output_size=self.L).permute(0, 2, 1)
                text_prompts.append(out)
            text_prompts = torch.stack(text_prompts)

            vision_prompts = []
            for i in range(self.prompt_depth):
                inp = self.vision_mlp_input[i].unsqueeze(0).expand(batch_size, -1, -1)
                out = self.vision_mlp[i](inp)
                # Pool N -> L
                out = F.adaptive_avg_pool1d(out.permute(0, 2, 1), output_size=self.L).permute(0, 2, 1)
                vision_prompts.append(out)
            vision_prompts = torch.stack(vision_prompts)

        if self.use_shared_prompt and self.prompt_strategy == "attention":
            return text_prompts, vision_prompts, shared_prompts
        else:
            return text_prompts, vision_prompts, None

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
        # Extract Inputs
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        missing_masks = inputs.get("missing_masks")  # [B, 2] (True if missing)

        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Mask for Availability
        text_missing = missing_masks[:, 0]
        image_missing = missing_masks[:, 1]

        # 1. Generate Proxy Prompts
        # [Layers, B, L, D]
        text_proxy, vision_proxy, shared_proxy = self._get_proxy_prompts(batch_size)

        # 2. Initial Embeddings
        # output: [B, SeqLen, D]
        output, attention_mask = self.embed_vilt(**inputs)

        # Identify Split Point
        # ViLT Layout: [CLS] + Text (len=T) + [SEP] + Image (len=V)
        # input_ids has length T (including padding).
        # We need to verify if embeddings include CLS/SEP in input_ids or adds them.
        # Transformers ViltEmbeddings adds CLS/SEP/Patch projections.
        # input_ids usually is raw text ids.
        # Assuming input_ids.shape[1] corresponds to text part length effectively.
        # Actually ViltModel.embeddings output shape is input_ids.shape[1] + image_patches + 1(CLS)?
        # Let's inspect shape at runtime if possible, or infer.
        # Standard ViLT: Text tokens match input_ids length.
        text_len = input_ids.shape[1]  # Typically 40

        # # Mask missing modalities
        # attention_mask = attention_mask.clone()
        # if text_missing.any():
        #     attention_mask[text_missing.bool(), 1:text_len] = 0
        # if image_missing.any():
        #     attention_mask[image_missing.bool(), text_len:] = 0
        # output = output * attention_mask.unsqueeze(-1)

        # Fix: Use masked mean for text to avoid padding
        mask_text = attention_mask[:, :text_len]
        text_mask_for_mean = mask_text.unsqueeze(-1).to(output.dtype)
        text_sum_mask = text_mask_for_mean.sum(dim=1).clamp(min=1e-9)

        # Init Gate Sources
        # Text Source: Mean of text tokens
        text_gate_source_init = (output[:, :text_len, :] * text_mask_for_mean).sum(dim=1) / text_sum_mask
        # Vision Source: Mean of vision tokens
        vision_gate_source_init = output[:, text_len:, :].mean(dim=1)

        # Fusion features for shared prompt gating
        if self.use_shared_prompt:
            fusion_gate_source_init = (text_gate_source_init + vision_gate_source_init) / 2
            prev_fusion_features = fusion_gate_source_init

        prev_vision_features = vision_gate_source_init
        prev_text_features = text_gate_source_init

        # Collect Dynamic Prompts for Loss
        text_dynamic_prompts_collected = []
        vision_dynamic_prompts_collected = []
        layerwise_text_features = []
        layerwise_vision_features = []
        if self.use_shared_prompt:
            shared_dynamic_prompts_collected = []

        # Prepare Attention Mask for ViLT
        # We will inject prompt tokens at start (conditionally merged)
        if self.use_shared_prompt:
            total_prompt_len = self.L_shared + self.L
        else:
            total_prompt_len = self.L

        prompt_mask = torch.ones(batch_size, total_prompt_len, device=device, dtype=attention_mask.dtype)

        # ViLT input: [CLS] (1) + Text (...)
        # We insert prompts after CLS (index 1)
        temp_mask = torch.cat([attention_mask[:, :1], prompt_mask, attention_mask[:, 1:]], dim=1)
        attention_mask_expanded = self._prepare_vilt_attention_mask(
            temp_mask, dtype=output.dtype, device=device
        )

        # Loop Layers
        num_layers = len(self.model.encoder.layer)

        for i, layer_module in enumerate(self.model.encoder.layer):
            if i < self.prompt_depth:
                # --- Text Prompt Gating ---
                # Gate with Vision Features
                gate_logits_t = self.text_gate_mlp(prev_vision_features)
                gate_t = torch.sigmoid(gate_logits_t).unsqueeze(1)  # [B, 1, D]

                # If Image Missing -> Gate = 1 (Identity)
                img_avail_mask = (~image_missing).float().view(-1, 1, 1)
                gate_t = gate_t * img_avail_mask + (1.0 - img_avail_mask)

                deep_prompt_t = text_proxy[i] * gate_t
                text_dynamic_prompts_collected.append(deep_prompt_t)

                # --- Vision Prompt Gating ---
                # Gate with Text Features
                gate_logits_v = self.vision_gate_mlp(prev_text_features)
                gate_v = torch.sigmoid(gate_logits_v).unsqueeze(1)

                # If Text Missing -> Gate = 1
                txt_avail_mask = (~text_missing).float().view(-1, 1, 1)
                gate_v = gate_v * txt_avail_mask + (1.0 - txt_avail_mask)

                deep_prompt_v = vision_proxy[i] * gate_v
                vision_dynamic_prompts_collected.append(deep_prompt_v)

                # --- NEW: Shared Prompt Gating ---
                if self.use_shared_prompt:
                    gate_logits_s = self.shared_gate_mlp(prev_fusion_features)
                    gate_s = torch.sigmoid(gate_logits_s).unsqueeze(1)  # [B, 1, D]

                    # When both missing → gate = 1 (identity)
                    both_missing = text_missing & image_missing
                    fusion_avail_mask = (~both_missing).float().view(-1, 1, 1)
                    gate_s = gate_s * fusion_avail_mask + (1.0 - fusion_avail_mask)

                    deep_prompt_s = shared_proxy[i] * gate_s
                    shared_dynamic_prompts_collected.append(deep_prompt_s)

                # --- Injection ---
                # Prepend: [CombinedPrompt, Content]
                # Logic: Use prompt_t when t missing, prompt_v when v missing, mean when complete.
                w_complete = (~text_missing & ~image_missing).float().view(-1, 1, 1)
                w_t_miss = text_missing.float().view(-1, 1, 1)
                w_v_miss = image_missing.float().view(-1, 1, 1)

                # Weights
                w_t = w_t_miss + 0.5 * w_complete
                w_v = w_v_miss + 0.5 * w_complete

                # Combine all prompts: [Shared, Text+Vision]
                if self.use_shared_prompt:
                    combined_prompt = torch.cat(
                        [
                            deep_prompt_s,  # L_shared
                            deep_prompt_t * w_t + deep_prompt_v * w_v,  # L (weighted combination)
                        ],
                        dim=1,
                    )  # [B, L_shared + L, D]
                else:
                    combined_prompt = deep_prompt_t * w_t + deep_prompt_v * w_v

                # Insert at index 1 (after CLS)
                layer_input = torch.cat([output[:, :1, :], combined_prompt, output[:, 1:, :]], dim=1)
            else:
                # No prompt injection (or use previous? Usually stop deep prompting)
                # But we need to handle mask size mismatch if we drop prompts.
                # If we drop prompts, we must revert attention mask.
                # However, usually Deep Prompting implies we only modify input at layers 0..K.
                # If we stop injecting, we feed `output` which has no prompts.
                # We must update `attention_mask` if input size changes.
                layer_input = output

            # Adjust mask if input size changed
            current_seq_len = layer_input.shape[1]
            if current_seq_len != temp_mask.shape[1]:
                # If we revert to original length (no prompts)
                # Recalculate mask for original
                mask_to_use = self._prepare_vilt_attention_mask(
                    attention_mask, dtype=output.dtype, device=device
                )
            else:
                mask_to_use = attention_mask_expanded

            # Forward Layer
            layer_outputs = layer_module(layer_input, attention_mask=mask_to_use)
            hidden_states = layer_outputs[0]

            # --- Post-Layer Processing ---
            if (i + 1) < self.prompt_depth:
                # Remove prompts to prepare for next layer's injection
                # Only if the next layer will inject new prompts
                if self.use_shared_prompt:
                    total_prompt_len = self.L_shared + self.L
                else:
                    total_prompt_len = self.L
                output = torch.cat(
                    [hidden_states[:, :1, :], hidden_states[:, 1 + total_prompt_len :, :]], dim=1
                )
            else:
                # Keep prompts (pass-through) for subsequent layers
                output = hidden_states

            # --- Extract Features for Next Gating ---
            # Update Gating Sources for next layer (i+1)
            if (i + 1) < self.prompt_depth:
                # Output is clean (stripped above)
                prev_text_features = (output[:, :text_len, :] * text_mask_for_mean).sum(dim=1) / text_sum_mask
                prev_vision_features = output[:, text_len:, :].mean(dim=1)
                if self.use_shared_prompt:
                    prev_fusion_features = (prev_text_features + prev_vision_features) / 2
            else:
                # Output has prompts: [CLS, Prompt, TextRest, Vision]
                # Strip prompts to get content
                if self.use_shared_prompt:
                    total_prompt_len = self.L_shared + self.L
                else:
                    total_prompt_len = self.L
                content = torch.cat([output[:, :1, :], output[:, 1 + total_prompt_len :, :]], dim=1)
                prev_text_features = (content[:, :text_len, :] * text_mask_for_mean).sum(
                    dim=1
                ) / text_sum_mask
                prev_vision_features = content[:, text_len:, :].mean(dim=1)
                if self.use_shared_prompt:
                    prev_fusion_features = (prev_text_features + prev_vision_features) / 2

            if i < self.prompt_depth:
                layerwise_text_features.append(prev_text_features)
                layerwise_vision_features.append(prev_vision_features)

        # --- Pool ---
        if self.prompt_depth > 0:
            # Remove prompts before pooling (to recover CLS at index 0)
            if self.use_shared_prompt:
                total_prompt_len = self.L_shared + self.L
            else:
                total_prompt_len = self.L
            output = torch.cat([output[:, :1, :], output[:, 1 + total_prompt_len :, :]], dim=1)

        output = self.model.layernorm(output)
        # ViLT Pooler usually takes CLS token (index 0)
        # output is now [CLS, Text, SEP, Image] (prompts removed)
        pooled_output = self.model.pooler(output)

        # --- Aux Loss ---
        aux_loss = 0.0
        text_dynamic_prompts = torch.stack(text_dynamic_prompts_collected)
        vision_dynamic_prompts = torch.stack(vision_dynamic_prompts_collected)

        text_p_mean = text_dynamic_prompts.mean(dim=2)
        vision_p_mean = vision_dynamic_prompts.mean(dim=2)

        num_loss_terms = 0

        for i in range(self.prompt_depth):
            # Text Prompt vs Layer-wise Text Feature
            valid_mask_txt = ~text_missing
            if valid_mask_txt.sum() > 1 and i < len(layerwise_text_features):
                tp = text_p_mean[i]
                ts = layerwise_text_features[i].detach()

                tp = F.normalize(tp, dim=-1)
                ts = F.normalize(ts, dim=-1)

                logits = torch.matmul(tp, ts.t()) * self.logit_scale.exp()

                valid_idx = torch.where(valid_mask_txt)[0]
                labels = torch.arange(len(valid_idx), device=device)

                aux_loss += F.cross_entropy(logits[valid_idx][:, valid_idx], labels)
                num_loss_terms += 1

            # Vision Prompt vs Layer-wise Vision Feature
            valid_mask_img = ~image_missing
            if valid_mask_img.sum() > 1 and i < len(layerwise_vision_features):
                vp = vision_p_mean[i]
                vs = layerwise_vision_features[i].detach()

                vp = F.normalize(vp, dim=-1)
                vs = F.normalize(vs, dim=-1)

                logits = torch.matmul(vp, vs.t()) * self.logit_scale.exp()

                valid_idx = torch.where(valid_mask_img)[0]
                labels = torch.arange(len(valid_idx), device=device)

                aux_loss += F.cross_entropy(logits[valid_idx][:, valid_idx], labels)
                num_loss_terms += 1

        if num_loss_terms > 0:
            aux_loss = aux_loss / num_loss_terms

        # --- Classifier ---
        logits = self.classifier(pooled_output)
        probs = F.softmax(logits, dim=-1)

        return {"logits": logits, "probs": probs, "aux_loss": aux_loss}

    def cal_loss(self, logits, probs, aux_loss=None, label=None, **kwargs):
        if len(label.shape) == 1:
            main_loss = F.cross_entropy(logits, label)
        else:
            main_loss = F.binary_cross_entropy_with_logits(logits, label.float())

        total_loss = main_loss
        if aux_loss is not None:
            total_loss += self.loss_alpha * aux_loss

        return total_loss, main_loss
