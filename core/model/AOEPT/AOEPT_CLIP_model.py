import copy
import math
from typing import Any

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


class AOEPT(Base):
    """
    AOEPT: Unified Multimodal Prompting with Adaptive Gating

    Uses AdaptivePromptGate for instance-aware, layer-wise gating that starts
    at ~0 to prevent shock to pre-trained backbone.
    """

    def __init__(
        self,
        cls_num: int,
        init_from_token: str | None = None,
        prompt_strategy: str = "attention",  # 'attention', 'init', 'mlp'
        N: int = 32,  # Init token length
        L: int = 16,  # Prompt length (L < N)
        seq_len: int = 77,
        prompt_depth: int = 6,  # Number of layers to prompt
        attn_num_heads: int = 8,  # Only used when prompt_strategy == 'attention'
        reduction_ratio: int = 4,  # Bottleneck ratio for gating MLP
        loss_alpha: float = 0.05,  # Weight of auxiliary loss
        use_attmatrix: bool = False,
        **kargs,
    ):
        super().__init__()
        self.arch = "CLIP"
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
        self.use_attmatrix = use_attmatrix
        self.use_shared_prompt = kargs.get("use_shared_prompt", True)
        self.L_shared = kargs.get("L_shared", self.L)

        # Load backbone with resized position embeddings to accommodate prompts
        # Default CLIP max_position_embeddings is 77
        total_prompt_len = (self.L + self.L_shared) if self.use_shared_prompt else self.L
        self.model = self.get_pretrained_backbone(self.arch, seq_len=seq_len + total_prompt_len)

        # Dimensions
        self.embed_dim = self.model.text_model.config.hidden_size  # 512
        self.vision_embed_dim = self.model.vision_model.config.hidden_size  # 768

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

        # Feature collection for prompt analysis
        self.statis = self.cfg.get("statis", None)
        self.collect_prompts = self.statis == "collect_prompts"
        self.prompt_collector = {}

        # Initialize trainable parameters
        self.init_trainable_para()

    def _get_current_epoch(self) -> int:
        """Return the current epoch if set by the trainer, else 0."""
        return getattr(self, "current_epoch", 0)

    def _init_tokens(self):
        """
        Load tokens from file and pool to length N.
        Stores:
            self.text_init_tokens: [prompt_depth, N, 512]
            self.vision_init_tokens: [prompt_depth, N, 768]
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
        # Format assumed same as MAPs: {'text_token': [Samples, Layers, D], ...}
        data: Any = torch.load(self.init_from_token, map_location=self.model.device)

        # Text
        if "text_token" in data:
            text_token = data["text_token"]

            if "cluster" in self.init_from_token:
                # Cluster file already has shape [Layers, K, D]; just take prompt_depth layers.
                self.N = text_token.shape[1]
                text_pooled = text_token.float()
            else:
                # Raw collection file has shape [Samples, Layers, D].
                # Filter all-zero samples, then pool Samples -> N along sample dim.
                text_non_zero = text_token.abs().sum(dim=(1, 2)) > 0
                text_token = text_token[text_non_zero]
                # [Samples, Layers, D] -> Permute -> [Layers, D, Samples]
                # AdaptivePool1d(N) -> [Layers, D, N] -> Permute -> [Layers, N, D]
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

            if "cluster" in self.init_from_token:
                vision_pooled = vision_token.float()
            else:
                vision_non_zero = vision_token.abs().sum(dim=(1, 2)) > 0
                vision_token = vision_token[vision_non_zero]
                vision_pooled = F.adaptive_avg_pool1d(
                    vision_token.float().permute(1, 2, 0), output_size=self.N
                ).permute(0, 2, 1)

            self.vision_init_tokens = nn.Parameter(vision_pooled[: self.prompt_depth].clone().detach())
        else:
            logger.warning("vision_token not found in init file. Initializing random vision tokens (missing_rate=1.0?).")
            self.vision_init_tokens = nn.Parameter(
                torch.randn(self.prompt_depth, self.N, self.vision_embed_dim) * 0.02
            )

        # Ensure they are parameters but we might freeze them depending on strategy
        # User says "use init prompt as Key and value" for attention.
        # "pool init token ... use it to init trainable prompt".

    def _init_prompt_strategy(self):
        """
        Initialize parameters based on prompt strategy.
        Strategies: 'attention', 'init', 'mlp'
        """
        if self.prompt_strategy == "attention":
            # Text
            # Trainable Query: [Layers, L, D]
            self.text_query = nn.Parameter(torch.randn(self.prompt_depth, self.L, self.embed_dim) * 0.02)
            # Keys/Values are self.text_init_tokens (Fixed or Finetuned? "use init prompt as Key and value")
            # Usually implies we use the loaded tokens as the memory.
            if self.use_attmatrix:
                text_hidden_dim = self.embed_dim // self.reduction_ratio
                self.text_k_proj = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(self.embed_dim, text_hidden_dim, bias=False),
                            nn.Linear(text_hidden_dim, self.embed_dim, bias=False),
                        )
                        for _ in range(self.prompt_depth)
                    ]
                )
                self.text_v_proj = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(self.embed_dim, text_hidden_dim, bias=False),
                            nn.Linear(text_hidden_dim, self.embed_dim, bias=False),
                        )
                        for _ in range(self.prompt_depth)
                    ]
                )

            # Vision
            self.vision_query = nn.Parameter(
                torch.randn(self.prompt_depth, self.L, self.vision_embed_dim) * 0.02
            )
            if self.use_attmatrix:
                vision_hidden_dim = self.vision_embed_dim // self.reduction_ratio
                self.vision_k_proj = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(self.vision_embed_dim, vision_hidden_dim, bias=False),
                            nn.Linear(vision_hidden_dim, self.vision_embed_dim, bias=False),
                        )
                        for _ in range(self.prompt_depth)
                    ]
                )
                self.vision_v_proj = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(self.vision_embed_dim, vision_hidden_dim, bias=False),
                            nn.Linear(vision_hidden_dim, self.vision_embed_dim, bias=False),
                        )
                        for _ in range(self.prompt_depth)
                    ]
                )

            self.text_layer_norm = nn.LayerNorm(self.embed_dim)
            self.vision_layer_norm = nn.LayerNorm(self.vision_embed_dim)

            if self.use_shared_prompt:
                # Shared query in bottleneck dimension (e.g., 256)
                shared_hidden_dim = self.embed_dim // self.reduction_ratio  # 512/4 = 128
                self.shared_query = nn.Parameter(
                    torch.randn(self.prompt_depth, self.L_shared, shared_hidden_dim) * 0.02
                )

                # Bottleneck projections: shared_query -> modality-specific queries
                self.shared_text_proj = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(shared_hidden_dim, shared_hidden_dim),
                            nn.GELU(),
                            nn.Linear(shared_hidden_dim, self.embed_dim),  # -> 512
                        )
                        for _ in range(self.prompt_depth)
                    ]
                )
                self.shared_vision_proj = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(shared_hidden_dim, shared_hidden_dim),
                            nn.GELU(),
                            nn.Linear(shared_hidden_dim, self.vision_embed_dim),  # -> 768
                        )
                        for _ in range(self.prompt_depth)
                    ]
                )

                # Layer norms for both modalities
                self.shared_text_layer_norm = nn.LayerNorm(self.embed_dim)
                self.shared_vision_layer_norm = nn.LayerNorm(self.vision_embed_dim)

        elif self.prompt_strategy == "init":
            # Text
            # Pool N -> L to init trainable parameter
            # [Layers, N, D] -> [Layers, D, N] -> Pool(L) -> [Layers, D, L] -> [Layers, L, D]
            text_pooled = F.adaptive_avg_pool1d(
                self.text_init_tokens.permute(0, 2, 1), output_size=self.L
            ).permute(0, 2, 1)
            self.text_proxy_prompts = nn.Parameter(text_pooled.clone().detach())

            # Vision
            vision_pooled = F.adaptive_avg_pool1d(
                self.vision_init_tokens.permute(0, 2, 1), output_size=self.L
            ).permute(0, 2, 1)
            self.vision_proxy_prompts = nn.Parameter(vision_pooled.clone().detach())

            if self.use_shared_prompt:
                # Text shared prompt
                text_pooled = F.adaptive_avg_pool1d(
                    self.text_init_tokens.permute(0, 2, 1), output_size=self.L_shared
                ).permute(0, 2, 1)
                self.shared_text_proxy_prompts = nn.Parameter(text_pooled.clone().detach())

                # Vision shared prompt
                vision_pooled = F.adaptive_avg_pool1d(
                    self.vision_init_tokens.permute(0, 2, 1), output_size=self.L_shared
                ).permute(0, 2, 1)
                self.shared_vision_proxy_prompts = nn.Parameter(vision_pooled.clone().detach())

        elif self.prompt_strategy == "mlp":
            # Text
            # Use N tokens directly as input
            self.text_mlp_input = nn.Parameter(
                self.text_init_tokens.clone().detach()
            )  # Input to MLP [Layers, N, D]

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

            if self.use_shared_prompt:
                # Shared input (bottleneck dimension)
                shared_hidden_dim = self.embed_dim // self.reduction_ratio
                # Initialize from text tokens (could also randomly initialize)
                text_pooled = F.adaptive_avg_pool1d(
                    self.text_init_tokens.permute(0, 2, 1), output_size=self.L_shared
                ).permute(0, 2, 1)
                # Project to bottleneck dimension
                self.shared_mlp_input = nn.Parameter(
                    F.adaptive_avg_pool1d(
                        text_pooled, output_size=shared_hidden_dim
                    )
                )

                # MLPs to generate text and vision shared prompts
                self.shared_text_mlp = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(shared_hidden_dim, shared_hidden_dim),
                            nn.GELU(),
                            nn.Linear(shared_hidden_dim, self.embed_dim),
                        )
                        for _ in range(self.prompt_depth)
                    ]
                )
                self.shared_vision_mlp = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(shared_hidden_dim, shared_hidden_dim),
                            nn.GELU(),
                            nn.Linear(shared_hidden_dim, self.vision_embed_dim),
                        )
                        for _ in range(self.prompt_depth)
                    ]
                )

        else:
            raise ValueError(f"Unknown prompt strategy: {self.prompt_strategy}")

    @staticmethod
    def _param_free_dot_attn(
        q: torch.Tensor, k: torch.Tensor, num_heads: int, v: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Parameter-free scaled dot-product attention (multi-head by reshaping only).
        Args:
            q: [B, L, D]
            k: [B, N, D]
            num_heads: number of heads to split D (no projections)
            v: [B, N, D]
        Returns:
            out: [B, L, D]
        """
        batch_size, q_len, embed_dim = q.shape
        _, k_len, _ = k.shape
        head_dim = embed_dim // num_heads

        q = q.view(batch_size, q_len, num_heads, head_dim).transpose(1, 2)  # [B, H, L, Dh]
        k = k.view(batch_size, k_len, num_heads, head_dim).transpose(1, 2)  # [B, H, N, Dh]
        if v is None:
            v = k
        else:
            v = v.view(batch_size, k_len, num_heads, head_dim).transpose(1, 2)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)  # [B, H, L, N]
        attn = attn_logits.softmax(dim=-1)
        out = torch.matmul(attn, v)  # [B, H, L, Dh]
        out = out.transpose(1, 2).contiguous().view(batch_size, q_len, embed_dim)
        return out

    def _init_gating(self):
        """
        Instance-aware gating with zero-initialization using channel-wise gating.

        Uses AdaptivePromptGate modules to generate per-dimension gate weights from
        cross-modal features (vision gates text, text gates vision).

        Gates are initialized to output ~0 (sigmoid(-5) ≈ 0.006), allowing the
        model to start with minimal prompt influence and gradually learn gating.
        """
        from core.model.AOEPT.module import AdaptivePromptGate

        # Text Gating (Using Vision Features)
        text_gate_hidden_dim = self.embed_dim
        self.text_gate_mlp = nn.ModuleList(
            [
                AdaptivePromptGate(
                    input_dim=self.vision_embed_dim,
                    hidden_dim=text_gate_hidden_dim,
                    reduction_ratio=self.reduction_ratio,
                )
                for _ in range(self.prompt_depth)
            ]
        )

        # Vision Gating (Using Text Features)
        vision_gate_hidden_dim = self.vision_embed_dim
        self.vision_gate_mlp = nn.ModuleList(
            [
                AdaptivePromptGate(
                    input_dim=self.embed_dim,
                    hidden_dim=vision_gate_hidden_dim,
                    reduction_ratio=self.reduction_ratio,
                )
                for _ in range(self.prompt_depth)
            ]
        )

        if self.use_shared_prompt:
            # Shared Text Gating (Using Vision Features)
            self.shared_text_gate_mlp = nn.ModuleList(
                [
                    AdaptivePromptGate(
                        input_dim=self.vision_embed_dim,  # 768
                        hidden_dim=self.embed_dim,  # 512
                        reduction_ratio=self.reduction_ratio,
                    )
                    for _ in range(self.prompt_depth)
                ]
            )

            # Shared Vision Gating (Using Text Features)
            self.shared_vision_gate_mlp = nn.ModuleList(
                [
                    AdaptivePromptGate(
                        input_dim=self.embed_dim,  # 512
                        hidden_dim=self.vision_embed_dim,  # 768
                        reduction_ratio=self.reduction_ratio,
                    )
                    for _ in range(self.prompt_depth)
                ]
            )

    def init_trainable_para(self):
        # Freeze backbone
        for param in self.model.parameters():
            param.requires_grad = False

        # Enable Strategy Parameters
        if self.prompt_strategy == "attention":
            self.text_query.requires_grad = True
            self.vision_query.requires_grad = True
            # Keys/Values (init tokens) - keep fixed or trainable?
            # "use init prompt as Key and value". Typically fixed if used as memory bank, or fine-tuned.
            # I will set them to fixed to avoid drifting too far from "init".
            self.text_init_tokens.requires_grad = False
            self.vision_init_tokens.requires_grad = False

            if self.use_attmatrix:
                for p in self.text_k_proj.parameters():
                    p.requires_grad = True
                for p in self.text_v_proj.parameters():
                    p.requires_grad = True
                for p in self.vision_k_proj.parameters():
                    p.requires_grad = True
                for p in self.vision_v_proj.parameters():
                    p.requires_grad = True

            for p in self.text_layer_norm.parameters():
                p.requires_grad = True
            for p in self.vision_layer_norm.parameters():
                p.requires_grad = True

            if self.use_shared_prompt:
                self.shared_query.requires_grad = True
                for p in self.shared_text_proj.parameters():
                    p.requires_grad = True
                for p in self.shared_vision_proj.parameters():
                    p.requires_grad = True
                for p in self.shared_text_layer_norm.parameters():
                    p.requires_grad = True
                for p in self.shared_vision_layer_norm.parameters():
                    p.requires_grad = True

        elif self.prompt_strategy == "init":
            self.text_proxy_prompts.requires_grad = True
            self.vision_proxy_prompts.requires_grad = True

            if self.use_shared_prompt:
                self.shared_text_proxy_prompts.requires_grad = True
                self.shared_vision_proxy_prompts.requires_grad = True

        elif self.prompt_strategy == "mlp":
            self.text_mlp_input.requires_grad = False  # Fixed input derived from pool
            self.vision_mlp_input.requires_grad = False
            for p in self.text_mlp.parameters():
                p.requires_grad = True
            for p in self.vision_mlp.parameters():
                p.requires_grad = True

            if self.use_shared_prompt:
                self.shared_mlp_input.requires_grad = False  # Fixed input
                for p in self.shared_text_mlp.parameters():
                    p.requires_grad = True
                for p in self.shared_vision_mlp.parameters():
                    p.requires_grad = True

        # Enable Gating
        if hasattr(self, "text_gate_mlp"):
            for p in self.text_gate_mlp.parameters():
                p.requires_grad = True
        if hasattr(self, "vision_gate_mlp"):
            for p in self.vision_gate_mlp.parameters():
                p.requires_grad = True

        if self.use_shared_prompt:
            if hasattr(self, "shared_text_gate_mlp"):
                for p in self.shared_text_gate_mlp.parameters():
                    p.requires_grad = True
            if hasattr(self, "shared_vision_gate_mlp"):
                for p in self.shared_vision_gate_mlp.parameters():
                    p.requires_grad = True

        # Enable Contrastive
        if hasattr(self, "logit_scale"):
            self.logit_scale.requires_grad = True

        # Enable Classifier & Base
        self.init_base_trainable_para(self.arch, self.model, self.classifier)

    def _get_proxy_prompts(self, batch_size):
        """
        Generate Layer-wise Proxy Prompts based on strategy.
        Returns:
            text_prompts: [Layers, B, L, D]
            vision_prompts: [Layers, B, L, D]
            shared_text_prompts: [Layers, B, L_shared, D] or None
            shared_vision_prompts: [Layers, B, L_shared, D_vision] or None
        """
        device = self.model.device

        if self.prompt_strategy == "attention":
            # Text
            # Query: [Layers, L, D] -> Expand B -> [Layers, B, L, D]
            # Key/Val: [Layers, N, D] -> Expand B -> [Layers, B, N, D]
            text_prompts = []
            for i in range(self.prompt_depth):
                q = self.text_query[i].unsqueeze(0).expand(batch_size, -1, -1)
                k_source = self.text_init_tokens[i].unsqueeze(0).expand(batch_size, -1, -1)

                if self.use_attmatrix:
                    k = self.text_k_proj[i](k_source)
                    v = self.text_v_proj[i](k_source)
                else:
                    k = k_source
                    v = None

                out = self._param_free_dot_attn(q, k, num_heads=self.attn_num_heads, v=v)
                out = self.text_layer_norm(out + q)
                text_prompts.append(out)
            text_prompts = torch.stack(text_prompts)  # [Layers, B, L, D]

            # Vision
            vision_prompts = []
            for i in range(self.prompt_depth):
                q = self.vision_query[i].unsqueeze(0).expand(batch_size, -1, -1)
                k_source = self.vision_init_tokens[i].unsqueeze(0).expand(batch_size, -1, -1)

                if self.use_attmatrix:
                    k = self.vision_k_proj[i](k_source)
                    v = self.vision_v_proj[i](k_source)
                else:
                    k = k_source
                    v = None

                out = self._param_free_dot_attn(q, k, num_heads=self.attn_num_heads, v=v)
                out = self.vision_layer_norm(out + q)
                vision_prompts.append(out)
            vision_prompts = torch.stack(vision_prompts)

            if self.use_shared_prompt:
                shared_text_prompts = []
                shared_vision_prompts = []
                for i in range(self.prompt_depth):
                    # Shared query (bottleneck dimension)
                    q_shared = (
                        self.shared_query[i].unsqueeze(0).expand(batch_size, -1, -1)
                    )  # [B, L_shared, hidden_dim]

                    # Project to modality-specific queries
                    q_text = self.shared_text_proj[i](q_shared)  # [B, L_shared, 512]
                    q_vision = self.shared_vision_proj[i](q_shared)  # [B, L_shared, 768]

                    # Get memory banks (same-modal)
                    text_memory = (
                        self.text_init_tokens[i].unsqueeze(0).expand(batch_size, -1, -1)
                    )  # [B, N, 512]
                    vision_memory = (
                        self.vision_init_tokens[i].unsqueeze(0).expand(batch_size, -1, -1)
                    )  # [B, N, 768]

                    # Attend over same-modal memory
                    out_text = self._param_free_dot_attn(q_text, text_memory, num_heads=self.attn_num_heads)
                    out_text = self.shared_text_layer_norm(out_text + q_text)
                    shared_text_prompts.append(out_text)

                    out_vision = self._param_free_dot_attn(
                        q_vision, vision_memory, num_heads=self.attn_num_heads
                    )
                    out_vision = self.shared_vision_layer_norm(out_vision + q_vision)
                    shared_vision_prompts.append(out_vision)

                shared_text_prompts = torch.stack(shared_text_prompts)  # [prompt_depth, B, L_shared, 512]
                shared_vision_prompts = torch.stack(shared_vision_prompts)  # [prompt_depth, B, L_shared, 768]
            else:
                shared_text_prompts = None
                shared_vision_prompts = None

        elif self.prompt_strategy == "init":
            # [Layers, L, D] -> [Layers, B, L, D]
            text_prompts = self.text_proxy_prompts.unsqueeze(1).expand(-1, batch_size, -1, -1)
            vision_prompts = self.vision_proxy_prompts.unsqueeze(1).expand(-1, batch_size, -1, -1)

            if self.use_shared_prompt:
                shared_text_prompts = self.shared_text_proxy_prompts.unsqueeze(1).expand(
                    -1, batch_size, -1, -1
                )
                shared_vision_prompts = self.shared_vision_proxy_prompts.unsqueeze(1).expand(
                    -1, batch_size, -1, -1
                )
            else:
                shared_text_prompts = None
                shared_vision_prompts = None

        elif self.prompt_strategy == "mlp":
            text_prompts = []
            for i in range(self.prompt_depth):
                # [N, D] -> [B, N, D] -> MLP
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

            if self.use_shared_prompt:
                shared_text_prompts = []
                shared_vision_prompts = []
                for i in range(self.prompt_depth):
                    inp = self.shared_mlp_input[i].unsqueeze(0).expand(batch_size, -1, -1)

                    out_text = self.shared_text_mlp[i](inp)
                    out_text = F.adaptive_avg_pool1d(
                        out_text.permute(0, 2, 1), output_size=self.L_shared
                    ).permute(0, 2, 1)
                    shared_text_prompts.append(out_text)

                    out_vision = self.shared_vision_mlp[i](inp)
                    out_vision = F.adaptive_avg_pool1d(
                        out_vision.permute(0, 2, 1), output_size=self.L_shared
                    ).permute(0, 2, 1)
                    shared_vision_prompts.append(out_vision)

                shared_text_prompts = torch.stack(shared_text_prompts)
                shared_vision_prompts = torch.stack(shared_vision_prompts)
            else:
                shared_text_prompts = None
                shared_vision_prompts = None

        return text_prompts, vision_prompts, shared_text_prompts, shared_vision_prompts

    def forward(self, **inputs):
        # Extract Inputs
        input_ids = inputs.get("input_ids")
        pixel_values = inputs.get("pixel_values")
        attention_mask = inputs.get("attention_mask")
        missing_masks = inputs.get("missing_masks")  # [B, 2] (True if missing)
        sample_ids = inputs.pop("ids", None)

        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Mask for Availability
        text_missing = missing_masks[:, 0]
        image_missing = missing_masks[:, 1]

        # 1. Generate Proxy Prompts (Base Prompts)
        # [Layers, B, L, D]
        text_proxy, vision_proxy, shared_text_proxy, shared_vision_proxy = self._get_proxy_prompts(batch_size)

        # 2. Initial Embeddings
        # Text
        # Construct inputs_embeds to ensure correct Position Embeddings for [Prompt, Text]
        p0_text = text_proxy[0]
        if self.use_shared_prompt:
            p0_shared = shared_text_proxy[0]
            p0_text = torch.cat([p0_shared, p0_text], dim=1)
        token_embeds = self.model.text_model.embeddings.token_embedding(input_ids)
        inputs_embeds = torch.cat([p0_text, token_embeds], dim=1)
        text_hidden_states = self.model.text_model.embeddings(inputs_embeds=inputs_embeds)

        # Vision
        image_hidden_states = self.model.vision_model.embeddings(pixel_values)
        image_hidden_states = self.model.vision_model.pre_layrnorm(image_hidden_states)

        # 3. Prepare Layer 0 Prompts
        # We need Gate Source for Layer 0.
        # Use initial embeddings (as in original AOEPT) or ungated?
        # Original AOEPT used raw embeddings to gate Layer 0.

        # Get Gate Sources (Raw Embeddings)
        # Text Source: Mean pooling (skip prompt)
        text_gate_source_init = text_hidden_states[:, self.L :, :].mean(dim=1)

        # Vision Source: Mean pooling
        vision_gate_source_init = image_hidden_states.mean(dim=1)

        # Keep EOS index for final text pooling (not used for gating)
        eos_token_id = self.model.text_model.config.eos_token_id
        eos_idx = (input_ids == eos_token_id).int().argmax(dim=-1)

        # Robust check: If EOS is missing (truncated), use the last token
        found_eos = (input_ids == eos_token_id).any(dim=-1)
        if not found_eos.all():
            seq_len_ids = input_ids.shape[1]
            eos_idx = torch.where(
                found_eos, eos_idx, torch.tensor(seq_len_ids - 1, device=device, dtype=eos_idx.dtype)
            )

        # Collect Dynamic Prompts for Loss
        text_dynamic_prompts_collected = []
        vision_dynamic_prompts_collected = []
        layerwise_text_features = []
        layerwise_vision_features = []

        if self.use_shared_prompt:
            shared_text_dynamic_prompts_collected = []
            shared_vision_dynamic_prompts_collected = []

        # --- Interleaved Forward Loop ---

        # Prepend initial placeholders (will be replaced/filled)
        # Text: Already prepended in Initial Embeddings
        pass

        # Vision: [P, Image]
        p0_vision = vision_proxy[0]
        if self.use_shared_prompt:
            p0_shared_v = shared_vision_proxy[0]
            p0_vision = torch.cat([p0_shared_v, p0_vision], dim=1)
        image_hidden_states = torch.cat([p0_vision, image_hidden_states], dim=1)

        # Attention Masks
        # Text
        total_prompt_len = (self.L_shared + self.L) if self.use_shared_prompt else self.L
        prompt_mask = torch.ones(batch_size, total_prompt_len, device=device, dtype=attention_mask.dtype)
        full_attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        # Vision: CLIP vision usually doesn't take mask in encoder (it's full attn),
        # but we pass None.

        # Prepare Loop
        # We need to track previous layer outputs for Cross-Gating.
        # Initialize with Init Sources
        prev_vision_features = vision_gate_source_init  # [B, D_vis]
        prev_text_features = text_gate_source_init  # [B, D_txt]

        # Determine max layers
        num_text_layers = len(self.model.text_model.encoder.layers)
        num_vision_layers = len(self.model.vision_model.encoder.layers)
        max_layers = max(num_text_layers, num_vision_layers)

        for i in range(max_layers):
            # --- Text Layer i ---
            if i < num_text_layers:
                # 1. Update Prompt (Injection) if within depth
                if i < self.prompt_depth:
                    # Gating Logic: Use previous Vision features to gate Text Prompt
                    # Gate Input: prev_vision_features
                    gate_logits_t = self.text_gate_mlp[i](prev_vision_features)
                    gate_t = torch.sigmoid(gate_logits_t).unsqueeze(1)  # [B, 1, D]

                    # Apply Mask: If Image is Missing, Gate = 1 (Identity)
                    # mask: [B, 1, 1]
                    img_avail_mask = (~image_missing).float().view(-1, 1, 1)
                    gate_t = gate_t * img_avail_mask + (1.0 - img_avail_mask)

                    # Apply Gate
                    deep_prompt_t = text_proxy[i] * gate_t
                    text_dynamic_prompts_collected.append(deep_prompt_t)

                    # Shared TEXT prompt gating (NEW)
                    if self.use_shared_prompt:
                        gate_logits_st = self.shared_text_gate_mlp[i](prev_vision_features)
                        gate_st = torch.sigmoid(gate_logits_st).unsqueeze(1)

                        # When image missing → gate = 1
                        img_avail_mask = (~image_missing).float().view(-1, 1, 1)
                        gate_st = gate_st * img_avail_mask + (1.0 - img_avail_mask)

                        deep_prompt_st = shared_text_proxy[i] * gate_st
                        shared_text_dynamic_prompts_collected.append(deep_prompt_st)

                        # Combine: [Shared_Text, Text]
                        combined_prompt = torch.cat([deep_prompt_st, deep_prompt_t], dim=1)
                    else:
                        combined_prompt = deep_prompt_t

                    # Inject combined prompt
                    features = text_hidden_states[:, total_prompt_len:, :]
                    text_hidden_states = torch.cat([combined_prompt, features], dim=1)

                # 2. Forward Layer
                seq_len = text_hidden_states.shape[1]
                causal_attention_mask = _create_4d_causal_attention_mask(
                    (batch_size, seq_len), text_hidden_states.dtype, device=device
                )
                expanded_mask = _prepare_4d_attention_mask(full_attention_mask, text_hidden_states.dtype)

                layer_module = self.model.text_model.encoder.layers[i]
                text_hidden_states = layer_module(
                    text_hidden_states,
                    attention_mask=expanded_mask,
                    causal_attention_mask=causal_attention_mask,
                    output_attentions=False,
                )[0]

                # 3. Extract Features for Next Vision Layer Gating
                # Use mean pooling (skip prompt tokens)
                prev_text_features = text_hidden_states[:, total_prompt_len:, :].mean(dim=1)  # [B, D]
                if i < self.prompt_depth:
                    layerwise_text_features.append(prev_text_features)

            # --- Vision Layer i ---
            if i < num_vision_layers:
                # 1. Update Prompt
                if i < self.prompt_depth:
                    # Gating Logic: Use previous Text features to gate Vision Prompt
                    gate_logits_v = self.vision_gate_mlp[i](prev_text_features)
                    gate_v = torch.sigmoid(gate_logits_v).unsqueeze(1)

                    txt_avail_mask = (~text_missing).float().view(-1, 1, 1)
                    gate_v = gate_v * txt_avail_mask + (1.0 - txt_avail_mask)

                    deep_prompt_v = vision_proxy[i] * gate_v
                    vision_dynamic_prompts_collected.append(deep_prompt_v)

                    # Shared VISION prompt gating (NEW)
                    if self.use_shared_prompt:
                        gate_logits_sv = self.shared_vision_gate_mlp[i](prev_text_features)
                        gate_sv = torch.sigmoid(gate_logits_sv).unsqueeze(1)

                        # When text missing → gate = 1
                        txt_avail_mask = (~text_missing).float().view(-1, 1, 1)
                        gate_sv = gate_sv * txt_avail_mask + (1.0 - txt_avail_mask)

                        deep_prompt_sv = shared_vision_proxy[i] * gate_sv
                        shared_vision_dynamic_prompts_collected.append(deep_prompt_sv)

                        # Combine: [Shared_Vision, Vision]
                        combined_prompt_v = torch.cat([deep_prompt_sv, deep_prompt_v], dim=1)
                    else:
                        combined_prompt_v = deep_prompt_v

                    # Inject combined prompt (NOTE: now total_prompt_len for vision too!)
                    features = image_hidden_states[:, total_prompt_len:, :]
                    image_hidden_states = torch.cat([combined_prompt_v, features], dim=1)

                # 2. Forward Layer
                layer_module = self.model.vision_model.encoder.layers[i]
                image_hidden_states = layer_module(
                    image_hidden_states,
                    attention_mask=None,
                    causal_attention_mask=None,
                    output_attentions=False,
                )[0]

                # 3. Extract Features for Next Text Layer Gating
                # Use mean pooling (skip prompt tokens)
                prev_vision_features = image_hidden_states[:, total_prompt_len:, :].mean(dim=1)  # [B, D]
                if i < self.prompt_depth:
                    layerwise_vision_features.append(prev_vision_features)

        # Collect sample-specific gated prompts if requested
        if self.collect_prompts and sample_ids is not None and len(text_dynamic_prompts_collected) > 0:
            from core.utils.stats_utils import get_sample_specific_prompts

            # Stack collected prompts: [prompt_depth, B, L, D]
            text_stacked = torch.stack(text_dynamic_prompts_collected)  # [prompt_depth, B, L, 512]
            vision_stacked = torch.stack(vision_dynamic_prompts_collected)  # [prompt_depth, B, L, 768]

            # Transpose to match expected format: [B, prompt_depth, L, D]
            text_prompts = text_stacked.permute(1, 0, 2, 3)
            vision_prompts = vision_stacked.permute(1, 0, 2, 3)

            get_sample_specific_prompts(
                epoch=self._get_current_epoch(),
                missing_aware_text_prompt=text_prompts,
                missing_aware_vision_prompt=vision_prompts,
                sample_ids=sample_ids,
                prompt_collector=self.prompt_collector,
            )

        # --- Post Processing ---

        # Remove Prompts (both streams now use total_prompt_len)
        total_prompt_len_final = (self.L_shared + self.L) if self.use_shared_prompt else self.L
        text_hidden_states = text_hidden_states[:, total_prompt_len_final:, :]
        image_hidden_states = image_hidden_states[:, total_prompt_len_final:, :]

        # Text Pooling (Final LN already in encoder? No, CLIP applies it after)
        text_hidden_states = self.model.text_model.final_layer_norm(text_hidden_states)
        text_pooled = text_hidden_states[torch.arange(batch_size, device=device), eos_idx]

        # Vision Pooling
        image_pooled = image_hidden_states[:, 0, :]  # CLS
        image_pooled = self.model.vision_model.post_layernorm(image_pooled)

        # Projections
        text_proj = self.model.text_projection(text_pooled)
        image_proj = self.model.visual_projection(image_pooled)

        # --- Auxiliary Loss (Contrastive) ---
        aux_loss = 0.0
        # Reconstruct collections to Stack
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

        # Classifier
        combined = torch.cat([image_proj, text_proj], dim=-1)
        logits = self.classifier(combined)
        probs = F.softmax(logits, dim=-1)

        return {
            "logits": logits,
            "probs": probs,
            "aux_loss": aux_loss,
            "prompt_collector": self.prompt_collector if self.collect_prompts else None,
        }

    def cal_loss(self, logits, probs, aux_loss=None, label=None, **kwargs):
        if len(label.shape) == 1:
            main_loss = F.cross_entropy(logits, label)
        else:
            main_loss = F.binary_cross_entropy_with_logits(logits, label.float())

        total_loss = main_loss
        if aux_loss is not None:
            total_loss += self.loss_alpha * aux_loss

        return total_loss, main_loss
