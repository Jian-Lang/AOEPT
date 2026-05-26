import torch
import torch.nn as nn


class AdaptivePromptGate(nn.Module):
    """
    Instance-aware adaptive gating for prompts using channel-wise gating.

    Uses a bottleneck MLP to generate per-dimension gate weights from input features.
    Zero-initialized to prevent shock to pre-trained models.

    This implements channel-wise gating where each dimension of the prompt embeddings
    is gated independently, allowing fine-grained control over which features are active.

    Args:
        input_dim: Dimension of input features (gate source)
        hidden_dim: Dimension of the prompt embeddings to gate
        reduction_ratio: Bottleneck ratio for parameter efficiency
    """

    def __init__(self, input_dim, hidden_dim, reduction_ratio=4):
        super().__init__()
        gate_hidden = input_dim // reduction_ratio

        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, hidden_dim),
        )

        # Zero-initialization: sigmoid(-5) ≈ 0.006
        # nn.init.zeros_(self.gate_net[-1].weight)
        # nn.init.constant_(self.gate_net[-1].bias, -5.0)

    def forward(self, source_features):
        """
        Args:
            source_features: [B, D] tensor from cross-modal features
        Returns:
            gate_logits: [B, hidden_dim] raw logits (no sigmoid applied)

        Note: Sigmoid is applied in the caller to allow for missing modality handling.
        """
        gate_logits = self.gate_net(source_features)  # [B, hidden_dim]
        return gate_logits
