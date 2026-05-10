import torch
from torch import nn


class GenomePolisher(nn.Module):
    """Hybrid 1D-CNN + Transformer encoder for base-wise polishing."""

    def __init__(
        self,
        input_channels: int = 4,
        cnn_channels: int = 128,
        kernel_size: int = 5,
        attention_heads: int = 8,
        dropout: float = 0.1,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, cnn_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=cnn_channels,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(cnn_channels)
        self.classifier = nn.Linear(cnn_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (batch, sequence_length, 4).

        Returns:
            Logits of shape (batch, sequence_length, num_classes).
        """
        x = x.transpose(1, 2)  # (batch, channels, seq_len)
        features = self.cnn(x).transpose(1, 2)  # (batch, seq_len, channels)
        attended, _ = self.attention(features, features, features)
        output = self.norm(features + attended)
        return self.classifier(output)
