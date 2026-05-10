import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from src.model import GenomePolisher


def train_genome_polisher(
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-2,
    max_grad_norm: float = 1.0,
    device: str | None = None,
) -> GenomePolisher:
    """Train GenomePolisher using AdamW and gradient clipping."""
    if x_train.ndim != 3 or x_train.shape[-1] != 4:
        raise ValueError("x_train must have shape (batch, sequence_length, 4)")

    if y_train.shape[:2] != x_train.shape[:2]:
        raise ValueError(
            "y_train shape must match x_train shape in first two dimensions "
            "(batch, sequence_length)"
        )

    if device is None:
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        try:
            torch_device = torch.device(device)
        except (TypeError, RuntimeError) as exc:
            raise ValueError(f"Invalid device specification: {device}") from exc
    model = GenomePolisher().to(torch_device)

    dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(torch_device)
            batch_y = batch_y.to(torch_device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), batch_y.reshape(-1))
            loss.backward()
            clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} - loss: {running_loss / len(dataloader):.4f}")

    return model


if __name__ == "__main__":
    samples, seq_len = 8, 100
    x_dummy = np.eye(4, dtype=np.float32)[np.random.randint(0, 4, (samples, seq_len))]
    y_dummy = np.random.randint(0, 4, (samples, seq_len), dtype=np.int64)
    train_genome_polisher(x_dummy, y_dummy, epochs=1)
    print("Training finished.")
