"""Federated learning training infrastructure for BA-FedSHAP.

Implements FedAvg with PyTorch models:
- LogisticRegressionModel: single linear layer
- MLPModel: 2-hidden-layer MLP
- local_train: client-side SGD training
- fedavg_aggregate: weighted model averaging
- FederatedTrainer: orchestrates multi-round FL
"""

from __future__ import annotations

import copy
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class LogisticRegressionModel(nn.Module):
    """Logistic regression as a single linear layer with sigmoid output."""

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """sklearn-compatible interface, returns shape (N, 2)."""
        self.eval()
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32)
            p1 = self.forward(t).squeeze(-1).numpy()
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class MLPModel(nn.Module):
    """Two-hidden-layer MLP with ReLU activations."""

    def __init__(self, n_features: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32)
            p1 = self.forward(t).squeeze(-1).numpy()
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def build_model(model_name: str, n_features: int) -> nn.Module:
    """Factory function for models."""
    if model_name == "logreg":
        return LogisticRegressionModel(n_features)
    elif model_name == "mlp":
        return MLPModel(n_features)
    else:
        raise ValueError(f"Unknown model '{model_name}'. Choose 'logreg' or 'mlp'.")


# ---------------------------------------------------------------------------
# Local training
# ---------------------------------------------------------------------------

def local_train(
    model: nn.Module,
    dataset: Dict,
    epochs: int = 5,
    lr: float = 0.01,
    momentum: float = 0.9,
    batch_size: int = 32,
    seed: int = 42,
) -> nn.Module:
    """Train model on a single client's dataset for `epochs` local epochs.

    Returns a new model (deep copy) trained locally; does NOT modify `model` in-place.
    """
    torch.manual_seed(seed)
    local_model = copy.deepcopy(model)
    local_model.train()

    X = torch.tensor(dataset["X"], dtype=torch.float32)
    y = torch.tensor(dataset["y"], dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(X, y),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    optimizer = optim.SGD(local_model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.BCELoss()

    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = local_model(xb).squeeze(-1)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

    return local_model


# ---------------------------------------------------------------------------
# FedAvg aggregation
# ---------------------------------------------------------------------------

def fedavg_aggregate(
    models: List[nn.Module],
    weights: Optional[List[float]] = None,
) -> nn.Module:
    """Weighted average of model parameters (FedAvg).

    Args:
        models: List of client models (all same architecture).
        weights: Per-client weights (e.g. n_samples). Defaults to uniform.

    Returns:
        New model with aggregated parameters.
    """
    if not models:
        raise ValueError("models list is empty")

    n = len(models)
    if weights is None:
        weights = [1.0 / n] * n
    else:
        total = sum(weights)
        weights = [w / total for w in weights]

    global_model = copy.deepcopy(models[0])

    with torch.no_grad():
        for name, param in global_model.named_parameters():
            param.data = torch.zeros_like(param.data)
            for model, w in zip(models, weights):
                param.data += w * dict(model.named_parameters())[name].data

    return global_model


# ---------------------------------------------------------------------------
# FederatedTrainer
# ---------------------------------------------------------------------------

class FederatedTrainer:
    """Orchestrates FedAvg training over multiple rounds.

    Usage::

        trainer = FederatedTrainer()
        global_model = trainer.fit(
            client_datasets=client_datasets,
            global_model=init_model,
            rounds=200,
            local_epochs=5,
            lr=0.01,
            momentum=0.9,
        )
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def fit(
        self,
        client_datasets: List[Dict],
        global_model: nn.Module,
        rounds: int = 200,
        local_epochs: int = 5,
        lr: float = 0.01,
        momentum: float = 0.9,
        batch_size: int = 32,
        participation_rate: float = 1.0,
        seed: Optional[int] = None,
    ) -> nn.Module:
        """Run FedAvg for `rounds` communication rounds.

        Args:
            client_datasets: List of client data dicts (keys: 'X', 'y', 'sensitive').
            global_model: Initial global model.
            rounds: Number of FL communication rounds.
            local_epochs: Local SGD epochs per round.
            lr: Learning rate for SGD.
            momentum: SGD momentum.
            batch_size: Minibatch size for local training.
            participation_rate: Fraction of clients sampled per round (1.0 = all).
            seed: Override for random seed.

        Returns:
            Trained global model.
        """
        rng = np.random.default_rng(seed or self.seed)
        n_clients = len(client_datasets)
        n_selected = max(1, int(n_clients * participation_rate))

        current_model = copy.deepcopy(global_model)

        for r in range(rounds):
            # Sample clients
            selected = rng.choice(n_clients, size=n_selected, replace=False).tolist()
            client_models = []
            client_weights = []

            for k in selected:
                local_seed = int(rng.integers(0, 2**31))
                local_model = local_train(
                    model=current_model,
                    dataset=client_datasets[k],
                    epochs=local_epochs,
                    lr=lr,
                    momentum=momentum,
                    batch_size=batch_size,
                    seed=local_seed,
                )
                client_models.append(local_model)
                client_weights.append(float(client_datasets[k]["n"]))

            current_model = fedavg_aggregate(client_models, weights=client_weights)

            if (r + 1) % 50 == 0 or r == 0:
                logger.info("Round %d/%d complete.", r + 1, rounds)

        logger.info("FedAvg training complete (%d rounds, %d clients).", rounds, n_clients)
        return current_model

    def evaluate(
        self,
        model: nn.Module,
        client_datasets: List[Dict],
    ) -> Dict[str, float]:
        """Evaluate global model on all clients, return mean accuracy and loss."""
        from sklearn.metrics import accuracy_score, log_loss  # type: ignore

        accs, losses = [], []
        for d in client_datasets:
            if d["n"] == 0:
                continue
            proba = model.predict_proba(d["X"])
            preds = (proba[:, 1] >= 0.5).astype(int)
            accs.append(accuracy_score(d["y"], preds))
            try:
                losses.append(log_loss(d["y"], proba))
            except Exception:
                pass

        return {
            "accuracy_mean": float(np.mean(accs)) if accs else float("nan"),
            "accuracy_std": float(np.std(accs)) if accs else float("nan"),
            "loss_mean": float(np.mean(losses)) if losses else float("nan"),
        }
