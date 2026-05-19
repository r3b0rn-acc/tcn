from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tcn.dataset import StandardScaler1D, make_windows_for_origins
from tcn.model import TCNForecaster


@dataclass(frozen=True)
class TCNTrainingConfig:
    lookback: int = 36
    horizons: tuple[int, ...] = (1, 3, 6, 12)
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    validation_fraction: float = 0.2
    patience: int = 8
    channels: tuple[int, ...] = (32, 32, 32)
    kernel_size: int = 3
    dropout: float = 0.1
    device: str = "auto"
    require_gpu: bool = False
    seed: int = 42


@dataclass
class TCNForecastResult:
    model_name: str
    predictions: np.ndarray
    actuals: np.ndarray
    origins: np.ndarray
    device: str
    train_losses: list[float]
    validation_losses: list[float]


def resolve_device(device: str = "auto", require_gpu: bool = False) -> torch.device:
    requested = device.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if require_gpu:
            raise RuntimeError("CUDA GPU is required but torch.cuda.is_available() is False")
        return torch.device("cpu")
    if requested in {"cuda", "gpu"}:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device was requested but is not available")
        return torch.device("cuda")
    if requested == "cpu":
        if require_gpu:
            raise RuntimeError("CPU was requested while require_gpu=True")
        return torch.device("cpu")
    raise ValueError("device must be one of: auto, cuda, gpu, cpu")


def train_tcn_forecaster(
    model_name: str,
    input_values: np.ndarray,
    target_values: np.ndarray,
    split_idx: int,
    train_origins: Sequence[int],
    test_origins: Sequence[int],
    config: TCNTrainingConfig,
) -> TCNForecastResult:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    input_scaler = StandardScaler1D.fit(input_values[:split_idx])
    target_scaler = StandardScaler1D.fit(target_values[:split_idx])
    scaled_inputs = input_scaler.transform(input_values)
    scaled_targets = target_scaler.transform(target_values)

    x_train, y_train = make_windows_for_origins(
        scaled_inputs,
        scaled_targets,
        train_origins,
        config.lookback,
        config.horizons,
    )
    x_test, y_test_scaled = make_windows_for_origins(
        scaled_inputs,
        scaled_targets,
        test_origins,
        config.lookback,
        config.horizons,
    )

    device = resolve_device(config.device, require_gpu=config.require_gpu)
    model = TCNForecaster(
        horizon_count=len(config.horizons),
        channels=config.channels,
        kernel_size=config.kernel_size,
        dropout=config.dropout,
    ).to(device)

    train_loader, validation_loader = _make_loaders(x_train, y_train, config, device.type == "cuda")
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_validation = float("inf")
    epochs_without_improvement = 0
    train_losses: list[float] = []
    validation_losses: list[float] = []

    for _epoch in range(config.epochs):
        model.train()
        epoch_losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        train_loss = float(np.mean(epoch_losses))
        validation_loss = _evaluate_loss(model, validation_loader, criterion, device)
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)

        if validation_loss < best_validation - 1e-6:
            best_validation = validation_loss
            epochs_without_improvement = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_tensor = torch.as_tensor(x_test, dtype=torch.float32, device=device)
        predictions_scaled = model(test_tensor).detach().cpu().numpy()

    predictions = target_scaler.inverse_transform(predictions_scaled)
    actuals = target_scaler.inverse_transform(y_test_scaled)
    return TCNForecastResult(
        model_name=model_name,
        predictions=predictions,
        actuals=actuals,
        origins=np.asarray(test_origins, dtype=int),
        device=str(device),
        train_losses=train_losses,
        validation_losses=validation_losses,
    )


def _make_loaders(
    x_train: np.ndarray,
    y_train: np.ndarray,
    config: TCNTrainingConfig,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader]:
    n_samples = len(x_train)
    validation_size = max(1, int(n_samples * config.validation_fraction))
    if n_samples - validation_size < 2:
        validation_size = 1
    split = n_samples - validation_size

    train_dataset = TensorDataset(
        torch.as_tensor(x_train[:split], dtype=torch.float32),
        torch.as_tensor(y_train[:split], dtype=torch.float32),
    )
    validation_dataset = TensorDataset(
        torch.as_tensor(x_train[split:], dtype=torch.float32),
        torch.as_tensor(y_train[split:], dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=pin_memory,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )
    return train_loader, validation_loader


def _evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            losses.append(float(criterion(model(batch_x), batch_y).detach().cpu()))
    return float(np.mean(losses))
