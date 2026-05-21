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
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    validation_fraction: float = 0.2
    patience: int = 15
    channels: tuple[int, ...] = (32, 32, 32, 32)
    kernel_size: int = 3
    dropout: float = 0.05
    gradient_clip_norm: float | None = 1.0
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


@dataclass(frozen=True)
class HorizonScaler:
    mean_: np.ndarray
    scale_: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "HorizonScaler":
        values = np.asarray(values, dtype=float)
        if values.ndim != 2:
            raise ValueError("values must have shape [samples, horizons]")
        scale = np.std(values, axis=0)
        scale = np.where(scale <= np.finfo(float).eps, 1.0, scale)
        return cls(mean_=np.mean(values, axis=0), scale_=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=float) - self.mean_) / self.scale_

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float) * self.scale_ + self.mean_


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

    target_values = np.asarray(target_values, dtype=float)
    train_origins = np.asarray(train_origins, dtype=int)
    test_origins = np.asarray(test_origins, dtype=int)

    input_scaler = StandardScaler1D.fit(input_values[:split_idx])
    scaled_inputs = input_scaler.transform(input_values)

    x_train, train_targets = make_windows_for_origins(
        scaled_inputs,
        target_values,
        train_origins,
        config.lookback,
        config.horizons,
    )
    x_test, actuals = make_windows_for_origins(
        scaled_inputs,
        target_values,
        test_origins,
        config.lookback,
        config.horizons,
    )

    train_residuals = make_residual_targets(target_values, train_origins, train_targets)
    target_scaler = HorizonScaler.fit(train_residuals)
    y_train = target_scaler.transform(train_residuals)

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
            if config.gradient_clip_norm is not None and config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip_norm)
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
        prediction_residuals_scaled = model(test_tensor).detach().cpu().numpy()

    prediction_residuals = target_scaler.inverse_transform(prediction_residuals_scaled)
    predictions = restore_level_predictions(target_values, test_origins, prediction_residuals)
    return TCNForecastResult(
        model_name=model_name,
        predictions=predictions,
        actuals=actuals,
        origins=test_origins,
        device=str(device),
        train_losses=train_losses,
        validation_losses=validation_losses,
    )


def make_residual_targets(
    target_values: np.ndarray,
    origins: Sequence[int],
    level_targets: np.ndarray,
) -> np.ndarray:
    target_values = np.asarray(target_values, dtype=float)
    origins = np.asarray(origins, dtype=int)
    level_targets = np.asarray(level_targets, dtype=float)
    return level_targets - target_values[origins][:, None]


def restore_level_predictions(
    target_values: np.ndarray,
    origins: Sequence[int],
    residual_predictions: np.ndarray,
) -> np.ndarray:
    target_values = np.asarray(target_values, dtype=float)
    origins = np.asarray(origins, dtype=int)
    residual_predictions = np.asarray(residual_predictions, dtype=float)
    return target_values[origins][:, None] + residual_predictions


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
