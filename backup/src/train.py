"""
train.py — Módulo de treinamento v2.0
======================================
Implementa:
  - RF-01: Early Stopping com patience e restauração do melhor estado
  - RF-02: Salvamento de checkpoint .pth em ./checkpoints/
  - RF-03: Learning Rate Scheduler (none | plateau | cosine)
  - RF-05: Logging estruturado + CSV de métricas por época
"""

import csv
import copy
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    """Retorna o learning rate atual do primeiro grupo de parâmetros.

    Args:
        optimizer: Otimizador PyTorch.

    Returns:
        Learning rate atual como float.
    """
    return optimizer.param_groups[0]["lr"]


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_acc: float,
    val_loss: float,
    class_names: List[str],
    architecture: str,
    output_dir: str = "./checkpoints",
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
) -> str:
    """Salva um checkpoint completo do modelo em disco.

    Args:
        model: Modelo treinado.
        optimizer: Otimizador com estado atual.
        epoch: Época em que o checkpoint foi salvo.
        val_acc: Acurácia de validação nessa época.
        val_loss: Perda de validação nessa época.
        class_names: Lista de nomes das classes.
        architecture: Nome da arquitetura (ex.: 'SmallResNet').
        output_dir: Diretório de destino para o arquivo .pth.
        mean: Média de normalização RGB usada no treinamento (RF-07).
        std: Desvio padrão de normalização RGB usada no treinamento (RF-07).

    Returns:
        Caminho absoluto do arquivo salvo.
    """
    # Estatísticas ImageNet como fallback se não fornecidas
    _fallback_mean = [0.485, 0.456, 0.406]
    _fallback_std = [0.229, 0.224, 0.225]

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filename = f"{architecture}_{optimizer.__class__.__name__.lower()}_acc{val_acc * 100:.2f}.pth"
    filepath = os.path.join(output_dir, filename)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc": val_acc,
        "val_loss": val_loss,
        "class_names": class_names,
        "architecture": architecture,
        "mean": mean if mean is not None else _fallback_mean,
        "std": std if std is not None else _fallback_std,
    }
    torch.save(checkpoint, filepath)
    logger.info("Checkpoint salvo em: %s", filepath)
    return filepath


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer_type: str = "adam",
    epochs: int = 30,
    lr: float = 0.001,
    device: Optional[torch.device] = None,
    patience: int = 10,
    scheduler_type: str = "plateau",
    class_names: Optional[List[str]] = None,
    architecture: str = "model",
    checkpoint_dir: str = "./checkpoints",
    metrics_path: Optional[str] = None,
    norm_mean: Optional[List[float]] = None,
    norm_std: Optional[List[float]] = None,
) -> Dict[str, List[float]]:
    """Treina o modelo com early stopping, scheduler e salvamento de checkpoint.

    Args:
        model: Arquitetura a ser treinada.
        train_loader: DataLoader de treinamento.
        val_loader: DataLoader de validação.
        optimizer_type: 'sgd', 'momentum' ou 'adam'.
        epochs: Número máximo de épocas de treinamento.
        lr: Learning rate inicial.
        device: Dispositivo de computação ('cuda' ou 'cpu').
        patience: Épocas sem melhoria de val_loss antes de parar (RF-01).
        scheduler_type: 'none', 'plateau' ou 'cosine' (RF-03).
        class_names: Lista de classes para o checkpoint (RF-02).
        architecture: Nome da arquitetura para nomear o checkpoint (RF-02).
        checkpoint_dir: Diretório para salvar o .pth (RF-02).
        metrics_path: Caminho do arquivo CSV para salvar métricas por época (RF-05).
        norm_mean: Média de normalização RGB usada no treinamento (RF-07), persistida no checkpoint.
        norm_std: Desvio padrão de normalização RGB usada no treinamento (RF-07), persistida no checkpoint.

    Returns:
        history: Dicionário com listas de métricas por época (train_loss, train_acc,
                 val_loss, val_acc, lr).

    Raises:
        ValueError: Se optimizer_type ou scheduler_type forem inválidos.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # ── Otimizador ────────────────────────────────────────────────────────────
    if optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer_type == "momentum":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Otimizador desconhecido: '{optimizer_type}'. Use 'sgd', 'momentum' ou 'adam'.")

    # ── Scheduler (RF-03) ─────────────────────────────────────────────────────
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
    if scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
    elif scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type != "none":
        raise ValueError(f"Scheduler desconhecido: '{scheduler_type}'. Use 'none', 'plateau' ou 'cosine'.")

    # ── Histórico e métricas ──────────────────────────────────────────────────
    history: Dict[str, List[float]] = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "lr": [],
    }

    # ── CSV de métricas (RF-05) ───────────────────────────────────────────────
    csv_file = None
    csv_writer = None
    if metrics_path:
        Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(metrics_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

    # ── Early Stopping (RF-01) ────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_model_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    best_epoch = 0

    logger.info(
        "Iniciando treinamento | Otimizador: %s | Épocas máx: %d | Patience: %d | Scheduler: %s | Device: %s",
        optimizer_type.upper(), epochs, patience, scheduler_type, device,
    )
    start_time = time.time()

    try:
        for epoch in range(epochs):
            # ── Fase de Treinamento ───────────────────────────────────────────
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]", leave=False)
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            epoch_train_loss = running_loss / total_train
            epoch_train_acc = correct_train / total_train

            # ── Fase de Validação ─────────────────────────────────────────────
            model.eval()
            running_val_loss = 0.0
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    running_val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

            epoch_val_loss = running_val_loss / total_val
            epoch_val_acc = correct_val / total_val
            current_lr = _get_current_lr(optimizer)

            # ── Scheduler step (RF-03) ────────────────────────────────────────
            if scheduler is not None:
                if scheduler_type == "plateau":
                    scheduler.step(epoch_val_loss)
                else:
                    scheduler.step()

            # ── Registrar histórico ───────────────────────────────────────────
            history["train_loss"].append(epoch_train_loss)
            history["train_acc"].append(epoch_train_acc)
            history["val_loss"].append(epoch_val_loss)
            history["val_acc"].append(epoch_val_acc)
            history["lr"].append(current_lr)

            if csv_writer:
                csv_writer.writerow([
                    epoch + 1,
                    f"{epoch_train_loss:.6f}", f"{epoch_train_acc:.6f}",
                    f"{epoch_val_loss:.6f}", f"{epoch_val_acc:.6f}",
                    f"{current_lr:.8f}",
                ])
                csv_file.flush()

            logger.info(
                "Epoch %d/%d | Train Loss: %.4f | Train Acc: %.4f | "
                "Val Loss: %.4f | Val Acc: %.4f | LR: %.6f",
                epoch + 1, epochs,
                epoch_train_loss, epoch_train_acc,
                epoch_val_loss, epoch_val_acc,
                current_lr,
            )

            # ── Early Stopping (RF-01) ────────────────────────────────────────
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(
                        "Early stopping ativado na época %d. Melhor época: %d (val_loss=%.4f).",
                        epoch + 1, best_epoch, best_val_loss,
                    )
                    break

    finally:
        if csv_file:
            csv_file.close()

    # ── Restaurar melhor estado (RF-01) ───────────────────────────────────────
    model.load_state_dict(best_model_state)
    logger.info("Melhor estado restaurado (época %d, val_loss=%.4f).", best_epoch, best_val_loss)

    # ── Salvar checkpoint (RF-02) ─────────────────────────────────────────────
    best_val_acc = max(history["val_acc"]) if history["val_acc"] else 0.0
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=best_epoch,
        val_acc=best_val_acc,
        val_loss=best_val_loss,
        class_names=class_names or [],
        architecture=architecture,
        output_dir=checkpoint_dir,
        mean=norm_mean,
        std=norm_std,
    )

    elapsed = time.time() - start_time
    logger.info("Treinamento concluído em %.0fm %.0fs.", elapsed // 60, elapsed % 60)

    return history
