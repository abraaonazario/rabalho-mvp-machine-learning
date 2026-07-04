"""
evaluate.py — Módulo de avaliação v2.0
========================================
Implementa:
  - RF-02: load_checkpoint() para carregar modelo salvo sem re-treino
  - RF-10: Plots salvos em diretório configurável (output_dir/plots/)
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

logger = logging.getLogger(__name__)


def load_checkpoint(
    path: str,
    model: nn.Module,
    device: Optional[torch.device] = None,
) -> Dict:
    """Carrega um checkpoint salvo e injeta os pesos no modelo fornecido.

    Args:
        path: Caminho para o arquivo .pth do checkpoint.
        model: Instância do modelo com a arquitetura correspondente ao checkpoint.
        device: Dispositivo de destino para carregar os tensores.

    Returns:
        Dicionário com metadados do checkpoint (epoch, val_acc, val_loss,
        class_names, architecture).

    Raises:
        FileNotFoundError: Se o arquivo de checkpoint não existir.
        KeyError: Se o checkpoint não contiver 'model_state_dict'.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint não encontrado: '{path}'")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if "model_state_dict" not in checkpoint:
        raise KeyError(f"O arquivo '{path}' não contém 'model_state_dict'. Formato inválido.")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info(
        "Checkpoint carregado: %s | Arquitetura: %s | Época: %d | Val Acc: %.4f",
        path,
        checkpoint.get("architecture", "N/A"),
        checkpoint.get("epoch", -1),
        checkpoint.get("val_acc", float("nan")),
    )
    return checkpoint


def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    class_names: List[str],
    device: Optional[torch.device] = None,
) -> Tuple[List[int], List[int]]:
    """Avalia o modelo gerando métricas avançadas via Scikit-Learn.

    Exibe Acurácia Global, Precisão, Recall e F1-Score (macro-média) no log.

    Args:
        model: Modelo a ser avaliado (deve estar no modo eval).
        dataloader: DataLoader do conjunto de teste.
        class_names: Lista de nomes das classes para o relatório.
        device: Dispositivo de computação.

    Returns:
        Tupla (all_labels, all_preds) com listas de rótulos reais e preditos.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    macro_acc = accuracy_score(all_labels, all_preds)

    logger.info("=== Relatório de Classificação ===\n%s", report)
    logger.info("Acurácia Global: %.4f", macro_acc)

    return all_labels, all_preds


def plot_training_curves(
    history: Dict[str, List[float]],
    title_suffix: str = "",
    output_dir: str = ".",
) -> str:
    """Plota as curvas de Loss e Acurácia de treinamento e validação.

    Args:
        history: Dicionário com listas 'train_loss', 'val_loss', 'train_acc', 'val_acc'.
        title_suffix: Sufixo para os títulos dos gráficos.
        output_dir: Diretório onde a imagem será salva.

    Returns:
        Caminho do arquivo de imagem salvo.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history["train_loss"], "b-", label="Treino Loss")
    ax1.plot(epochs, history["val_loss"], "r-", label="Validação Loss")
    ax1.set_title(f"Função de Perda (Loss) — {title_suffix}")
    ax1.set_xlabel("Épocas")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, history["train_acc"], "b-", label="Treino Acurácia")
    ax2.plot(epochs, history["val_acc"], "r-", label="Validação Acurácia")
    ax2.set_title(f"Acurácia — {title_suffix}")
    ax2.set_xlabel("Épocas")
    ax2.set_ylabel("Acurácia")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    safe_name = title_suffix.replace(" ", "_").replace("+", "and").replace("/", "-")
    filepath = os.path.join(output_dir, f"{safe_name}_curves.png")
    plt.savefig(filepath, dpi=150)
    plt.close(fig)

    logger.info("Curvas de treinamento salvas em: %s", filepath)
    return filepath


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    output_dir: str = ".",
) -> str:
    """Renderiza a matriz de confusão normalizada usando Seaborn.

    Args:
        y_true: Lista de rótulos reais.
        y_pred: Lista de rótulos preditos.
        class_names: Nomes das classes para os eixos.
        output_dir: Diretório onde a imagem será salva.

    Returns:
        Caminho do arquivo de imagem salvo.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.title("Matriz de Confusão — Modelo Campeão")
    plt.ylabel("Classe Real")
    plt.xlabel("Classe Predita")
    plt.tight_layout()

    filepath = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(filepath, dpi=150)
    plt.close()

    logger.info("Matriz de confusão salva em: %s", filepath)
    return filepath


def show_dataset_samples(
    dataloader: torch.utils.data.DataLoader,
    class_names: List[str],
    output_dir: str = ".",
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
) -> str:
    """Salva um mosaico de amostras do dataset de treinamento.

    Args:
        dataloader: DataLoader de treinamento.
        class_names: Nomes das classes.
        output_dir: Diretório de saída.
        mean: Média usada na normalização (para desnormalizar a exibição).
        std: Desvio padrão usado na normalização.

    Returns:
        Caminho do arquivo de imagem salvo.
    """
    import torchvision

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    mean = mean or [0.485, 0.456, 0.406]
    std = std or [0.229, 0.224, 0.225]

    inputs, _ = next(iter(dataloader))
    out = torchvision.utils.make_grid(inputs[:8])
    out = out * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)

    plt.figure(figsize=(15, 5))
    plt.imshow(out.permute(1, 2, 0).clamp(0, 1).numpy())
    plt.title("Amostragem do Dataset EuroSAT")
    plt.axis("off")
    plt.tight_layout()

    filepath = os.path.join(output_dir, "mosaico_satelite.png")
    plt.savefig(filepath, dpi=150)
    plt.close()

    logger.info("Mosaico do dataset salvo em: %s", filepath)
    return filepath


def visualize_predictions(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    class_names: List[str],
    output_dir: str = ".",
    device: Optional[torch.device] = None,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
) -> str:
    """Salva diagnóstico visual de acertos e erros do modelo.

    Args:
        model: Modelo para inferência.
        dataloader: DataLoader de validação/teste.
        class_names: Nomes das classes.
        output_dir: Diretório de saída para a imagem.
        device: Dispositivo de computação.
        mean: Média para desnormalização visual.
        std: Desvio padrão para desnormalização visual.

    Returns:
        Caminho do arquivo de imagem salvo.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    mean = mean or [0.485, 0.456, 0.406]
    std = std or [0.229, 0.224, 0.225]

    model.to(device)
    model.eval()
    inputs, labels = next(iter(dataloader))
    inputs = inputs.to(device)
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    inputs = inputs.cpu()

    plt.figure(figsize=(16, 8))
    for i in range(min(8, len(inputs))):
        ax = plt.subplot(2, 4, i + 1)
        ax.axis("off")
        img = inputs[i] * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)
        plt.imshow(img.permute(1, 2, 0).clamp(0, 1).numpy())
        color = "green" if preds[i] == labels[i] else "red"
        ax.set_title(
            f"Real: {class_names[labels[i]]}\nPred: {class_names[preds[i].item()]}",
            color=color, fontsize=10,
        )
    plt.suptitle("Diagnóstico Visual: Acertos (verde) e Erros (vermelho)", fontsize=14)
    plt.tight_layout()

    filepath = os.path.join(output_dir, "diagnostico_visual.png")
    plt.savefig(filepath, dpi=150)
    plt.close()

    logger.info("Diagnóstico visual salvo em: %s", filepath)
    return filepath
