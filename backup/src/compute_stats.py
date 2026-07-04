"""
compute_stats.py — Cálculo de Estatísticas de Normalização do EuroSAT v2.0
============================================================================
Implementa:
  - RF-07: Computa média e desvio padrão RGB do conjunto de treinamento EuroSAT
           e salva em ./data/eurosat_stats.json.

Uso:
    python src/compute_stats.py --data-dir ./data --output ./data/eurosat_stats.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Adiciona a raiz ao path
sys.path.append(str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)
SEED = 42


def compute_mean_std(
    data_dir: str = "./data",
    output_path: str = "./data/eurosat_stats.json",
    batch_size: int = 256,
) -> dict:
    """Computa a média e desvio padrão por canal RGB do conjunto de treino EuroSAT.

    Utiliza o mesmo split estratificado 70/15/15 com seed=42 definido em dataset.py,
    garantindo que as estatísticas sejam calculadas apenas sobre o conjunto de
    treinamento (sem vazamento de informação do val/test).

    Args:
        data_dir: Diretório com o dataset EuroSAT.
        output_path: Caminho de destino para o arquivo JSON de saída.
        batch_size: Tamanho do batch para o cálculo iterativo.

    Returns:
        Dicionário {'mean': [R, G, B], 'std': [R, G, B]}.
    """
    logger.info("Carregando EuroSAT para cálculo de estatísticas...")

    # Transformação mínima: apenas ToTensor (sem normalização)
    base_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    base_dataset = datasets.EuroSAT(root=data_dir, download=True, transform=base_transform)
    labels = [target for _, target in base_dataset.samples]

    # Apenas o conjunto de treinamento (70%)
    train_idx, _ = train_test_split(
        np.arange(len(base_dataset)),
        test_size=0.30,
        random_state=SEED,
        stratify=labels,
    )
    train_subset = Subset(base_dataset, train_idx)
    loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False, num_workers=2)

    logger.info("Calculando média e desvio padrão sobre %d amostras de treino...", len(train_idx))

    # Algoritmo de Welford para estabilidade numérica (dois passes)
    mean = torch.zeros(3)
    for images, _ in tqdm(loader, desc="Passo 1/2 — Média"):
        mean += images.mean(dim=[0, 2, 3]) * images.size(0)
    mean /= len(train_subset)

    std = torch.zeros(3)
    for images, _ in tqdm(loader, desc="Passo 2/2 — Desvio Padrão"):
        std += ((images - mean.view(3, 1, 1)) ** 2).mean(dim=[0, 2, 3]) * images.size(0)
    std = torch.sqrt(std / len(train_subset))

    stats = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "n_samples": len(train_idx),
        "image_size": 64,
        "dataset": "EuroSAT-RGB",
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info("Estatísticas salvas em: %s", output_path)
    logger.info("  Mean: R=%.4f G=%.4f B=%.4f", *stats["mean"])
    logger.info("  Std:  R=%.4f G=%.4f B=%.4f", *stats["std"])
    return stats


def _parse_args() -> argparse.Namespace:
    """Configura os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Computa estatísticas de normalização do EuroSAT (RF-07)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", default="./data", help="Diretório do dataset EuroSAT.")
    parser.add_argument("--output", default="./data/eurosat_stats.json", help="Arquivo JSON de saída.")
    parser.add_argument("--batch-size", type=int, default=256, help="Tamanho do batch para o cálculo.")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()
    compute_mean_std(
        data_dir=args.data_dir,
        output_path=args.output,
        batch_size=args.batch_size,
    )
