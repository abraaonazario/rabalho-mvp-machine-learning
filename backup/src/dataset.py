"""
dataset.py — Módulo de carregamento de dados v2.0
===================================================
Implementa:
  - RF-07: Carrega estatísticas de normalização do EuroSAT de ./data/eurosat_stats.json
           com fallback para valores ImageNet se o arquivo não existir.
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)

# Estatísticas padrão ImageNet (fallback) — RGB aproximado
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

# Estatísticas calibradas EuroSAT RGB (referência — RF-07)
_EUROSAT_MEAN_REF = [0.3444, 0.3803, 0.4078]
_EUROSAT_STD_REF = [0.2027, 0.1369, 0.1150]

SEED = 42


def _load_normalization_stats(data_dir: str) -> Tuple[List[float], List[float]]:
    """Carrega estatísticas de normalização do EuroSAT, com fallback para ImageNet.

    Procura pelo arquivo ``eurosat_stats.json`` dentro de ``data_dir``.
    Se encontrado, usa os valores calculados sobre o dataset real (RF-07).
    Caso contrário, usa as estatísticas genéricas do ImageNet.

    Args:
        data_dir: Diretório raiz dos dados onde ``eurosat_stats.json`` pode estar.

    Returns:
        Tupla (mean, std) com listas de 3 floats (canais R, G, B).
    """
    stats_path = Path(data_dir) / "eurosat_stats.json"
    if stats_path.exists():
        try:
            with open(stats_path, "r", encoding="utf-8") as f:
                stats = json.load(f)
            mean = stats["mean"]
            std = stats["std"]
            logger.info("Estatísticas EuroSAT carregadas de %s | Mean=%s | Std=%s", stats_path, mean, std)
            return mean, std
        except (KeyError, json.JSONDecodeError) as exc:
            logger.warning("Falha ao ler %s (%s). Usando fallback ImageNet.", stats_path, exc)

    logger.info("eurosat_stats.json não encontrado. Usando estatísticas ImageNet como fallback.")
    return _IMAGENET_MEAN, _IMAGENET_STD


class DatasetWrapper(torch.utils.data.Dataset):
    """Wrapper para aplicar transformações específicas em subsets (treino ou validação).

    Args:
        subset: Subset do dataset base.
        transform: Pipeline de transformações torchvision a ser aplicado.
    """

    def __init__(self, subset: Subset, transform: Optional[object] = None) -> None:
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Retorna amostra e rótulo com transformação aplicada."""
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self) -> int:
        """Retorna o número de amostras no subset."""
        return len(self.subset)


def get_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 64,
    use_augmentation: bool = False,
    use_sampler: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """Carrega o dataset EuroSAT com split estratificado 70/15/15.

    Aplica normalização calibrada para o EuroSAT (RF-07), com fallback automático
    para as estatísticas do ImageNet se o arquivo ``eurosat_stats.json`` não for
    encontrado em ``data_dir``.

    Args:
        data_dir: Diretório raiz para download e cache do EuroSAT.
        batch_size: Tamanho do mini-batch para todos os loaders.
        use_augmentation: Se True, aplica Data Augmentation no treino.
        use_sampler: Se True, usa WeightedRandomSampler para rebalancear classes.

    Returns:
        Tupla (train_loader, val_loader, test_loader, class_names).

    Raises:
        RuntimeError: Se o download ou leitura do dataset falhar.
    """
    # ── Estatísticas de normalização (RF-07) ──────────────────────────────────
    mean, std = _load_normalization_stats(data_dir)

    # ── Transformações de Validação/Teste ─────────────────────────────────────
    val_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # ── Transformações de Treino ──────────────────────────────────────────────
    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(360),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        train_transform = val_transform

    # ── Download e carregamento do EuroSAT ───────────────────────────────────
    logger.info("Carregando dataset EuroSAT de '%s'...", data_dir)
    base_dataset = datasets.EuroSAT(root=data_dir, download=True)
    labels = [target for _, target in base_dataset.samples]

    # ── Split estratificado 70/15/15 com seed fixo ────────────────────────────
    train_idx, temp_idx = train_test_split(
        np.arange(len(base_dataset)),
        test_size=0.30,
        random_state=SEED,
        stratify=labels,
    )
    temp_labels = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_labels,
    )

    logger.info(
        "Split: %d treino | %d validação | %d teste (total: %d)",
        len(train_idx), len(val_idx), len(test_idx), len(base_dataset),
    )

    # ── Aplicar transformações específicas por split ───────────────────────────
    train_dataset = DatasetWrapper(Subset(base_dataset, train_idx), transform=train_transform)
    val_dataset = DatasetWrapper(Subset(base_dataset, val_idx), transform=val_transform)
    test_dataset = DatasetWrapper(Subset(base_dataset, test_idx), transform=val_transform)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    if use_sampler:
        train_labels = [labels[i] for i in train_idx]
        class_sample_count = np.array([
            len(np.where(np.array(train_labels) == t)[0]) for t in np.unique(train_labels)
        ])
        weight = 1.0 / class_sample_count
        samples_weight = torch.from_numpy(np.array([weight[t] for t in train_labels])).double()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=2)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, base_dataset.classes
