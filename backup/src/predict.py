"""
predict.py — Inferência em imagens avulsas v2.0
================================================
Implementa:
  - RF-09: CLI para inferência com --checkpoint, --image, --topk
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Adiciona a raiz do projeto ao sys.path para importações relativas
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models import BaselineCNN, SmallResNet, SmallVGG

logger = logging.getLogger(__name__)

# Estatísticas de normalização padrão (fallback ImageNet)
_DEFAULT_MEAN = [0.485, 0.456, 0.406]
_DEFAULT_STD = [0.229, 0.224, 0.225]

_ARCHITECTURE_MAP = {
    "BaselineCNN": BaselineCNN,
    "SmallVGG": SmallVGG,
    "SmallResNet": SmallResNet,
}


def _load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple:
    """Carrega o modelo e metadados a partir de um checkpoint .pth.

    Args:
        checkpoint_path: Caminho para o arquivo .pth.
        device: Dispositivo de destino.

    Returns:
        Tupla (model, class_names, checkpoint_dict).

    Raises:
        FileNotFoundError: Se o checkpoint não existir.
        ValueError: Se a arquitetura no checkpoint não for reconhecida.
    """
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint não encontrado: '{checkpoint_path}'")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    architecture = checkpoint.get("architecture", "SmallResNet")
    class_names: List[str] = checkpoint.get("class_names", [])
    num_classes = len(class_names) if class_names else 10

    if architecture not in _ARCHITECTURE_MAP:
        raise ValueError(
            f"Arquitetura '{architecture}' não reconhecida. "
            f"Disponíveis: {list(_ARCHITECTURE_MAP.keys())}"
        )

    model_cls = _ARCHITECTURE_MAP[architecture]
    model = model_cls(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info("Modelo '%s' carregado | Classes: %d | Época: %d | Val Acc: %.4f",
                architecture, num_classes, checkpoint.get("epoch", -1), checkpoint.get("val_acc", float("nan")))
    return model, class_names, checkpoint


def _preprocess_image(image_path: str, mean: List[float], std: List[float]) -> torch.Tensor:
    """Carrega e pré-processa uma imagem para inferência.

    Redimensiona para 64×64, converte para tensor e normaliza com os parâmetros
    fornecidos, adicionando dimensão de batch (1, C, H, W).

    Args:
        image_path: Caminho para o arquivo de imagem (JPEG ou PNG).
        mean: Média de normalização por canal.
        std: Desvio padrão de normalização por canal.

    Returns:
        Tensor com shape (1, 3, 64, 64).

    Raises:
        FileNotFoundError: Se a imagem não existir.
        OSError: Se o arquivo não for uma imagem válida.
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Imagem não encontrada: '{image_path}'")

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)


def predict(
    checkpoint_path: str,
    image_path: str,
    topk: int = 3,
    device_str: str = "auto",
) -> List[Tuple[str, float]]:
    """Executa inferência em uma imagem e retorna as Top-K previsões.

    Args:
        checkpoint_path: Caminho para o arquivo .pth do checkpoint.
        image_path: Caminho para a imagem de entrada.
        topk: Número de previsões alternativas a retornar.
        device_str: 'cuda', 'cpu' ou 'auto' (detecta automaticamente).

    Returns:
        Lista de tuplas (classe, probabilidade) ordenadas da maior para menor.
    """
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    model, class_names, checkpoint = _load_model_from_checkpoint(checkpoint_path, device)

    mean = checkpoint.get("mean", _DEFAULT_MEAN)
    std = checkpoint.get("std", _DEFAULT_STD)
    tensor = _preprocess_image(image_path, mean, std).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1).squeeze(0)

    k = min(topk, len(class_names))
    top_probs, top_indices = torch.topk(probs, k)

    results = [
        (class_names[idx.item()] if class_names else f"Classe {idx.item()}", prob.item())
        for idx, prob in zip(top_indices, top_probs)
    ]
    return results


def _parse_args() -> argparse.Namespace:
    """Configura e retorna os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Inferência em imagem avulsa usando checkpoint treinado — DataLuta v2.0",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Caminho para o arquivo .pth do checkpoint.")
    parser.add_argument("--image", required=True, help="Caminho para a imagem de entrada (JPEG ou PNG).")
    parser.add_argument("--topk", type=int, default=3, help="Número de previsões Top-K a exibir.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                        help="Dispositivo de computação.")
    return parser.parse_args()


def main() -> None:
    """Ponto de entrada principal do script de inferência."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()

    try:
        results = predict(
            checkpoint_path=args.checkpoint,
            image_path=args.image,
            topk=args.topk,
            device_str=args.device,
        )
    except (FileNotFoundError, ValueError, KeyError) as exc:
        logger.error("Erro na inferência: %s", exc)
        sys.exit(1)

    print("\n" + "=" * 50)
    print(f"  Classificação de Imagem: {args.image}")
    print("=" * 50)
    for rank, (class_name, prob) in enumerate(results, 1):
        marker = "🏆" if rank == 1 else f"  #{rank}"
        print(f"  {marker}  {class_name:<30s} {prob * 100:>6.2f}%")
    print("=" * 50 + "\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
