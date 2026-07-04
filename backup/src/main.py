"""
main.py — Ponto de entrada principal v2.0
==========================================
Implementa:
  - RF-04: CLI com argparse (11 argumentos configuráveis)
  - RF-05: Logging estruturado em arquivo + console
  - RF-06: set_seed() para reprodutibilidade global (CPU + GPU)
  - RF-10: Geração automática de relatório Markdown ao final
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Adiciona a raiz do projeto ao path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.dataset import get_dataloaders, _load_normalization_stats
from src.evaluate import (
    evaluate_model,
    plot_confusion_matrix,
    plot_training_curves,
    visualize_predictions,
)
from src.models import BaselineCNN, SmallResNet, SmallVGG
from src.train import train_model
from src import reporter

_ARCHITECTURE_MAP = {
    "baseline": BaselineCNN,
    "vgg": SmallVGG,
    "resnet": SmallResNet,
}

_ARCHITECTURE_NAMES = {
    "baseline": "BaselineCNN",
    "vgg": "SmallVGG",
    "resnet": "SmallResNet",
}


# ── RF-06: Reprodutibilidade Global ───────────────────────────────────────────
def set_seed(seed: int) -> None:
    """Configura semente global para reprodutibilidade em CPU e GPU.

    Define seeds para os módulos: random, numpy, torch (CPU), torch.cuda (GPU),
    e ativa o modo determinístico do cuDNN.

    Args:
        seed: Valor inteiro da semente aleatória.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.getLogger(__name__).info("Semente global configurada: seed=%d", seed)


# ── RF-05: Configuração de Logging ────────────────────────────────────────────
def setup_logging(log_path: str) -> None:
    """Configura o sistema de logging com handlers para arquivo e console.

    O nível padrão é INFO. Logs aparecem simultaneamente no terminal e no
    arquivo especificado por `log_path`.

    Args:
        log_path: Caminho completo do arquivo .log de saída.
    """
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s — %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )


# ── RF-04: CLI ────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    """Configura e retorna os argumentos de linha de comando.

    Returns:
        Namespace com todos os argumentos parseados.
    """
    parser = argparse.ArgumentParser(
        description="Classificador de Padrões Socioterritoriais — DataLuta v2.0",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Arquitetura e treinamento
    parser.add_argument("--model", default="resnet", choices=["baseline", "vgg", "resnet"],
                        help="Arquitetura CNN a ser treinada.")
    parser.add_argument("--optimizer", default="adam", choices=["sgd", "momentum", "adam"],
                        help="Algoritmo de otimização.")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Número máximo de épocas de treinamento.")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Tamanho do mini-batch.")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate inicial.")
    parser.add_argument("--augmentation", action="store_true", default=True,
                        help="Ativar Data Augmentation no treino.")
    parser.add_argument("--no-augmentation", dest="augmentation", action="store_false",
                        help="Desativar Data Augmentation no treino.")
    parser.add_argument("--scheduler", default="plateau", choices=["none", "plateau", "cosine"],
                        help="Learning Rate Scheduler.")
    parser.add_argument("--patience", type=int, default=10,
                        help="Épocas de tolerância para Early Stopping.")
    parser.add_argument("--num-blocks", type=int, default=2,
                        help="Número de blocos residuais por camada (SmallResNet).")

    # Caminhos
    parser.add_argument("--data-dir", default="./data",
                        help="Diretório raiz do dataset EuroSAT.")
    parser.add_argument("--output-dir", default="./outputs",
                        help="Diretório raiz para logs, métricas e plots.")
    parser.add_argument("--checkpoint-dir", default="./checkpoints",
                        help="Diretório para salvar checkpoints .pth.")

    # Reprodutibilidade
    parser.add_argument("--seed", type=int, default=42,
                        help="Semente aleatória global.")

    return parser.parse_args()


# ── Pipeline Principal ────────────────────────────────────────────────────────
def main() -> None:
    """Pipeline completo de treinamento e avaliação — DataLuta v2.0.

    Fluxo:
        1. Parsing de argumentos CLI (RF-04)
        2. Configuração de logging e diretórios de saída (RF-05)
        3. Reprodutibilidade global via set_seed (RF-06)
        4. Carregamento do dataset com split estratificado
        5. Instanciação da arquitetura selecionada
        6. Treinamento com early stopping e scheduler (RF-01, RF-03)
        7. Salvamento de checkpoint (RF-02)
        8. Avaliação no conjunto de teste
        9. Geração de plots e relatório Markdown (RF-10)
    """
    args = parse_args()

    # ── Criação do run_id e diretórios ───────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{args.model}_{args.optimizer}"
    run_dir = os.path.join(args.output_dir, run_id)
    plots_dir = os.path.join(run_dir, "plots")
    log_path = os.path.join(run_dir, "training.log")
    metrics_path = os.path.join(run_dir, "metrics.csv")

    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    # ── Logging (RF-05) ──────────────────────────────────────────────────────
    setup_logging(log_path)
    logger = logging.getLogger(__name__)
    logger.info("=== DataLuta v2.0 — Classificador de Padrões Socioterritoriais ===")
    logger.info("Run ID: %s", run_id)

    # ── Salvar config.json (RF-05) ───────────────────────────────────────────
    config = {
        "model": args.model,
        "optimizer": args.optimizer,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "augmentation": args.augmentation,
        "scheduler": args.scheduler,
        "patience": args.patience,
        "num_blocks": args.num_blocks,
        "seed": args.seed,
        "data_dir": args.data_dir,
        "run_id": run_id,
    }
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    logger.info("Configuração salva em: %s", config_path)

    # ── Reprodutibilidade (RF-06) ────────────────────────────────────────────
    set_seed(args.seed)

    # ── Dispositivo ──────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Dispositivo de processamento: %s", device)

    # ── [Passo 1] Dataset ────────────────────────────────────────────────────
    logger.info("[Passo 1] Carregando dataset EuroSAT...")
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        use_augmentation=args.augmentation,
    )
    # Recupera as estatísticas de normalização usadas (para persistir no checkpoint)
    norm_mean, norm_std = _load_normalization_stats(args.data_dir)
    logger.info("Classes identificadas (%d): %s", len(class_names), class_names)

    # ── [Passo 2] Modelo ──────────────────────────────────────────────────────
    logger.info("[Passo 2] Instanciando arquitetura: %s", args.model.upper())
    model_cls = _ARCHITECTURE_MAP[args.model]
    architecture_name = _ARCHITECTURE_NAMES[args.model]

    if args.model == "resnet":
        model = model_cls(num_classes=len(class_names), num_blocks=args.num_blocks)
        logger.info("SmallResNet: num_blocks=%d por camada (total blocos: %d)", args.num_blocks, args.num_blocks * 3)
    else:
        model = model_cls(num_classes=len(class_names))

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Parâmetros treináveis: %s", f"{n_params:,}")

    # ── [Passo 3] Treinamento ─────────────────────────────────────────────────
    logger.info("[Passo 3] Iniciando treinamento...")
    start_time = time.time()

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer_type=args.optimizer,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        patience=args.patience,
        scheduler_type=args.scheduler,
        class_names=class_names,
        architecture=architecture_name,
        checkpoint_dir=args.checkpoint_dir,
        metrics_path=metrics_path,
        norm_mean=norm_mean,
        norm_std=norm_std,
    )

    end_time = time.time()

    # ── [Passo 4] Avaliação ───────────────────────────────────────────────────
    logger.info("[Passo 4] Avaliando no conjunto de teste...")
    y_true, y_pred = evaluate_model(model, test_loader, class_names, device=device)

    # ── [Passo 5] Plots ───────────────────────────────────────────────────────
    logger.info("[Passo 5] Gerando visualizações...")
    plot_training_curves(history, title_suffix=f"{architecture_name} ({args.optimizer})", output_dir=plots_dir)
    plot_confusion_matrix(y_true, y_pred, class_names, output_dir=plots_dir)

    try:
        visualize_predictions(model, val_loader, class_names, output_dir=plots_dir, device=device)
    except Exception as exc:
        logger.warning("Não foi possível gerar diagnóstico visual: %s", exc)

    # ── Encontrar checkpoint gerado ───────────────────────────────────────────
    checkpoint_dir_path = Path(args.checkpoint_dir)
    checkpoint_files = sorted(checkpoint_dir_path.glob(f"{architecture_name}_*.pth"), key=os.path.getmtime)
    checkpoint_path = str(checkpoint_files[-1]) if checkpoint_files else None

    # ── [Passo 6] Relatório Markdown (RF-10) ─────────────────────────────────
    logger.info("[Passo 6] Gerando relatório do experimento...")
    reporter.generate_report(
        config=config,
        history=history,
        class_names=class_names,
        y_true=y_true,
        y_pred=y_pred,
        output_dir=run_dir,
        checkpoint_path=checkpoint_path,
        start_time=start_time,
        end_time=end_time,
    )

    logger.info("=== Pipeline concluído! Resultados em: %s ===", run_dir)


if __name__ == "__main__":
    main()
