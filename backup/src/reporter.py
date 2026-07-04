"""
reporter.py — Gerador de Relatório de Experimento v2.0
=======================================================
Implementa:
  - RF-10: Gera relatorio_experimento.md automaticamente ao final de cada run.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def generate_report(
    config: Dict,
    history: Dict[str, List[float]],
    class_names: List[str],
    y_true: Optional[List[int]],
    y_pred: Optional[List[int]],
    output_dir: str,
    checkpoint_path: Optional[str] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> str:
    """Gera um relatório Markdown completo do experimento.

    O relatório contém configuração de hiperparâmetros, tabela de métricas por
    época, métricas finais por classe (se y_true/y_pred fornecidos), referências
    aos plots gerados e timestamps de execução.

    Args:
        config: Dicionário com todos os hiperparâmetros do experimento.
        history: Histórico de treino (train_loss, val_loss, train_acc, val_acc).
        class_names: Lista de nomes das classes do dataset.
        y_true: Rótulos reais do conjunto de teste (opcional).
        y_pred: Rótulos preditos pelo modelo (opcional).
        output_dir: Diretório de saída onde o relatório e os plots estão.
        checkpoint_path: Caminho do checkpoint .pth salvo (opcional).
        start_time: Timestamp de início do treinamento (time.time()).
        end_time: Timestamp de fim do treinamento (time.time()).

    Returns:
        Caminho absoluto do arquivo relatorio_experimento.md gerado.
    """
    from sklearn.metrics import classification_report

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    report_path = os.path.join(output_dir, "relatorio_experimento.md")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed_str = "N/A"
    if start_time and end_time:
        elapsed = end_time - start_time
        elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"

    lines = [
        "# Relatório de Experimento — Classificador de Padrões Socioterritoriais",
        "",
        f"> **Gerado em:** {now}  ",
        f"> **Tempo total de treinamento:** {elapsed_str}",
        "",
        "---",
        "",
        "## 1. Configuração do Experimento",
        "",
        "| Parâmetro | Valor |",
        "|-----------|-------|",
    ]
    for key, value in config.items():
        lines.append(f"| `{key}` | `{value}` |")

    lines += [
        "",
        "---",
        "",
        "## 2. Curva de Aprendizagem por Época",
        "",
        "| Época | Train Loss | Train Acc | Val Loss | Val Acc | LR |",
        "|-------|-----------|-----------|---------|---------|-----|",
    ]
    for i, (tl, ta, vl, va) in enumerate(zip(
        history.get("train_loss", []),
        history.get("train_acc", []),
        history.get("val_loss", []),
        history.get("val_acc", []),
    )):
        lr = history.get("lr", [None] * (i + 1))[i]
        lr_str = f"{lr:.6f}" if lr is not None else "—"
        lines.append(f"| {i + 1} | {tl:.4f} | {ta:.4f} | {vl:.4f} | {va:.4f} | {lr_str} |")

    best_val_acc = max(history.get("val_acc", [0.0]))
    best_epoch = history.get("val_acc", [0.0]).index(best_val_acc) + 1
    lines += [
        "",
        f"> **Melhor época:** {best_epoch} — Val Acc: **{best_val_acc:.4f}** ({best_val_acc * 100:.2f}%)",
        "",
    ]

    # ── Métricas por classe (se disponíveis) ──────────────────────────────────
    if y_true is not None and y_pred is not None:
        lines += [
            "---",
            "",
            "## 3. Métricas Finais por Classe (Conjunto de Teste)",
            "",
            "```",
            classification_report(y_true, y_pred, target_names=class_names, digits=4),
            "```",
            "",
        ]

    # ── Plots — lista todos os .png presentes em plots/ ────────────────────
    plots_dir = os.path.join(output_dir, "plots")
    lines += [
        "---",
        "",
        "## 4. Visualizações",
        "",
    ]
    if os.path.isdir(plots_dir):
        found_plots = sorted(Path(plots_dir).glob("*.png"))
        for plot_path in found_plots:
            plot_file = plot_path.name
            lines.append(f"![{plot_file}](plots/{plot_file})")
            lines.append("")
    if not os.path.isdir(plots_dir) or not list(Path(plots_dir).glob("*.png")):
        lines.append("> Nenhum plot disponível ainda.")
        lines.append("")

    # ── Checkpoint ────────────────────────────────────────────────────────────
    if checkpoint_path:
        lines += [
            "---",
            "",
            "## 5. Checkpoint Salvo",
            "",
            f"- **Arquivo:** `{checkpoint_path}`",
            "",
        ]

    # ── Configuração para reproduzir ─────────────────────────────────────────
    lines += [
        "---",
        "",
        "## 6. Como Reproduzir Este Experimento",
        "",
        "```bash",
        "python src/main.py \\",
    ]
    for key, value in config.items():
        lines.append(f"  --{key.replace('_', '-')} {value} \\")
    lines[-1] = lines[-1].rstrip(" \\")
    lines += ["```", ""]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("Relatório do experimento gerado em: %s", report_path)
    return report_path
