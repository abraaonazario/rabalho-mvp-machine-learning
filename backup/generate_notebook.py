import json
import os

def read_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def main():
    src_dir = os.path.join(os.path.dirname(__file__), 'src')
    
    # Lendo o conteúdo dos arquivos
    dataset_code = read_file(os.path.join(src_dir, 'dataset.py'))
    models_code = read_file(os.path.join(src_dir, 'models.py'))
    train_code = read_file(os.path.join(src_dir, 'train.py'))
    evaluate_code = read_file(os.path.join(src_dir, 'evaluate.py'))
    
    # Formatando para Colab
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Classificação de Padrões Socioterritoriais em Imagens de Satélite (EuroSAT)\n",
                "### Trabalho Final de Visão Computacional (Projeto DataLuta)\n",
                "---\n",
                "Este *notebook* apresenta a implementação ponta-a-ponta de Redes Neurais Convolucionais para a detecção de padrões de uso e ocupação do solo, com integração ao ecossistema analítico do projeto de Doutorado **DataLuta**.\n\n",
                "> ⚠️ **Instrução de Execução:** Para garantir um treinamento eficiente (especialmente com as arquiteturas ResNet e VGG), ative a aceleração por hardware. No menu superior, acesse `Ambiente de execução` > `Alterar tipo de ambiente de execução` e selecione o acelerador **GPU (T4)**."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Instalação de Dependências\n",
                "!pip install scikit-learn seaborn matplotlib torch torchvision tqdm pandas\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 2. Base de Dados e Pipeline (Dataset)\n", "Atendendo ao requisito do PDF de dividir treino e validação, carregamos a base EuroSAT com e sem Data Augmentation para comparar seu impacto."]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in dataset_code.split('\n')]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 3. Arquiteturas Distintas (Modelos)\n", "Atendendo ao requisito de 'Arquiteturas distintas', apresentamos três níveis de abstração: CNN pura, família VGG e família ResNet."]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in models_code.split('\n')]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 4. Otimização e Avaliação Rigorosa\n", "Aqui implementamos as métricas de Acurácia, F1-Score, Precisão e Recall e a plotagem da Matriz de Confusão, conforme cobrado pelo professor."]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in train_code.split('\n')] + ["\n"] + [line + "\n" for line in evaluate_code.split('\n')]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. EXPERIMENTO 1: Impacto dos Otimizadores e Arquitetura Básica\n",
                "Vamos avaliar a BaselineCNN usando o otimizador clássico SGD versus o adaptativo ADAM, sem aplicar data augmentation."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                "train_loader_no_aug, val_loader, test_loader, class_names = get_dataloaders(data_dir='./data', use_augmentation=False, batch_size=128)\n\n",
                "print('--- Gerando Mosaico de Exemplo do Dataset ---')\n",
                "show_dataset_samples(train_loader_no_aug, class_names)\n\n",
                "print('--- Experimento 1A: Baseline CNN com SGD ---')\n",
                "model_base_sgd = BaselineCNN(num_classes=10)\n",
                "hist_base_sgd = train_model(model_base_sgd, train_loader_no_aug, val_loader, optimizer_type='sgd', epochs=5, device=device)\n",
                "plot_training_curves(hist_base_sgd, 'Baseline SGD')\n\n",
                "print('--- Experimento 1B: Baseline CNN com ADAM ---')\n",
                "model_base_adam = BaselineCNN(num_classes=10)\n",
                "hist_base_adam = train_model(model_base_adam, train_loader_no_aug, val_loader, optimizer_type='adam', epochs=5, device=device)\n",
                "plot_training_curves(hist_base_adam, 'Baseline ADAM')\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. EXPERIMENTO 2: Impacto da Arquitetura e Data Augmentation (Modelo Final)\n",
                "Sabendo que o ADAM é superior, aplicaremos a rede avançada SmallResNet aliada a rotações e flips de Satélite (Data Augmentation), gerando as métricas finais (Acurácia Global, Macro e F1) com 15 épocas."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "train_loader_aug, _, _, _ = get_dataloaders(data_dir='./data', use_augmentation=True, batch_size=128)\n\n",
                "print('--- Experimento Final: SmallResNet com ADAM e Data Augmentation ---')\n",
                "model_resnet = SmallResNet(num_classes=10)\n",
                "hist_resnet = train_model(model_resnet, train_loader_aug, val_loader, optimizer_type='adam', epochs=15, device=device)\n",
                "plot_training_curves(hist_resnet, 'SmallResNet ADAM + Augmentation')\n\n",
                "# Matriz de Confusão e Relatório Completo\n",
                "print('\\nGERANDO RELATÓRIO DO EXPERIMENTO FINAL E MATRIZ DE CONFUSÃO:\\n')\n",
                "y_true, y_pred = evaluate_model(model_resnet, test_loader, class_names, device=device)\n",
                "plot_confusion_matrix(y_true, y_pred, class_names)\n",
                "print('\\nDIAGNÓSTICO VISUAL DE ACERTOS E ERROS NO TEST SET:\\n')\n",
                "visualize_predictions(model_resnet, test_loader, class_names, device=device)\n"
            ]
        }
    ]
    
    notebook = {
        "cells": cells,
        "metadata": {
            "colab": {
                "name": "Trabalho_Final_Satelite_Rigido_V1.ipynb",
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    os.makedirs(os.path.join(os.path.dirname(__file__), 'notebooks'), exist_ok=True)
    out_path = os.path.join(os.path.dirname(__file__), 'notebooks', 'Trabalho_Final_Satelite_Rigido_V1.ipynb')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
        
    print(f"Notebook rigoroso gerado com sucesso em: {out_path}")

if __name__ == '__main__':
    main()
