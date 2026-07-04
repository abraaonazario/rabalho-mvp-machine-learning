# Trabalho Final - Visão Computacional (Classificação de Imagens de Satélite)

Este repositório contém os entregáveis referentes ao trabalho final da disciplina de Visão Computacional. O projeto prático implementa arquiteturas de Deep Learning (Redes Neurais Convolucionais treinadas do zero) para a classificação de padrões socioterritoriais no dataset EuroSAT.

## Organização dos Arquivos

* `Relatorio_Final_Satelite.md` / `.pdf`: Relatório técnico detalhando toda a metodologia, experimentos e resultados.
* `Trabalho_Final_Satelite_Rigido_V1.ipynb - Colab.pdf`: Execução do Notebook base contendo o fluxo de ponta a ponta gerado no Google Colab.
* `src/`: Diretório contendo as implementações e rotinas complementares ao projeto (como `evaluate.py`).
* `requirements.txt`: Arquivo listando as bibliotecas essenciais de Python necessárias.
* Arquivos `.png`: Imagens de gráficos, curvas de aprendizado e amostras da base geradas pelos scripts e usadas no relatório.

## Instruções Básicas de Execução

Recomenda-se utilizar um ambiente virtual local (como `venv` ou `conda`) para testar ou executar os códigos do projeto, ou abri-los diretamente no **Google Colab** para aproveitar a aceleração via GPU.

### 1. Instalação das Dependências

Abra seu terminal na pasta raiz deste trabalho e instale as dependências usando o `pip`:

```bash
pip install -r requirements.txt
```

*(As bibliotecas principais incluem `torch`, `torchvision`, `scikit-learn`, `matplotlib`, entre outras).*

### 2. Execução dos Códigos

Para rodar as rotinas auxiliares de avaliação ou geração dos resultados:

```bash
# Exemplo de uso de scripts da pasta src
python src/evaluate.py
```

Ou acesse o código-fonte principal que foi encapsulado no Notebook e abra-o em um ambiente Jupyter.
