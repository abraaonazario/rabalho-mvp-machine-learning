# Minimum Viable Product (MVP) - Machine Learning & Analytics

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abraaonazario/rabalho-mvp-machine-learning/blob/main/MVP_CoberturaFlorestal_Colab.ipynb)

**Projeto: Classificação Multiclasse de Cobertura Florestal utilizando Variáveis Cartográficas e Ambientais**

Este repositório contém a infraestrutura de dados e a implementação computacional (Notebook) desenvolvida para a avaliação final da disciplina de Machine Learning & Analytics. O trabalho foi conduzido em alinhamento metodológico com a pesquisa de Doutorado intitulada **DataLuta**, cujo foco engloba o monitoramento socioterritorial e o uso do solo.

---

## 1. Contextualização e Objetivos
A identificação precisa da cobertura vegetal e do uso do solo possui implicações críticas no monitoramento ambiental, gestão de recursos hídricos e planejamento territorial. Tradicionalmente, este problema é abordado através do processamento de imagens de satélite (Visão Computacional). No entanto, este projeto investiga uma abordagem complementar: a modelagem preditiva utilizando exclusivamente variáveis cartográficas e ambientais subjacentes ao terreno.

**Objetivo Principal:**
Desenvolver, avaliar e otimizar um pipeline completo de Machine Learning para a classificação multiclasse do tipo de vegetação predominante em parcelas de terra (30x30 metros), minimizando a influência da assimetria distribucional (desbalanceamento de classes) na métrica de avaliação.

---

## 2. Descrição da Base de Dados
Foi utilizado o dataset acadêmico **Forest Cover Type** (Blackard & Dean, 1999), originalmente disponibilizado através do *UCI Machine Learning Repository*. Para viabilizar a execução integral do ciclo metodológico em ambientes de nuvem (e.g., Google Colab) mantendo as proporções estatísticas da distribuição original, os dados presentes neste repositório representam uma subamostra estratificada.

- **Registros:** 60.000 instâncias.
- **Atributos Preditivos:** 54 variáveis, divididas em:
  - 10 Variáveis Contínuas (ex: Elevação, Declividade, Aspecto, Distância Hidrológica).
  - 44 Variáveis Binárias (Indicadores de Áreas de Preservação e Classificações de Solo).
- **Variável-Alvo:** `Cover_Type`, consistindo em 7 classes discretas (Spruce/Fir, Lodgepole Pine, Ponderosa Pine, Cottonwood/Willow, Aspen, Douglas-fir, Krummholz).

---

## 3. Metodologia e Pipeline Analítico
A pesquisa seguiu o fluxo padronizado de extração de conhecimento em bases de dados (KDD):

### 3.1. Engenharia de Variáveis (Feature Engineering)
Em virtude da complexidade topográfica, novos estimadores sintéticos foram derivados para potencializar a separabilidade linear e não-linear, destacando-se a criação da métrica `Distance_To_Water` (Distância Euclidiana real combinando distâncias horizontais e verticais) e o isolamento do gradiente de altitude.

### 3.2. Pré-processamento e Mitigação de Vazamento de Dados (Data Leakage)
A conversão de escala das variáveis contínuas foi conduzida por meio da padronização (`StandardScaler`). Em rigor acadêmico, a transformação foi encapsulada em `Pipelines` do `scikit-learn`, assegurando que os parâmetros estatísticos fossem extraídos exclusivamente da porção de treinamento, prevenindo qualquer forma de viés avaliativo decorrente do vazamento de dados.

### 3.3. Algoritmos de Modelagem Preditiva
A seleção dos algoritmos balizou-se no avanço progressivo da complexidade de hipóteses:
1. **Dummy Classifier:** Estabelecimento do Limiar Base (Baseline Stratified).
2. **Logistic Regression:** Avaliação do grau de separabilidade linear multivariada.
3. **Decision Tree:** Introdução da segmentação não-linear e extração de regras de decisão (interpretabilidade).
4. **Random Forest:** Ensemble supervisionado por Bagging (variância reduzida).
5. **HistGradientBoosting:** Ensemble otimizado iterativo por Boosting, adequado para instâncias desbalanceadas.
6. **Perceptron Multicamadas (MLP):** Avaliação por Redes Neurais Densas, em sinergia exploratória com a arquitetura do projeto subsequente de Visão Computacional.

### 3.4. Otimização Estocástica de Hiperparâmetros
Foi conduzida uma busca aleatória computacionalmente eficiente (`RandomizedSearchCV`) sobre a estrutura da Random Forest, adotando Validação Cruzada (*3-fold Cross-Validation*). O foco primário foi o ajuste do hiperparâmetro de peso balanceado das classes (`class_weight`), mitigando o erro sobre as classes minoritárias (como Cottonwood/Willow).

### 3.5. Avaliação de Desempenho
Considerando que a Acurácia Global é uma métrica ilusória frente a distribuições assimétricas (onde a classe dominante infla estatisticamente a métrica preditiva global), o modelo foi formalmente otimizado e julgado por meio da métrica **F1-Score (Macro)**. Tal critério assegura que o acerto isolado em biomas minoritários possua peso metodológico equivalente ao das classes predominantes.

---

## 4. Instruções de Reprodutibilidade
Para assegurar a transparência científica e a replicabilidade integral deste experimento sem restrições de ambiente local, todo o código fonte foi desenhado para ser invocado na plataforma *Google Colab*.

**Passos para Execução:**
1. Realize o download do arquivo `MVP_CoberturaFlorestal_Colab.ipynb` deste repositório.
2. Acesse o ambiente [Google Colab](https://colab.research.google.com/) e efetue o upload do referido Notebook.
3. Não há necessidade de importar os dados manualmente. O Notebook fará o apontamento e a extração automatizada da amostra `forest_cover_sample.csv` hospedada em seu link "Raw" nativo neste mesmo repositório do GitHub.
4. Navegue até o menu superior e selecione `Ambiente de Execução` > `Executar tudo`.
5. Todos os resultados, curvas de treinamento e a matriz de confusão normalizada serão consolidados progressivamente ao final do log de execução.

---

**Autor:** Abraão Gualberto Nazario  
**Contexto de Pesquisa:** Doutorado - Projeto DataLuta  
**Disciplina:** Machine Learning & Analytics  
**Linguagem de Implementação:** Python 3.10+
