# Roteiro de Apresentação: Trabalho Final de Visão Computacional
**Projeto:** Classificação de Padrões Socioterritoriais em Imagens de Satélite (EuroSAT)

> **Dica para a Apresentação:** Este roteiro foi estruturado para você montar seus slides no PowerPoint/Canva. Em cada slide, a seção **"No Slide"** indica o que você deve colar na tela (em tópicos ou imagens geradas no Colab). A seção **"O que falar"** é o seu roteiro verbal para explicar o slide ao Professor Wesley.

---

## 🛰️ Slide 1 – Introdução e O Problema de Pesquisa

**No Slide (O que colocar na tela):**
* **Título:** Classificação Socioterritorial via Imagens de Satélite
* **Objetivo:** Treinamento *from scratch* (do zero) de Redes Neurais Convolucionais para mapeamento de uso do solo.
* **A Conexão Prática:** Servir de motor visual aéreo para complementar a análise textual de processos jurídicos (Projeto DataLuta).

**O que falar para o professor:**
- "Olá a todos. O objetivo deste trabalho final foi desenvolver um classificador visual capaz de identificar automaticamente se uma determinada área geográfica corresponde a uso agrícola, desmatamento ou ocupação urbana."
- "A premissa metodológica exigida foi treinar arquiteturas convolucionais estritamente do zero, para validarmos a verdadeira capacidade de extração de características sem depender de redes pré-treinadas (transfer learning)."
- "O impacto prático disso é criar um 'olho satelital' que poderá ser acoplado ao framework do DataLuta, cruzando o que o satélite vê com o que os documentos jurídicos descrevem."

---

## 🌍 Slide 2 – Base de Dados (Dataset EuroSAT)

**No Slide (O que colocar na tela):**
* **Título:** Base de Dados EuroSAT
* **Tamanho:** 27.000 amostras anotadas.
* **Recorte de Classes (Foco Socioterritorial):**
  - Área de Agricultura (Annual / Permanent Crop)
  - Desmatamento / Pasto (Pasture)
  - Ocupação/Construções (Residential / Industrial)
  - Área Preservada (Forest / Herbaceous)
* **Imagem Sugerida:** *(Insira aqui a imagem `mosaico_satelite.png` gerada pelo seu código).*

**O que falar para o professor:**
- "Para esse treinamento, adotei como referencial canônico o dataset EuroSAT."
- "Elegemos categorias globais com um foco muito claro no mapeamento socioterritorial: precisamos separar o que é área preservada do que é agricultura agressiva ou desmatamento."
- "Como podem ver pelo mosaico extraído do nosso Dataloader, as texturas são complexas e exigem que a rede aprenda padrões muito além de simples blocos de cores."

---

## 🛠️ Slide 3 – Metodologia e Data Augmentation

**No Slide (O que colocar na tela):**
* **Título:** Metodologia e Preparação Tensorial
* **Resolução:** Redução para 64x64 pixels (simulação de baixa resolução operacional).
* **Data Augmentation Específico (A Grande Sacada):**
  - Rotação aleatória de 360º.
  - Espelhamentos (Flips Horizontais e Verticais).
  - Ruído Fotométrico (ColorJitter).

**O que falar para o professor:**
- "Em relação à preparação dos tensores, forcei a redução da malha para 64x64 pixels para provar que a topologia sobrevive a baixas resoluções operacionais."
- "Mas a grande sacada de Visão Computacional aqui está no **Data Augmentation**. Diferente do Trabalho 1, onde placas de trânsito perdem o sentido se rotacionadas ou espelhadas, imagens de satélite possuem *invariância geométrica integral*."
- "Olhar um campo de soja de ponta-cabeça continua sendo um campo de soja. Por isso, forcei rotações de 360 graus e espelhamentos onidirecionais para robustecer os pesos da rede."

---

## 🏗️ Slide 4 – Progressão de Arquiteturas Convolucionais

**No Slide (O que colocar na tela):**
* **Título:** As 3 Topologias Avaliadas
* **1. Baseline CNN:** Piso referencial de extração.
* **2. SmallVGG:** Empilhamento profundo e convoluções duplas (captura de texturas complexas).
* **3. SmallResNet (O Estado da Arte):** Blocos residuais (*skip connections*) + Global Average Pooling.

**O que falar para o professor:**
- "Para não apenas rodar código, mas avaliar como a abstração ocorre, desenvolvi três arquiteturas em complexidade crescente."
- "Começamos com uma Baseline CNN simples, passamos por uma SmallVGG com blocos convolucionais duplos para buscar espectros mais profundos de textura."
- "E finalmente escalamos para a SmallResNet, introduzindo conexões residuais que nos permitiram mitigar o desvanecimento do gradiente e extrair o limite teto de aprendizagem desses 64 pixels."

---

## 🧪 Slide 5 – Experimento 1: A Luta no Loss Landscape

**No Slide (O que colocar na tela):**
* **Título:** Exp 1 - Otimizadores (SGD x Adam)
* **SGD Puro:** Aprendizado engessado, estagnação precoce.
* **SGD + Momentum:** Amortecimento direcional.
* **Adam:** Convergência drástica pelo cálculo adaptativo.
* **Imagem Sugerida:** *(Cole aqui o gráfico `Baseline_CNN_(Adam)_curves.png`)*.

**O que falar para o professor:**
- "No nosso primeiro experimento isolado, travei o Data Augmentation para observar puramente o comportamento da descida de gradiente."
- "Testamos o SGD primário, que oscilou e estagnou rapidamente na planície do loss. Ao aplicar o Momentum, ganhamos direção. Porém, ao saltarmos para o otimizador Adam com seu momento adaptativo coordenado, a convergência da rede base foi exponencialmente mais estável, como visto nesta curva."

---

## 🏆 Slide 6 – Modelo Campeão (SmallResNet) e Matriz de Confusão

**No Slide (O que colocar na tela):**
* **Título:** Resultados Finais: SmallResNet + Augmentation
* **Resultados Consolidados:** 
  - Acurácia Global: 91.26%
  - Macro Accuracy: 91.05%
* **Imagem Sugerida:** *(Cole aqui a imagem `confusion_matrix_resnet.png`)*.

**O que falar para o professor:**
- "No experimento final, combinamos a robustez da ResNet com o Data Augmentation estocástico extremo e o BatchNorm."
- "Os resultados atingiram 91.26% de Acurácia Global. Mas o que mais me interessa é a Matriz de Confusão."
- "A diagonal central ficou densa, mostrando excelente acerto. O mais interessante são os erros nos cruzamentos laterais: a rede costuma confundir 'Desmatamento Crônico' com 'Agricultura em período de safra baixa/solo nu', o que é uma confusão perfeitamente compreensível até para o olho humano, validando que a abstração cromática funcionou."

---

## 👁️ Slide 7 – Diagnóstico Visual de Predições

**No Slide (O que colocar na tela):**
* **Título:** Avaliação Qualitativa na Prática
* **Acertos:** Excelente separação de florestas virgens vs. culturas lineares.
* **Erros Comuns:** Fronteiras de transição cromática.
* **Imagem Sugerida:** *(Cole aqui a imagem `diagnostico_visual.png` com os Falsos Positivos e Falsos Negativos)*.

**O que falar para o professor:**
- "Extraímos exemplos do conjunto de validação para um diagnóstico visual. Aqui vemos exatamente onde a rede crava com confiança e onde ela sofre."
- "As lavouras (padrão estriado) e a floresta densa (padrão difuso verde) são categorizados com maestria."
- "As falhas ocorrem majoritariamente nas zonas de transição. Construções muito espalhadas parecem solo exposto, confundindo a rede."

---

## 🚀 Slide 8 – Conclusões e Integração Futura

**No Slide (O que colocar na tela):**
* **Título:** Conclusões e Trabalhos Futuros
* **Desempenho:** Arquiteturas densas resolvem o mapeamento aéreo quando treinadas com Augmentation Omnidirecional.
* **Próximo Passo (DataLuta):** Fundir inferência espacial (CNN Satélite) com inferência jurídica textual (LLMs) em um único framework híbrido.

**O que falar para o professor:**
- "Como conclusão deste trabalho, fica provado empíricamente que as texturas de satélite demandam injeção de invariância de rotação para estabilizar o aprendizado."
- "O próximo passo tecnológico, como tese de doutorado, é acoplar esse modelo convolucional que fizemos hoje como o 'motor de visão' de uma IA Multimodal."
- "O objetivo é que, ao lermos uma petição jurídica sobre desmatamento via RAG/LLM, o sistema simultaneamente classifique a imagem de satélite da coordenada informada pelo juiz, criando a primeira plataforma híbrida de denúncia socioambiental cruzada."
