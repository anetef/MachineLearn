# 📊 Análise Comparativa de Modelos de Classificação: R vs. Python

O objetivo principal é realizar uma análise preditiva e comparativa utilizando algoritmos de aprendizado de máquina supervisionado para problemas de modelagem e classificação de risco de crédito.

O projeto foi construído de forma colaborativa e equitativa, espelhando exatamente a mesma arquitetura lógica de dados, tratamento de valores ausentes (*missing values*) e engenharia de recursos (*feature engineering*) tanto na linguagem **Python** quanto na linguagem **R**.

---

## 👥 Estrutura do Grupo e Organização Ágil

O projeto seguiu a metodologia ágil Scrum para garantir entregas incrementais consistentes e alinhamento de papéis:
* **Desenvolvimento Python:** Implementação inicial das rotinas de higienização de nulos, escalonamento e pipelines de treino do Scikit-Learn.
* **Desenvolvimento R:** Tradução e pareamento exato da arquitetura de dados usando pacotes nativos do ecossistema CRAN (`caret`, `e1071`, `rpart`, `randomForest`).
* **Analytics & Tech Writing:** Consolidação das matrizes de confusão higienizadas, extração de métricas de validação e escrita técnica do artigo comparativo.

---

## 📂 Estrutura de Arquivos do Repositório

```text
.
├── credit_data.csv                       # Base 1: Dados de Crédito (Professor)
├── Credit Score Classification Dataset.csv # Base 2: Classificação de Score (Kaggle)
│
├── modelo_unificado_naive_bayes.py       # Pipeline Naive Bayes em Python
├── modelo_unificado_arvore.py            # Pipeline Árvore de Decisão em Python
├── modelo_unificado_random_forest.py      # Pipeline Random Forest em Python
│
├── modelo_unificado_naive_bayes.R       # Pipeline Naive Bayes em R
├── modelo_unificado_arvore.R            # Pipeline Árvore de Decisão em R
├── modelo_unificado_random_forest.R     # Pipeline Random Forest em R
└── README.md                             # Documentação oficial do repositório
```
## 🛠️ Requisitos e Configuração de Ambiente (Multiplataforma)

Este guia detalha todos os pré-requisitos, dependências e comandos necessários para configurar o ambiente de desenvolvimento local e executar os códigos unificados de Machine Learning em **Python** e **R** em qualquer sistema operacional (Windows, macOS ou Linux).

---

## 🐍 Configuração do Ambiente Python

O ecossistema Python é utilizado para rodar os algoritmos através da biblioteca **Scikit-Learn**. Os scripts realizam o carregamento, higienização de valores nulos, escalonamento de variáveis e cálculo de matrizes de confusão.

### Pré-requisitos do Sistema
* **Python**: Versão `3.10` ou superior (Instale via site oficial python.org, Anaconda ou gerenciador de pacotes do seu SO).
* **Gerenciador de Pacotes**: `pip` atualizado.

### Instalação das Dependências
Abra o terminal do seu sistema operacional (Prompt de Comando/PowerShell no Windows, Terminal no macOS/Linux) ou o terminal integrado da sua IDE e execute:

```bash
pip install pandas numpy scikit-learn
```

---

## 📊 Configuração do Ambiente R (Multiplataforma)

Este documento orienta a instalação do interpretador **R** e de suas respectivas dependências de Machine Learning necessárias para executar os scripts unificados do projeto (`modelo_unificado_...R`).

---

### 📥 Instalação do Interpretador R-Base

Antes de instalar as bibliotecas, é necessário ter o núcleo do R instalado no sistema operacional:

* **Windows:** Baixe e execute o instalador oficial do **R for Windows** através do CRAN: [https://cran.r-project.org/bin/windows/base/](https://cran.r-project.org/bin/windows/base/)
* **macOS:** Baixe a versão correspondente ao seu processador (Intel ou Apple Silicon M1/M2/M3) em: [https://cran.r-project.org/bin/macosx/](https://cran.r-project.org/bin/macosx/)
* **Linux (Ubuntu/Debian):** Abra o terminal do sistema e instale via gerenciador de pacotes nativo:
  ```bash
  sudo apt update
  sudo apt install r-base-core -y
