# install.packages("randomForest")
# install.packages("caret")

# --- IMPORTAÇÃO DE PACOTES (Nossas ferramentas no R) ---
library(randomForest) # Pacote principal para criar a Floresta Aleatória.
library(caret) # Framework gigantesco de Machine Learning no R. Usamos para extrair métricas detalhadas e criar o One-Hot Encoding.

# Criamos essa função para automatizar o cálculo das métricas e garantir que a matemática seja idêntica para as duas bases.
avaliar_modelo <- function(matriz_confusao, tipo_base) {
  
  # A função confusionMatrix do pacote 'caret' calcula dezenas de métricas estatísticas automaticamente a partir da matriz.
  dados_metricas <- confusionMatrix(matriz_confusao)
  acuracia <- dados_metricas$overall['Accuracy'] * 100 # Puxa a Acurácia geral.
  
  # SE FOR A BASE DO PROFESSOR (Binária: Pagou ou Deu Calote):
  if (tipo_base == "prof") {
    precisao <- dados_metricas$byClass['Precision'] * 100
    revocacao <- dados_metricas$byClass['Sensitivity'] * 100 # No R, Revocação é chamada de 'Sensitivity' (Sensibilidade).
    f1 <- dados_metricas$byClass['F1'] * 100
    
  # SE FOR A BASE DO KAGGLE (Multiclasse: 3 categorias de score):
  } else {
    # Aqui fazemos o equivalente ao "average='weighted'" do Python.
    # Calculamos o peso de cada classe (quantos clientes tem em cada categoria).
    pesos <- colSums(matriz_confusao) / sum(matriz_confusao)
    
    # Multiplicamos as métricas pelo peso e somamos. Isso garante uma nota ponderada justa, sem que uma classe pequena distorça o resultado.
    precisao <- sum(dados_metricas$byClass[, 'Precision'] * pesos, na.rm = TRUE) * 100
    revocacao <- sum(dados_metricas$byClass[, 'Sensitivity'] * pesos, na.rm = TRUE) * 100
    f1 <- sum(dados_metricas$byClass[, 'F1'] * pesos, na.rm = TRUE) * 100
  }
  
  # Imprime os resultados formatados no console (%.2f%% formata para 2 casas decimais).
  cat(sprintf("  Acurácia  : %.2f%%\n", acuracia))
  cat(sprintf("  Precisão  : %.2f%%\n", precisao))
  cat(sprintf("  Revocação : %.2f%%\n", revocacao))
  cat(sprintf("  F1-Score  : %.2f%%\n", f1))
  cat("  Matriz de Confusão:\n  (Linha: Real, Coluna: Previsto)\n")
  print(t(matriz_confusao)) # 't()' transpõe a matriz para ela ficar no mesmo padrão visual (linhas/colunas) do Python.
  cat(strrep("-", 40), "\n")
}

print("=== RUNNING RANDOM FOREST PIPELINE ===")


# ==============================================================================
# 📊 PARTE 1: PROCESSAMENTO DA BASE 1 (Dados do Professor)
# ==============================================================================
base_prof <- read.csv("credit_data.csv") # Lê o arquivo CSV.

# --- FAXINA DE DADOS ---
# Encontra quem tem idade negativa e substitui por 40.92 (imputação pela média válida).
base_prof$age[base_prof$age < 0] <- 40.92

# 'is.na' encontra os valores nulos (vazios) na coluna de empréstimo. Substituímos esses espaços pela média (mean) da própria coluna.
base_prof$loan[is.na(base_prof$loan)] <- mean(base_prof$loan, na.rm = TRUE)

# --- ESCALONAMENTO ---
# A função 'scale' é o equivalente exato do StandardScaler do Python. Ela pega as colunas 2, 3 e 4 numéricas e as padroniza.
X_prof_scaled <- scale(base_prof[, 2:4])

# Junta as características escalonadas com a coluna alvo (default), transformando o alvo em um 'Fator' (categoria) para o R entender que é classificação.
base_prof_final <- data.frame(X_prof_scaled, default = as.factor(base_prof$default))

# Barreira de segurança: remove qualquer linha que ainda tenha algum resquício de valor nulo para não quebrar o Random Forest.
base_prof_final <- na.omit(base_prof_final)

# --- DIVISÃO TREINO/TESTE ---
set.seed(0) # Congela o sorteio para sempre pegar os mesmos clientes. Garantia de reprodutibilidade.
# 'sample' sorteia aleatoriamente 75% dos índices das linhas.
amostra_p <- sample(1:nrow(base_prof_final), size = round(0.75 * nrow(base_prof_final)))

treino_p <- base_prof_final[amostra_p, ]  # Pega as linhas sorteadas (Treino)
teste_p  <- base_prof_final[-amostra_p, ] # Pega todo o resto (o sinal de menos exclui a amostra) (Teste)

# --- TREINAMENTO DO MODELO ---
set.seed(0) # Outra semente para garantir que a geração das 40 árvores seja igual toda vez.

# 'default ~ .' significa: "Tente prever o 'default' (calote) usando TODAS (o ponto) as outras variáveis". ntree = 40 garante comparação justa com o Python.
rf_prof <- randomForest(default ~ ., data = treino_p, ntree = 40)

# Fase de teste: passa os perfis financeiros sem a resposta e pede para o modelo prever.
prev_rf_p <- predict(rf_prof, newdata = teste_p)

cat("\n📊 === RESULTADOS: BASE DO PROFESSOR ===\n")
avaliar_modelo(table(teste_p$default, prev_rf_p), "prof") # Envia a matriz para a nossa função calculadora.


# ==============================================================================
# 📊 PARTE 2: PROCESSAMENTO DA BASE 2 (Dataset Multiclasse do Kaggle)
# ==============================================================================
base_kaggle <- read.csv("Credit Score Classification Dataset.csv")

# --- TRATAMENTO DE TEXTOS (ONE-HOT ENCODING NO R) ---
# Aqui usamos a função 'dummyVars' do pacote Caret. É a versão R do get_dummies do Python. Ela pega as variáveis de texto e cria colunas binárias independentes.
dummies <- dummyVars(" ~ .", data = base_kaggle[, names(base_kaggle) != "Credit.Score"])

# Aplica a binarização gerando a nova tabela.
X_kaggle_dummies <- predict(dummies, newdata = base_kaggle)

# Escalonamento matemático.
X_kaggle_scaled <- scale(X_kaggle_dummies)

# Junta a tabela binarizada e escalonada com a nossa coluna alvo ('Credit Score').
base_kaggle_final <- data.frame(X_kaggle_scaled, Credit.Score = as.factor(base_kaggle$Credit.Score))

# --- DIVISÃO TREINO/TESTE ---
set.seed(1) # Semente fixada em 1 para reproduzir a exata separação de treino/teste que foi feita no Python para o Kaggle.
amostra_k <- sample(1:nrow(base_kaggle_final), size = round(0.75 * nrow(base_kaggle_final)))
treino_k <- base_kaggle_final[amostra_k, ]
teste_k  <- base_kaggle_final[-amostra_k, ]

# --- TREINAMENTO DO MODELO ---
set.seed(0)
# Treina prever o Credit.Score usando as variáveis restantes, criando 40 árvores de decisão que farão votação.
rf_kagg <- randomForest(Credit.Score ~ ., data = treino_k, ntree = 40)
prev_rf_k <- predict(rf_kagg, newdata = teste_k)

cat("\n📊 === RESULTADOS: BASE DO KAGGLE ===\n")
avaliar_modelo(table(teste_k$Credit.Score, prev_rf_k), "kaggle")
