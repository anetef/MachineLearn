# =====================================================================
# 🚀 ALGORITMO: ÁRVORE DE DECISÃO (EQUIVALENTE EM R)
# =====================================================================

# install.packages("rpart")
# install.packages("caret")
library(rpart)
library(caret)

# (A função avaliar_modelo é idêntica para manter o padrão das métricas)
avaliar_modelo <- function(matriz_confusao, tipo_base) {
  dados_metricas <- confusionMatrix(matriz_confusao)
  acuracia <- dados_metricas$overall['Accuracy'] * 100
  if (tipo_base == "prof") {
    precisao <- dados_metricas$byClass['Precision'] * 100
    revocacao <- dados_metricas$byClass['Sensitivity'] * 100
    f1 <- dados_metricas$byClass['F1'] * 100
  } else {
    pesos <- colSums(matriz_confusao) / sum(matriz_confusao)
    precisao <- sum(dados_metricas$byClass[, 'Precision'] * pesos, na.rm = TRUE) * 100
    revocacao <- sum(dados_metricas$byClass[, 'Sensitivity'] * pesos, na.rm = TRUE) * 100
    f1 <- sum(dados_metricas$byClass[, 'F1'] * pesos, na.rm = TRUE) * 100
  }
  cat(sprintf("  Acurácia  : %.2f%%\n", acuracia))
  cat(sprintf("  Precisão  : %.2f%%\n", precisao))
  cat(sprintf("  Revocação : %.2f%%\n", revocacao))
  cat(sprintf("  F1-Score  : %.2f%%\n", f1))
  cat("  Matriz de Confusão:\n  (Linha: Real, Coluna: Previsto)\n")
  print(t(matriz_confusao))
  cat(strrep("-", 40), "\n")
}

print("=== RUNNING DECISION TREE PIPELINE ===")

# --- BASE 1: PROFESSOR ---
base_prof <- read.csv("credit_data.csv")
base_prof$age[base_prof$age < 0] <- 40.92
base_prof$loan[is.na(base_prof$loan)] <- mean(base_prof$loan, na.rm = TRUE)

X_prof_scaled <- scale(base_prof[, 2:4])
base_prof_final <- data.frame(X_prof_scaled, default = as.factor(base_prof$default))

set.seed(0)
amostra_p <- sample(1:nrow(base_prof_final), size = round(0.75 * nrow(base_prof_final)))
treino_p <- base_prof_final[amostra_p, ]
teste_p  <- base_prof_final[-amostra_p, ]

# Configurado com Entropia (information) e profundidade máxima = 3
dt_prof <- rpart(default ~ ., data = treino_p, method = "class", 
                 parms = list(split = "information"), control = rpart.control(maxdepth = 3))
prev_dt_p <- predict(dt_prof, newdata = teste_p, type = "class")

cat("\n📊 === RESULTADOS: BASE DO PROFESSOR ===\n")
avaliar_modelo(table(teste_p$default, prev_dt_p), "prof")

# --- BASE 2: KAGGLE ---
base_kaggle <- read.csv("Credit Score Classification Dataset.csv")
dummies <- dummyVars(" ~ .", data = base_kaggle[, names(base_kaggle) != "Credit.Score"])
X_kaggle_dummies <- predict(dummies, newdata = base_kaggle)
X_kaggle_scaled <- scale(X_kaggle_dummies)
base_kaggle_final <- data.frame(X_kaggle_scaled, Credit.Score = as.factor(base_kaggle$Credit.Score))

set.seed(1)
amostra_k <- sample(1:nrow(base_kaggle_final), size = round(0.75 * nrow(base_kaggle_final)))
treino_k <- base_kaggle_final[amostra_k, ]
teste_k  <- base_kaggle_final[-amostra_k, ]

dt_kagg <- rpart(Credit.Score ~ ., data = treino_k, method = "class", 
                 parms = list(split = "information"), control = rpart.control(maxdepth = 3))
prev_dt_k <- predict(dt_kagg, newdata = teste_k, type = "class")

cat("\n📊 === RESULTADOS: BASE DO KAGGLE ===\n")
avaliar_modelo(table(teste_k$Credit.Score, prev_dt_k), "kaggle")