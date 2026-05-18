import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

print("1. Carregando a base do Kaggle para o Naive Bayes...")
base = pd.read_csv('Credit Score Classification Dataset.csv')

y = base['Credit Score'].values
X_tabela = pd.get_dummies(base.drop('Credit Score', axis=1))
X = X_tabela.values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.25, random_state=1)

print("\n2. Treinando o motor Naive Bayes...")
classificador = GaussianNB()
classificador.fit(X_treino, y_treino)

print("3. Fazendo o teste com novos clientes...")
previsoes = classificador.predict(X_teste)

print("\n=== RESULTADOS FINAIS: NAIVE BAYES (KAGGLE) ===")
print(f"Acurácia: {accuracy_score(y_teste, previsoes) * 100:.2f}%")
print("Matriz de Confusão:")
print(confusion_matrix(y_teste, previsoes))
print(f"Precisão: {precision_score(y_teste, previsoes, average='weighted', zero_division=0) * 100:.2f}%")
print(f"Revocação: {recall_score(y_teste, previsoes, average='weighted') * 100:.2f}%")
print(f"F1-Score: {f1_score(y_teste, previsoes, average='weighted') * 100:.2f}%")