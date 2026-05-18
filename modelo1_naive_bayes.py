import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

print("1. Carregando e limpando os dados...")
base = pd.read_csv('credit_data.csv')

# removendo a coluna inútil
base = base.drop('clientid', axis=1)

# tratando idades negativas e valores vazios
base.loc[base['age'] < 0, 'age'] = 40.92
base['age'] = base['age'].fillna(base['age'].mean())

# separando atributos (X) e a classe alvo (y)
X = base.iloc[:, 0:3].values
y = base.iloc[:, 3].values

# escalonamento 
scaler = StandardScaler()
X = scaler.fit_transform(X)

# dividindo 75% para treinar e 25% para testar
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.25, random_state=1)

print("2. Treinando o modelo Naive Bayes...")
classificador = GaussianNB()
classificador.fit(X_treino, y_treino)

print("3. Fazendo o teste com novos clientes...")
previsoes = classificador.predict(X_teste)

# calculando os resultados finais!
acuracia = accuracy_score(y_teste, previsoes)
matriz = confusion_matrix(y_teste, previsoes)

print("\n=== RESULTADOS FINAIS ===")
print(f"Acurácia (Taxa de Acerto): {acuracia * 100:.2f}%")
print("Matriz de Confusão:")
print(matriz)
print(f"Precisão: {precision_score(y_teste, previsoes) * 100:.2f}%")
print(f"Revocação: {recall_score(y_teste, previsoes) * 100:.2f}%")
print(f"F1-Score: {f1_score(y_teste, previsoes) * 100:.2f}%")