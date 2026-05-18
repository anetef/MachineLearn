import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

print("1. Carregando e limpando os dados...")
base = pd.read_csv('credit_data.csv')
base = base.drop('clientid', axis=1)

base.loc[base['age'] < 0, 'age'] = 40.92
base['age'] = base['age'].fillna(base['age'].mean())

X = base.iloc[:, 0:3].values
y = base.iloc[:, 3].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.25, random_state=1)

print("2. Treinando o modelo Random Forest...")
# n_estimators=40 significa que ele vai criar 40 árvores de decisão e fazer uma votação
classificador = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)
classificador.fit(X_treino, y_treino)

print("3. Fazendo o teste com novos clientes...")
previsoes = classificador.predict(X_teste)

print("\n=== RESULTADOS FINAIS: RANDOM FOREST ===")
print(f"Acurácia: {accuracy_score(y_teste, previsoes) * 100:.2f}%")
print("Matriz de Confusão:")
print(confusion_matrix(y_teste, previsoes))
print(f"Precisão: {precision_score(y_teste, previsoes) * 100:.2f}%")
print(f"Revocação: {recall_score(y_teste, previsoes) * 100:.2f}%")
print(f"F1-Score: {f1_score(y_teste, previsoes) * 100:.2f}%")