import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
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

print("2. Treinando o modelo de Árvore de Decisão...")
classificador = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=3)
classificador.fit(X_treino, y_treino)

print("3. Fazendo o teste com novos clientes...")
previsoes = classificador.predict(X_teste)

print("\n=== RESULTADOS FINAIS: ÁRVORE DE DECISÃO ===")
print(f"Acurácia: {accuracy_score(y_teste, previsoes) * 100:.2f}%")
print("Matriz de Confusão:")
print(confusion_matrix(y_teste, previsoes))
print(f"Precisão: {precision_score(y_teste, previsoes) * 100:.2f}%")
print(f"Revocação: {recall_score(y_teste, previsoes) * 100:.2f}%")
print(f"F1-Score: {f1_score(y_teste, previsoes) * 100:.2f}%")
# gerando imagem da arvore
print("\n4. Gerando e salvando o gráfico da Árvore de Decisão em Alta Resolução...")

plt.figure(figsize=(24, 12)) 

tree.plot_tree(classificador, 
               feature_names=['Renda', 'Idade', 'Empréstimo'], 
               class_names=['Bom Pagador', 'Inadimplente'],
               filled=True, 
               rounded=True,
               fontsize=10) 

plt.savefig('grafico_arvore_credito_HD.png', bbox_inches='tight', dpi=300)

print("✅ Imagem 'grafico_arvore_credito_HD.png' salva com sucesso!")