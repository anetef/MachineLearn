import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn import tree

print("1. Carregando a base do Kaggle...")
# le base de dados externa
base = pd.read_csv('Credit Score Classification Dataset.csv')

y = base['Credit Score'].values

# get_dummies transforma palavras (como Male/Female) em colunas matemáticas de 0 e 1
X_tabela = pd.get_dummies(base.drop('Credit Score', axis=1))
X = X_tabela.values

# escalonamento 
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.25, random_state=1)

print("\n2. Treinando a Nova Árvore de Decisão...")
classificador = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=3)
classificador.fit(X_treino, y_treino)

print("3. Fazendo o teste com novos clientes...")
previsoes = classificador.predict(X_teste)

print("\n=== RESULTADOS FINAIS KAGGLE ===")
print(f"Acurácia: {accuracy_score(y_teste, previsoes) * 100:.2f}%")
print("Matriz de Confusão:")
print(confusion_matrix(y_teste, previsoes))

print(f"Precisão: {precision_score(y_teste, previsoes, average='weighted', zero_division=0) * 100:.2f}%")
print(f"Revocação: {recall_score(y_teste, previsoes, average='weighted') * 100:.2f}%")
print(f"F1-Score: {f1_score(y_teste, previsoes, average='weighted') * 100:.2f}%")

print("\n4. Gerando e salvando o novo gráfico HD...")
plt.figure(figsize=(24, 12)) 

# desenhando a árvore adaptada para os novos nomes e as 3 classes
tree.plot_tree(classificador, 
               feature_names=X_tabela.columns.tolist(), 
               class_names=classificador.classes_.tolist(),
               filled=True, 
               rounded=True,
               fontsize=10)

plt.savefig('arvore_Kaggle_HD.png', bbox_inches='tight', dpi=300)
print("✅ Imagem 'arvore_Kaggle_HD.png' salva com sucesso na sua pasta!")