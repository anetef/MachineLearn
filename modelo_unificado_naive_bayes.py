import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score


print(" INICIANDO PIPELINE DE INTELIGÊNCIA ARTIFICIAL ")

# PARTE 1: PROCESSAMENTO DA BASE DE DADOS credit_data.csv
print("\n[BASE 1] Carregando dados do professor (credit_data.csv)...")
base_prof = pd.read_csv('credit_data.csv')

# faxina dos dados credit_data.csv (tratando idades negativas e nulos de forma direta)
base_prof.loc[base_prof['age'] < 0, 'age'] = 40.92

# TRUQUE DEFINITIVO: preenche os nulos direto no array numérico com o Pandas
base_prof = base_prof.fillna(base_prof.mean(numeric_only=True))
#X_prof = pd.DataFrame(X_prof).fillna(base_prof['loan'].mean()).values

# separando X e y (usando as colunas numéricas de renda, idade e empréstimo)
X_prof = base_prof.iloc[:, 1:4].values
y_prof = base_prof.iloc[:, 4].values

# Verificando se ainda existem nulos na base do professor
#print("Valores nulos por coluna:\n", base_prof.isnull().sum())

# escalonamento dos dados
scaler = StandardScaler()
X_prof = scaler.fit_transform(X_prof)
# divisão treino/teste
X_tr_p, X_te_p, y_tr_p, y_te_p = train_test_split(X_prof, y_prof, test_size=0.25, random_state=0)

# treinamento do algoritmo
gnb_prof = GaussianNB()
gnb_prof.fit(X_tr_p, y_tr_p)
prev_prof = gnb_prof.predict(X_te_p)

# exibindo resultados da base 1
print("\n📊 === RESULTADOS: BASE DO PROFESSOR ===")
print(f"Acurácia : {accuracy_score(y_te_p, prev_prof) * 100:.2f}%")
print(f"Precisão : {precision_score(y_te_p, prev_prof) * 100:.2f}%")
print(f"Revocação : {recall_score(y_te_p, prev_prof) * 100:.2f}%")
print(f"F1-Score : {f1_score(y_te_p, prev_prof) * 100:.2f}%")
print("Matriz de Confusão:")
print(confusion_matrix(y_te_p, prev_prof))


print("\n--------------------------------------------------")
print("[BASE 2] Carregando dados do Kaggle (Credit Score Classification Dataset.csv)...")
base_kaggle = pd.read_csv('Credit Score Classification Dataset.csv')

# preparação dos dados do Kaggle (Usando o get_dummies para os textos)
y_kaggle = base_kaggle['Credit Score'].values
X_tab_kaggle = pd.get_dummies(base_kaggle.drop('Credit Score', axis=1))
X_kaggle = X_tab_kaggle.values

# escalonamento dos dados
X_kaggle = scaler.fit_transform(X_kaggle)

# divisão treino/teste
X_tr_k, X_te_k, y_tr_k, y_te_k = train_test_split(X_kaggle, y_kaggle, test_size=0.25, random_state=1)

# treinamento do algoritmo
gnb_kaggle = GaussianNB()
gnb_kaggle.fit(X_tr_k, y_tr_k)
prev_kaggle = gnb_kaggle.predict(X_te_k)

# exibindo resultados da base 2
print("\n📊 === RESULTADOS: BASE DO KAGGLE ===")
print(f"Acurácia : {accuracy_score(y_te_k, prev_kaggle) * 100:.2f}%")
print(f"Precisão : {precision_score(y_te_k, prev_kaggle, average='weighted', zero_division=0) * 100:.2f}%")
print(f"Revocação : {recall_score(y_te_k, prev_kaggle, average='weighted', zero_division=0) * 100:.2f}%") 
print(f"F1-Score : {f1_score(y_te_k, prev_kaggle, average='weighted', zero_division=0) * 100:.2f}%")
print("Matriz de Confusão:")
print(confusion_matrix(y_te_k, prev_kaggle))
