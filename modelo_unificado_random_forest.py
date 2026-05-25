import pandas as pd # Importa o Pandas. É a nossa ferramenta para ler, limpar e manipular tabelas de dados (como um Excel programável).
from sklearn.preprocessing import StandardScaler # Ferramenta matemática para esmagar os dados e deixá-los na mesma escala (padronização).
from sklearn.model_selection import train_test_split # Ferramenta que corta a base de dados em duas partes: Treino (estudo) e Teste (prova).
from sklearn.ensemble import RandomForestClassifier # Importa o algoritmo Random Forest (Floresta Aleatória), que trabalha criando múltiplas árvores.
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score # Nossos "juízes". Avaliam o desempenho do modelo.

print(" INICIANDO PIPELINE DE INTELIGÊNCIA ARTIFICIAL ")

# ==============================================================================
# 📊 PARTE 1: PROCESSAMENTO DA BASE 1 (Dados do Professor)
# ==============================================================================
print("\n[BASE 1] Carregando dados do professor (credit_data.csv)...")
# Lê o arquivo CSV e guarda dentro de uma variável chamada base_prof
base_prof = pd.read_csv('credit_data.csv') 

# --- FAXINA DE DADOS (PRÉ-PROCESSAMENTO) ---
# .loc localiza todas as linhas onde a idade é menor que 0. Em seguida, substitui essas idades irreais pela média válida (40.92).
base_prof.loc[base_prof['age'] < 0, 'age'] = 40.92

# O comando .fillna procura espaços vazios (clientes sem valor de empréstimo) e injeta a média matemática da coluna neles. Evita o erro de NaN (Not a Number).
base_prof['loan'] = base_prof['loan'].fillna(base_prof['loan'].mean())

# --- SEPARAÇÃO DAS VARIÁVEIS (X e Y) ---
# .iloc fatia a tabela. Pega todas as linhas (:) e as colunas numéricas de 1 a 3 (renda, idade, empréstimo). .values transforma em matriz matemática. (Variáveis Independentes)
X_prof = base_prof.iloc[:, 1:4].values 

# Pega apenas a coluna 4 (0 para pagou, 1 para calote). Este é o nosso Alvo. (Variável Dependente)
y_prof = base_prof.iloc[:, 4].values 

# --- ESCALONAMENTO ---
scaler = StandardScaler() # Instancia o padronizador matemático.
X_prof = scaler.fit_transform(X_prof) # Aplica a padronização no X, garantindo que variáveis como Renda (alta) não esmagem a Idade (baixa).

# --- DIVISÃO DA BASE EM TREINO E TESTE ---
# Corta os dados: 75% para a máquina estudar (treino) e 25% escondidos para testar (teste). random_state=0 fixa a semente aleatória para o resultado ser sempre igual.
X_tr_p, X_te_p, y_tr_p, y_te_p = train_test_split(X_prof, y_prof, test_size=0.25, random_state=0)

# --- TREINAMENTO DO MODELO RANDOM FOREST ---
# Cria a inteligência artificial. n_estimators=40 significa que criamos 40 árvores de decisão. criterion='entropy' usa entropia para medir a desordem nas divisões.
rf_prof = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)

# O comando .fit é a fase de estudo! A máquina cruza os perfis (X_tr) com as respostas (y_tr) e aprende as regras do banco.
rf_prof.fit(X_tr_p, y_tr_p) 

# O comando .predict é a fase de prova! A máquina recebe apenas os perfis (X_te) e tenta prever quem dá calote.
prev_prof = rf_prof.predict(X_te_p) 

# --- AVALIAÇÃO DO MODELO (BASE 1) ---
print("\n📊 === RESULTADOS: BASE DO PROFESSOR ===")
print(f"Acurácia : {accuracy_score(y_te_p, prev_prof) * 100:.2f}%") # Taxa de acerto geral.
print(f"Precisão : {precision_score(y_te_p, prev_prof) * 100:.2f}%") # Dos que o modelo disse ser calote, quantos realmente eram?
print(f"Revocação : {recall_score(y_te_p, prev_prof) * 100:.2f}%") # De todos os caloteiros reais, quantos o modelo achou?
print(f"F1-Score : {f1_score(y_te_p, prev_prof) * 100:.2f}%") # Média harmônica que pune o modelo se ele não equilibrar Precisão e Revocação.
print("Matriz de Confusão:")
print(confusion_matrix(y_te_p, prev_prof)) # Mostra os Falsos Positivos, Falsos Negativos, Verdadeiros Positivos e Verdadeiros Negativos.


# ==============================================================================
# 📊 PARTE 2: PROCESSAMENTO DA BASE 2 (Dataset Multiclasse do Kaggle)
# ==============================================================================
print("\n--------------------------------------------------")
print("[BASE 2] Carregando dados do Kaggle (Credit Score Classification Dataset.csv)...")
base_kaggle = pd.read_csv('Credit Score Classification Dataset.csv')

# --- TRATAMENTO DE TEXTOS (ONE-HOT ENCODING) ---
# Isola a coluna 'Credit Score' (nosso alvo com as 3 categorias).
y_kaggle = base_kaggle['Credit Score'].values 

# get_dummies pega a tabela sem o alvo e aplica o One-Hot Encoding: transforma textos (ex: 'Male', 'High School') em colunas numéricas binárias (0 ou 1) independentes.
X_tab_kaggle = pd.get_dummies(base_kaggle.drop('Credit Score', axis=1)) 
X_kaggle = X_tab_kaggle.values # Transforma a nova tabela binarizada em matriz pura.

# --- ESCALONAMENTO E DIVISÃO (Idêntico à Base 1) ---
X_kaggle = scaler.fit_transform(X_kaggle) 
X_tr_k, X_te_k, y_tr_k, y_te_k = train_test_split(X_kaggle, y_kaggle, test_size=0.25, random_state=1) # random_state=1 usado para diferenciar o embaralhamento.

# --- TREINAMENTO DO MODELO (Idêntico à Base 1) ---
# A configuração do algoritmo se mantém idêntica (40 árvores, entropia) para permitir uma comparação científica justa.
rf_kaggle = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)
rf_kaggle.fit(X_tr_k, y_tr_k)
prev_kaggle = rf_kaggle.predict(X_te_k)

# --- AVALIAÇÃO DO MODELO (BASE 2) ---
print("\n📊 === RESULTADOS: BASE DO KAGGLE ===")
print(f"Acurácia : {accuracy_score(y_te_k, prev_kaggle) * 100:.2f}%")

# Como a Base 2 possui três categorias (multiclasse) e não apenas duas (binária), usamos o parâmetro average='weighted'. Ele tira uma média ponderada das métricas respeitando o volume de dados de cada classe.
print(f"Precisão : {precision_score(y_te_k, prev_kaggle, average='weighted', zero_division=0) * 100:.2f}%")
print(f"Revocação : {recall_score(y_te_k, prev_kaggle, average='weighted', zero_division=0) * 100:.2f}%")
print(f"F1-Score : {f1_score(y_te_k, prev_kaggle, average='weighted', zero_division=0) * 100:.2f}%") 

print("Matriz de Confusão:")
print(confusion_matrix(y_te_k, prev_kaggle)) # Como são 3 classes, essa matriz será 3x3.
