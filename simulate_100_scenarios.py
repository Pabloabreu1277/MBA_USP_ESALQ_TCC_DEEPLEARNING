"""
Script Python para simular 100 cenários com valores aleatórios usando o modelo Chiller treinado.

Este script realiza as seguintes etapas:
1. Carrega o modelo Keras treinado (
`chiller_model_python.keras").
2. Carrega o scaler salvo (
`scaler_python.pkl").
3. Gera 100 conjuntos de dados de entrada aleatórios sintéticos.
    - A lógica de geração dos dados é consistente com `create_dataset.py`.
4. Prepara os dados de entrada:
    - Normaliza as features usando o scaler carregado.
5. Realiza as predições com o modelo carregado para cada um dos 100 cenários.
6. Armazena as entradas originais, as saídas preditas (ajuste de setpoint e comando de habilitação).
7. Salva todos esses dados em um arquivo CSV (`simulacao_100_cenarios_resultados.csv`) para uso posterior
   no cálculo da carga térmica e na geração da planilha Excel.
"""

# Bloco 1: Importação de Bibliotecas
# ----------------------------------
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

print("TensorFlow version:", tf.__version__)

# Bloco 2: Carregamento do Modelo e Scaler
# ----------------------------------------
MODEL_PATH = "/home/ubuntu/chiller_model_python.keras"
SCALER_PATH = "/home/ubuntu/scaler_python.pkl"
OUTPUT_CSV_PATH = "/home/ubuntu/simulacao_100_cenarios_resultados.csv"

try:
    model = load_model(MODEL_PATH)
    print(f"Modelo {MODEL_PATH} carregado com sucesso.")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    exit()

try:
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    print(f"Scaler {SCALER_PATH} carregado com sucesso.")
except FileNotFoundError:
    print(f"Erro: Arquivo {SCALER_PATH} não encontrado.")
    exit()
except Exception as e:
    print(f"Erro ao carregar o scaler: {e}")
    exit()

# Bloco 3: Geração de 100 Cenários com Dados Aleatórios
# -----------------------------------------------------
num_scenarios = 100
np.random.seed(456) # Usar uma seed diferente para garantir novos dados

scenarios_data = {
    'temperatura_ambiente_c': np.random.uniform(0, 50, num_scenarios),
    'temperatura_entrada_chiller_c': np.random.uniform(5, 40, num_scenarios),
    'pressao_agua_saida_bar': np.random.uniform(1, 5, num_scenarios),
    'setpoint_inicial_c': np.random.uniform(4, 15, num_scenarios),
    'status_bomba_primaria': np.random.randint(0, 2, num_scenarios),
    'status_bomba_secundaria': np.random.randint(0, 2, num_scenarios),
    'status_valvula_bloqueio': np.random.randint(0, 2, num_scenarios)
}

df_scenarios = pd.DataFrame(scenarios_data)

# Gerar 'temperatura_saida_chiller_c' de forma condicional (lógica similar ao create_dataset.py)
# Esta é uma *entrada* para o modelo, simulando a leitura real do sensor de saída do Chiller.
df_scenarios["temperatura_saida_chiller_c"] = df_scenarios.apply(
    lambda row: row["temperatura_entrada_chiller_c"] - np.random.uniform(2, 10)
    if row["status_bomba_primaria"] == 1
    and row["status_bomba_secundaria"] == 1
    and row["status_valvula_bloqueio"] == 1
    and row["pressao_agua_saida_bar"] >= 1.5
    else row["temperatura_entrada_chiller_c"] - np.random.uniform(-1, 2),
    axis=1,
)
df_scenarios["temperatura_saida_chiller_c"] = df_scenarios["temperatura_saida_chiller_c"].clip(lower=1)
df_scenarios["temperatura_saida_chiller_c"] = df_scenarios.apply(lambda row: min(row["temperatura_saida_chiller_c"], row["temperatura_entrada_chiller_c"]), axis=1)

print(f"{num_scenarios} cenários aleatórios gerados para simulação.")

# Bloco 4: Preparação dos Dados de Entrada para o Modelo
# -----------------------------------------------------
# Assegurar que a ordem das colunas é a mesma usada no treinamento
feature_columns = [
    "temperatura_ambiente_c",
    "temperatura_entrada_chiller_c",
    "temperatura_saida_chiller_c", # Importante: esta é uma das features de entrada
    "pressao_agua_saida_bar",
    "setpoint_inicial_c",
    "status_bomba_primaria",
    "status_bomba_secundaria",
    "status_valvula_bloqueio",
]

# Reordenar colunas do DataFrame de cenários para corresponder à ordem das features do modelo
df_scenarios_input_features = df_scenarios[feature_columns].copy()

# Normalizar as features usando o scaler carregado
X_scenarios_scaled = scaler.transform(df_scenarios_input_features)

print("Dados de entrada dos cenários normalizados e prontos para predição.")

# Bloco 5: Realização das Predições para os 100 Cenários
# -------------------------------------------------------
print("Realizando predições para os 100 cenários...")
all_predictions = model.predict(X_scenarios_scaled)

# As predições são uma lista, onde cada elemento corresponde a uma saída do modelo
# predictions[0] são os ajustes de setpoint (regressão)
# predictions[1] são as probabilidades de comando de habilitação (classificação)

ajustes_setpoint_pred = all_predictions[0].flatten()
comandos_habilitacao_pred_proba = all_predictions[1].flatten()

# Converter as probabilidades do comando de habilitação para classes (0 ou 1)
comandos_habilitacao_pred_classe = (comandos_habilitacao_pred_proba > 0.5).astype(int)

# Bloco 6: Armazenamento dos Resultados
# -------------------------------------
# Adicionar as predições ao DataFrame original dos cenários
df_scenarios['ajuste_setpoint_pred_c'] = ajustes_setpoint_pred
df_scenarios['comando_habilitacao_pred_proba'] = comandos_habilitacao_pred_proba
df_scenarios['comando_habilitacao_pred_classe'] = comandos_habilitacao_pred_classe

# Calcular o setpoint ajustado pela rede neural
df_scenarios['setpoint_ajustado_rede_c'] = df_scenarios['setpoint_inicial_c'] + df_scenarios['ajuste_setpoint_pred_c']

# Reordenar colunas para melhor visualização no CSV (opcional, mas útil)
final_columns_order = feature_columns + ['ajuste_setpoint_pred_c', 'comando_habilitacao_pred_proba', 'comando_habilitacao_pred_classe', 'setpoint_ajustado_rede_c']
df_resultados_finais = df_scenarios[final_columns_order].copy()

# Salvar o DataFrame completo em um arquivo CSV
df_resultados_finais.to_csv(OUTPUT_CSV_PATH, index=False, float_format='%.4f')
print(f"Resultados da simulação dos 100 cenários salvos em: {OUTPUT_CSV_PATH}")

print("\nPrimeiras 5 linhas dos resultados da simulação:")
print(df_resultados_finais.head())

print("\nScript de simulação de 100 cenários concluído.")


