"""
Script Python para simular o modelo Chiller treinado com valores de entrada específicos.

Este script realiza as seguintes etapas:
1. Carrega o modelo Keras treinado (
`chiller_model_python.keras").
2. Carrega o scaler salvo (
`scaler_python.pkl").
3. Define os valores de entrada para a simulação conforme especificado pelo usuário.
4. Prepara os dados de entrada:
    - Cria um DataFrame com os valores de entrada.
    - Normaliza as features usando o scaler carregado.
5. Realiza a predição com o modelo carregado.
6. Exibe os resultados da predição (ajuste de setpoint e comando de habilitação).
7. Apresenta uma representação da arquitetura do modelo (summary).
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

try:
    model = load_model(MODEL_PATH)
    print(f"Modelo 	{MODEL_PATH}	 carregado com sucesso.")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    exit()

try:
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    print(f"Scaler 	{SCALER_PATH}	 carregado com sucesso.")
except FileNotFoundError:
    print(f"Erro: Arquivo 	{SCALER_PATH}	 não encontrado.")
    exit()
except Exception as e:
    print(f"Erro ao carregar o scaler: {e}")
    exit()

# Bloco 3: Definição dos Valores de Entrada para Simulação
# -------------------------------------------------------
# Valores fornecidos pelo usuário:
# Temperatura ambiente = 30°C
# Temperatura de entrada = 20°C
# Temperatura de saída = 18°C
# Pressão da água = 2,5 bar
# Setpoint inicial = 6°C
# Bomba primária = 1 (ligada)
# Bomba secundária = 1 (ligada)
# Válvula de bloqueio = 1 (ligada)

input_data_dict = {
    "temperatura_ambiente_c": [30.0],
    "temperatura_entrada_chiller_c": [20.0],
    "temperatura_saida_chiller_c": [18.0], # Esta é uma entrada para o modelo
    "pressao_agua_saida_bar": [2.5],
    "setpoint_inicial_c": [6.0],
    "status_bomba_primaria": [1],  # 1 para ligada, 0 para desligada
    "status_bomba_secundaria": [1], # 1 para ligada, 0 para desligada
    "status_valvula_bloqueio": [1]  # 1 para aberta/ligada, 0 para fechada/desligada
}

input_df = pd.DataFrame(input_data_dict)

print("\nValores de Entrada para Simulação:")
print(input_df)

# Bloco 4: Preparação dos Dados de Entrada
# ---------------------------------------
# Assegurar que a ordem das colunas é a mesma usada no treinamento
feature_columns = [
    "temperatura_ambiente_c",
    "temperatura_entrada_chiller_c",
    "temperatura_saida_chiller_c",
    "pressao_agua_saida_bar",
    "setpoint_inicial_c",
    "status_bomba_primaria",
    "status_bomba_secundaria",
    "status_valvula_bloqueio",
]

input_df_ordered = input_df[feature_columns]

# Normalizar as features usando o scaler carregado
# O scaler espera um array numpy ou DataFrame com as mesmas colunas que foram usadas para o fit
input_scaled = scaler.transform(input_df_ordered)

print("\nDados de Entrada Normalizados:")
print(input_scaled)

# Bloco 5: Realização da Predição
# -------------------------------
print("\nRealizando predição com o modelo...")
predictions = model.predict(input_scaled)

# As predições são uma lista, onde cada elemento corresponde a uma saída do modelo
ajuste_setpoint_pred = predictions[0][0][0]  # A saída é [[valor]], então pegamos [0][0]
comando_habilitacao_pred_proba = predictions[1][0][0] # A saída é [[probabilidade]], pegamos [0][0]

# Converter a probabilidade do comando de habilitação para uma classe (0 ou 1)
comando_habilitacao_pred_classe = 1 if comando_habilitacao_pred_proba > 0.5 else 0

# Bloco 6: Exibição dos Resultados da Predição
# -------------------------------------------
print("\nResultados da Simulação:")
print(f"  Ajuste de Setpoint Sugerido (ajuste_setpoint_c): {ajuste_setpoint_pred:.4f} °C")
print(f"  Probabilidade de Comando de Habilitação: {comando_habilitacao_pred_proba:.4f}")
print(f"  Comando de Habilitação Sugerido (0=Bloqueado, 1=Habilitado): {comando_habilitacao_pred_classe}")

# Bloco 7: Visualização da Arquitetura do Modelo
# ---------------------------------------------
# A arquitetura do modelo já foi impressa durante o treinamento.
# Para fins de simulação, podemos reimprimi-la aqui.
print("\nArquitetura do Modelo (Resumo):")
model.summary()

print("\nScript de simulação Python concluído.")

# Para uma "visualização em tempo real" mais interativa do fluxo de dados,
# seria necessário um framework de UI (como Dash, Flask, Streamlit) ou uma simulação passo a passo
# com prints detalhados em cada etapa, o que já foi feito de certa forma neste script.
# A visualização da arquitetura é dada pelo model.summary().
# O fluxo de dados é: Dados de Entrada -> Normalização -> Modelo -> Predições.


