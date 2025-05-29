# -*- coding: utf-8 -*-
"""
Script para simular os 100 cenários usando o modelo V2.

1. Carrega a planilha base original (".xlsx").
2. Prepara as features para o modelo V2 (excluindo setpoint_inicial_c).
3. Carrega o modelo V2 e o scaler V2.
4. Normaliza os dados e executa a inferência para obter o setpoint ótimo direto e o comando.
5. Calcula a carga térmica usando o SETPOINT INICIAL ORIGINAL da planilha.
6. Calcula a carga térmica usando o SETPOINT ÓTIMO previsto pela REDE V2.
7. Calcula a diferença de energia.
8. Salva os resultados em um novo arquivo CSV.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import os

# --- Configurações ---
BASE_EXCEL_PATH = "/home/ubuntu/upload/simulacao_100_cenarios_completo.xlsx" # Usar a planilha original
MODEL_PATH = "/home/ubuntu/chiller_model_python_v2.keras"
SCALER_PATH = "/home/ubuntu/scaler_python_v2.pkl"
OUTPUT_CSV_PATH = "/home/ubuntu/simulacao_100_cenarios_v2_temp.csv"

# Constantes para cálculo de carga térmica (usar valores consistentes)
VAZAO_M3H = 150
DENSIDADE_AGUA_KG_M3 = 1000
CALOR_ESPECIFICO_KJ_KG_C = 4.186
VAZAO_KGS = (VAZAO_M3H * DENSIDADE_AGUA_KG_M3) / 3600

# --- Funções Auxiliares ---
def calcular_carga_termica(temp_entrada, setpoint, comando_habilitacao):
    """Calcula a carga térmica em kW se o comando for 1."""
    if comando_habilitacao == 1:
        delta_t = temp_entrada - setpoint
        delta_t = max(0, delta_t) # Garante que não seja negativo
        carga_kw = VAZAO_KGS * CALOR_ESPECIFICO_KJ_KG_C * delta_t
        return carga_kw
    else:
        return 0.0

# --- Bloco Principal ---
print(f"Carregando planilha base original: {BASE_EXCEL_PATH}")
try:
    df_base = pd.read_excel(BASE_EXCEL_PATH)
    print("Planilha base carregada com sucesso.")
except FileNotFoundError:
    print(f"Erro: Arquivo {BASE_EXCEL_PATH} não encontrado.")
    exit()
except Exception as e:
    print(f"Erro ao carregar a planilha: {e}")
    exit()

# Verificar colunas essenciais para input do modelo V2 e para comparação
required_model_cols = [
    'temperatura_ambiente_c',
    'temperatura_entrada_chiller_c',
    'temperatura_saida_chiller_c',
    'pressao_agua_saida_bar',
    'status_bomba_primaria',
    'status_bomba_secundaria',
    'status_valvula_bloqueio'
]
required_comparison_cols = ['setpoint_inicial_c'] # Necessário para comparação

all_required_cols = required_model_cols + required_comparison_cols
if not all(col in df_base.columns for col in all_required_cols):
    print(f"Erro: Colunas necessárias não encontradas na planilha. Necessário: {all_required_cols}")
    exit()

df_sim = df_base.copy()

# Preparar dados de entrada para a rede neural V2 (7 features)
X_input = df_sim[required_model_cols]

print("Carregando modelo V2 e scaler V2...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print("Modelo V2 e scaler V2 carregados.")
except Exception as e:
    print(f"Erro ao carregar modelo V2 ou scaler V2: {e}")
    exit()

print("Normalizando dados de entrada...")
try:
    # Verificar se o scaler foi treinado com as colunas corretas
    if list(scaler.feature_names_in_) != required_model_cols:
        print("Aviso: As colunas do scaler não correspondem exatamente às colunas de entrada do modelo V2.")
        print(f"Scaler espera: {list(scaler.feature_names_in_)}")
        print(f"Modelo usa: {required_model_cols}")
        # Tentar prosseguir mesmo assim, mas pode gerar erro ou resultado incorreto
    X_scaled = scaler.transform(X_input)
except ValueError as e:
    print(f"Erro ao normalizar dados: {e}. Verifique se as colunas estão corretas e na ordem esperada pelo scaler V2.")
    exit()
except AttributeError:
     print("Aviso: Scaler não possui 'feature_names_in_'. Prosseguindo com a normalização.")
     X_scaled = scaler.transform(X_input)

print("Executando inferência com o modelo V2...")
predictions = model.predict(X_scaled)
pred_setpoint_final_otimo = predictions[0].flatten()
pred_comando_prob = predictions[1].flatten()

# Processar predições
df_sim['setpoint_rede_v2_c'] = pred_setpoint_final_otimo
df_sim['comando_habilitacao_rede_v2_prob'] = pred_comando_prob
df_sim['comando_habilitacao_rede_v2'] = (pred_comando_prob >= 0.5).astype(int)

# Limitar setpoint da rede a um mínimo razoável (ex: 1°C)
df_sim['setpoint_rede_v2_c'] = df_sim['setpoint_rede_v2_c'].clip(lower=1.0)

print("Calculando cargas térmicas (Original vs Rede V2)...")

# Carga com setpoint inicial ORIGINAL da planilha
df_sim['carga_termica_kw_inicial_original'] = df_sim.apply(
    lambda row: calcular_carga_termica(
        row['temperatura_entrada_chiller_c'],
        row['setpoint_inicial_c'], # Usar o setpoint original da planilha
        row['comando_habilitacao_rede_v2'] # Usar o comando da rede V2 para decidir se calcula
    ),
    axis=1
)

# Carga com setpoint da REDE V2
df_sim['carga_termica_kw_rede_v2'] = df_sim.apply(
    lambda row: calcular_carga_termica(
        row['temperatura_entrada_chiller_c'],
        row['setpoint_rede_v2_c'],
        row['comando_habilitacao_rede_v2'] # Usar o comando da rede V2 para decidir se calcula
    ),
    axis=1
)

# Calcular diferença (Positivo = Rede V2 economizou)
df_sim['diferenca_energia_kw_v2'] = df_sim['carga_termica_kw_inicial_original'] - df_sim['carga_termica_kw_rede_v2']

print("Salvando resultados V2 intermediários em CSV...")
# Selecionar e reordenar colunas para o output
output_cols_v2 = required_model_cols + [
    'setpoint_inicial_c', # Incluir o setpoint inicial original para referência
    'setpoint_rede_v2_c',
    'comando_habilitacao_rede_v2_prob',
    'comando_habilitacao_rede_v2',
    'carga_termica_kw_inicial_original',
    'carga_termica_kw_rede_v2',
    'diferenca_energia_kw_v2'
]
# Adicionar colunas originais que não são input direto, se existirem e forem úteis
original_extra_cols = [col for col in df_base.columns if col not in all_required_cols and col not in ['carga_termica_kw_inicial', 'carga_termica_kw_rede', 'diferenca_energia_kw']]
final_cols = original_extra_cols + output_cols_v2

# Garantir que todas as colunas finais existem no df_sim
final_cols = [col for col in final_cols if col in df_sim.columns]

df_output = df_sim[final_cols]

try:
    df_output.to_csv(OUTPUT_CSV_PATH, index=False, float_format='%.2f')
    print(f"Resultados V2 salvos em: {OUTPUT_CSV_PATH}")
except Exception as e:
    print(f"Erro ao salvar o CSV V2: {e}")
    exit()

print("\nScript de simulação V2 concluído.")
print("Próximos passos: Converter CSV V2 para Excel e gerar gráfico V2.")

