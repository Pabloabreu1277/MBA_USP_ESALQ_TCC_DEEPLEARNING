# -*- coding: utf-8 -*-
"""
Script Python para gerar um dataset sintético para o projeto Chiller.

Versão 2: Modificado para gerar um alvo 'setpoint_final_otimo_c' 
independente do 'setpoint_inicial_c'.

Este script gera dados simulados para as seguintes variáveis:
Entradas (Features):
- temperatura_ambiente_c: Temperatura ambiente em Celsius.
- temperatura_entrada_chiller_c: Temperatura da água entrando no Chiller.
- temperatura_saida_chiller_c: Temperatura da água saindo do Chiller.
- pressao_agua_saida_bar: Pressão da água na linha de saída.
- setpoint_inicial_c: Setpoint de temperatura configurado inicialmente (usado para comparação).
- status_bomba_primaria: Status da bomba primária (0 ou 1).
- status_bomba_secundaria: Status da bomba secundária (0 ou 1).
- status_valvula_bloqueio: Status da válvula de bloqueio (0 ou 1).

Saídas (Alvos):
- setpoint_final_otimo_c: O setpoint que a rede deve prever como ótimo (regressão).
- comando_habilitacao: Comando para habilitar (1) ou bloquear (0) o Chiller (classificação).
"""

# Bloco 1: Importação de Bibliotecas
# ----------------------------------
import pandas as pd
import numpy as np
import random

# Bloco 2: Definição de Parâmetros
# --------------------------------
NUM_AMOSTRAS = 2000  # Número de amostras de dados a serem geradas
ARQUIVO_SAIDA = "chiller_dataset_v2.csv" # Nome do arquivo CSV de saída

# Limites e condições para geração de dados
TEMP_AMBIENTE_MIN, TEMP_AMBIENTE_MAX = 0, 50
TEMP_ENTRADA_MIN, TEMP_ENTRADA_MAX = 5, 40
PRESSAO_MIN, PRESSAO_MAX = 1.0, 5.0
PRESSAO_CRITICA = 1.5 # Pressão mínima para operação normal
SETPOINT_INICIAL_MIN, SETPOINT_INICIAL_MAX = 4, 15
DELTA_T_MIN, DELTA_T_MAX = 2, 10 # Redução de temperatura esperada quando operacional
SETPOINT_OTIMO_MIN, SETPOINT_OTIMO_MAX = 4.0, 10.0 # Limites para o setpoint ótimo previsto

# Bloco 3: Geração dos Dados
# --------------------------
dados = []

for _ in range(NUM_AMOSTRAS):
    # Gerar valores aleatórios para as entradas
    temp_ambiente = round(random.uniform(TEMP_AMBIENTE_MIN, TEMP_AMBIENTE_MAX), 2)
    temp_entrada = round(random.uniform(TEMP_ENTRADA_MIN, TEMP_ENTRADA_MAX), 2)
    pressao = round(random.uniform(PRESSAO_MIN, PRESSAO_MAX), 2)
    setpoint_inicial = round(random.uniform(SETPOINT_INICIAL_MIN, SETPOINT_INICIAL_MAX), 2)
    bomba_primaria = random.randint(0, 1)
    bomba_secundaria = random.randint(0, 1)
    valvula_bloqueio = random.randint(0, 1)

    # Determinar o comando de habilitação (lógica original mantida)
    comando_habilitacao = 1
    if not all([bomba_primaria, bomba_secundaria, valvula_bloqueio]) or pressao < PRESSAO_CRITICA:
        comando_habilitacao = 0

    # Gerar temperatura de saída com base no comando
    if comando_habilitacao == 1:
        # Se operacional, aplica uma redução de temperatura
        reducao = random.uniform(DELTA_T_MIN, DELTA_T_MAX)
        temp_saida = round(temp_entrada - reducao, 2)
        # Garantir que a saída não seja menor que um limite (ex: 1°C) e menor que a entrada
        temp_saida = max(1.0, temp_saida)
        temp_saida = min(temp_saida, temp_entrada - 0.1) # Garante temp_saida < temp_entrada
    else:
        # Se bloqueado, a redução é mínima ou nenhuma
        reducao_minima = random.uniform(0, 0.5)
        temp_saida = round(temp_entrada - reducao_minima, 2)
        temp_saida = max(1.0, temp_saida)
        temp_saida = min(temp_saida, temp_entrada) # Pode ser igual se redução for 0

    # Determinar o setpoint final ótimo (NOVA LÓGICA)
    if comando_habilitacao == 0:
        # Se o chiller não pode operar, o setpoint ótimo é menos relevante.
        # Poderia ser um valor alto, ou o próprio setpoint inicial.
        # Vamos definir como um valor alto para indicar "não resfriar".
        setpoint_final_otimo = 15.0
    else:
        # Lógica heurística para um setpoint ótimo baseado nas condições:
        base_setpoint = 7.0
        # Ajuste baseado na temperatura ambiente (mais frio -> setpoint mais alto)
        ambient_adj = (temp_ambiente - 25.0) * -0.1 # Ex: Se 35C, adj = -1.0; Se 15C, adj = +1.0
        # Ajuste baseado na eficiência aparente (delta T)
        delta_t_real = temp_entrada - temp_saida
        efficiency_adj = (delta_t_real - 5.0) * 0.05 # Ex: Se delta=8, adj = +0.15; Se delta=3, adj = -0.1

        optimal_setpoint = base_setpoint + ambient_adj + efficiency_adj
        # Adicionar ruído aleatório
        optimal_setpoint += random.uniform(-0.5, 0.5)
        # Limitar aos valores mínimo e máximo definidos
        setpoint_final_otimo = round(np.clip(optimal_setpoint, SETPOINT_OTIMO_MIN, SETPOINT_OTIMO_MAX), 2)

    # Adicionar a amostra à lista
    dados.append({
        "temperatura_ambiente_c": temp_ambiente,
        "temperatura_entrada_chiller_c": temp_entrada,
        "temperatura_saida_chiller_c": temp_saida,
        "pressao_agua_saida_bar": pressao,
        "setpoint_inicial_c": setpoint_inicial, # Mantido como feature
        "status_bomba_primaria": bomba_primaria,
        "status_bomba_secundaria": bomba_secundaria,
        "status_valvula_bloqueio": valvula_bloqueio,
        "setpoint_final_otimo_c": setpoint_final_otimo, # NOVO ALVO
        "comando_habilitacao": comando_habilitacao
    })

# Bloco 4: Criação do DataFrame e Salvamento
# ------------------------------------------
df_dataset = pd.DataFrame(dados)

# Reordenar colunas (opcional, mas bom para clareza)
colunas_ordenadas = [
    # Features
    "temperatura_ambiente_c",
    "temperatura_entrada_chiller_c",
    "temperatura_saida_chiller_c",
    "pressao_agua_saida_bar",
    "setpoint_inicial_c",
    "status_bomba_primaria",
    "status_bomba_secundaria",
    "status_valvula_bloqueio",
    # Alvos
    "setpoint_final_otimo_c",
    "comando_habilitacao"
]
df_dataset = df_dataset[colunas_ordenadas]

# Salvar o DataFrame em um arquivo CSV
try:
    df_dataset.to_csv(ARQUIVO_SAIDA, index=False)
    print(f"Dataset sintético V2 gerado e salvo com sucesso em ", ARQUIVO_SAIDA)
    print(f"Número de amostras: {len(df_dataset)}")
    print("\nPrimeiras 5 linhas do dataset:")
    print(df_dataset.head())
except Exception as e:
    print(f"Erro ao salvar o arquivo CSV: {e}")

print("\nScript de geração de dados V2 concluído.")

