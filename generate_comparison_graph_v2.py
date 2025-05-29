# -*- coding: utf-8 -*-
"""Gera um gráfico comparativo de carga térmica (Setpoint Original vs. Rede Neural V2)."""

import pandas as pd
import matplotlib
matplotlib.use("Agg") # Use Agg backend for non-GUI environments
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configurações ---
EXCEL_PATH = "/home/ubuntu/simulacao_100_cenarios_v2_final.xlsx"
OUTPUT_GRAPH_PATH = "/home/ubuntu/grafico_economia_energia_v2.png"

# --- Carregar Dados ---
print(f"Carregando dados da planilha V2: {EXCEL_PATH}")
try:
    df = pd.read_excel(EXCEL_PATH)
    print("Dados V2 carregados com sucesso.")
except FileNotFoundError:
    print(f"Erro: Arquivo {EXCEL_PATH} não encontrado.")
    exit()
except Exception as e:
    print(f"Erro ao carregar a planilha V2: {e}")
    exit()

# --- Filtrar Dados ---
# Considerar apenas os cenários onde a rede neural V2 habilitou o chiller
df_filtered = df[df["comando_habilitacao_rede_v2"] == 1].copy()

if df_filtered.empty:
    print("Erro: Nenhum cenário encontrado onde a rede neural V2 habilitou o chiller. Não é possível gerar o gráfico.")
    # Tentar gerar um gráfico vazio ou com uma mensagem?
    # Por enquanto, vamos sair se não houver dados.
    exit()

print(f"Número de cenários com Chiller habilitado pela rede V2: {len(df_filtered)}")

# Adicionar um índice para o eixo X do gráfico
df_filtered["cenario_index"] = range(1, len(df_filtered) + 1)

# --- Gerar Gráfico ---
print("Gerando gráfico comparativo V2...")
plt.style.use("seaborn-v0_8-whitegrid")
plt.figure(figsize=(14, 7))

# Plotar as cargas térmicas
sns.lineplot(data=df_filtered, x="cenario_index", y="carga_termica_kw_inicial_original", label="Carga Térmica (Setpoint Original)", marker=".", linestyle="-", color="red")
sns.lineplot(data=df_filtered, x="cenario_index", y="carga_termica_kw_rede_v2", label="Carga Térmica (Setpoint Rede Neural V2)", marker=".", linestyle="-", color="green")

# Calcular e exibir economia total (ou aumento de consumo)
carga_total_original = df_filtered["carga_termica_kw_inicial_original"].sum()
carga_total_rede_v2 = df_filtered["carga_termica_kw_rede_v2"].sum()
diferenca_total = carga_total_original - carga_total_rede_v2

if diferenca_total > 0:
    economia_texto = f"Economia Total (Rede V2 vs Original): {diferenca_total:.2f} kW"
else:
    economia_texto = f"Aumento Consumo (Rede V2 vs Original): {-diferenca_total:.2f} kW"

# Configurações do gráfico
plt.title(f"Comparativo de Carga Térmica: Setpoint Original vs. Rede Neural V2\n({economia_texto})")
plt.xlabel("Cenário (Chiller Habilitado pela Rede V2)")
plt.ylabel("Carga Térmica (kW)")
plt.legend()
plt.tight_layout()

# --- Salvar Gráfico ---
try:
    plt.savefig(OUTPUT_GRAPH_PATH)
    print(f"Gráfico V2 salvo como: {OUTPUT_GRAPH_PATH}")
except Exception as e:
    print(f"Erro ao salvar o gráfico V2: {e}")
    exit()

print("Geração do gráfico V2 concluída.")

