# -*- coding: utf-8 -*-
"""Gera um gráfico combinado: 
1. Comparativo de carga térmica (Setpoint Original vs. Rede Neural V2 - Forçado).
2. Distribuição da economia/aumento percentual de energia.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg") # Use Agg backend for non-GUI environments
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configurações ---
EXCEL_PATH = "/home/ubuntu/simulacao_forced_v2_final.xlsx"
OUTPUT_GRAPH_PATH = "/home/ubuntu/grafico_comparativo_distribuicao_forced_v2.png"

# --- Carregar Dados ---
print(f"Carregando dados da planilha V2 (forçado): {EXCEL_PATH}")
try:
    df = pd.read_excel(EXCEL_PATH)
    print("Dados V2 (forçado) carregados com sucesso.")
except FileNotFoundError:
    print(f"Erro: Arquivo {EXCEL_PATH} não encontrado.")
    exit()
except Exception as e:
    print(f"Erro ao carregar a planilha V2 (forçado): {e}")
    exit()

# --- Filtrar Dados ---
# Considerar apenas os cenários onde a rede neural V2 habilitou o chiller
df_filtered = df[df["comando_habilitacao_rede_v2"] == 1].copy()

if df_filtered.empty:
    print("Erro: Nenhum cenário encontrado onde a rede neural V2 habilitou o chiller (com inputs forçados). Não é possível gerar o gráfico.")
    exit()

print(f"Número de cenários com Chiller habilitado pela rede V2 (forçado): {len(df_filtered)}")

# Adicionar um índice para o eixo X do gráfico de linhas
df_filtered["cenario_index"] = range(1, len(df_filtered) + 1)

# --- Gerar Gráfico Combinado ---
print("Gerando gráfico combinado V2 (forçado)...")
plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(2, 1, figsize=(14, 12)) # 2 linhas, 1 coluna

# --- Gráfico Superior: Comparativo de Carga Térmica ---
ax1 = axes[0]
sns.lineplot(data=df_filtered, x="cenario_index", y="carga_termica_kw_inicial_original", label="Carga Térmica (Setpoint Original)", marker=".", linestyle="-", color="red", ax=ax1)
sns.lineplot(data=df_filtered, x="cenario_index", y="carga_termica_kw_rede_v2", label="Carga Térmica (Setpoint Rede Neural V2)", marker=".", linestyle="-", color="green", ax=ax1)

# Calcular e exibir economia total no título do subplot
carga_total_original = df_filtered["carga_termica_kw_inicial_original"].sum()
carga_total_rede_v2 = df_filtered["carga_termica_kw_rede_v2"].sum()
diferenca_total = carga_total_original - carga_total_rede_v2

if diferenca_total > 0:
    economia_texto = f"Economia Total (Rede V2 vs Original): {diferenca_total:.2f} kW"
else:
    economia_texto = f"Aumento Consumo (Rede V2 vs Original): {-diferenca_total:.2f} kW"

ax1.set_title(f"Comparativo de Carga Térmica (Equipamentos Forçados ON)\n({economia_texto})")
ax1.set_xlabel("Cenário (Chiller Habilitado pela Rede V2)")
ax1.set_ylabel("Carga Térmica (kW)")
ax1.legend()
ax1.grid(True)

# --- Gráfico Inferior: Distribuição Percentual da Economia ---
ax2 = axes[1]
# Remover infinitos caso existam (onde carga inicial era 0)
df_plot_dist = df_filtered[np.isfinite(df_filtered["diferenca_percentual_v2"])].copy()

if not df_plot_dist.empty:
    sns.histplot(data=df_plot_dist, x="diferenca_percentual_v2", kde=True, ax=ax2, bins=15)
    # Adicionar linha vertical no 0 para referência
    ax2.axvline(0, color='k', linestyle='--', linewidth=1) # Corrigido
    ax2.set_title("Distribuição da Diferença Percentual de Consumo (Rede V2 vs Original)")
    ax2.set_xlabel("Diferença Percentual (%) [Positivo = Economia da Rede V2]")
    ax2.set_ylabel("Contagem de Cenários")
    ax2.grid(True)
else:
    # Corrigido
    ax2.text(0.5, 0.5, "Não há dados válidos para exibir a distribuição percentual.", 
             horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
    ax2.set_title("Distribuição da Diferença Percentual de Consumo")
    ax2.set_xlabel("Diferença Percentual (%)")
    ax2.set_ylabel("Contagem de Cenários")

# Ajustar layout e salvar
plt.tight_layout()
try:
    plt.savefig(OUTPUT_GRAPH_PATH)
    print(f"Gráfico combinado V2 (forçado) salvo como: {OUTPUT_GRAPH_PATH}")
except Exception as e:
    print(f"Erro ao salvar o gráfico combinado V2 (forçado): {e}")
    exit()

print("Geração do gráfico combinado V2 (forçado) concluída.")

