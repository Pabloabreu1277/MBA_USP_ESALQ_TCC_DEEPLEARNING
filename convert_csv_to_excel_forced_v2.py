# -*- coding: utf-8 -*-
"""Converte o arquivo CSV da simulação V2 forçada para Excel."""

import pandas as pd

CSV_PATH = "/home/ubuntu/simulacao_forced_v2_temp.csv"
EXCEL_PATH = "/home/ubuntu/simulacao_forced_v2_final.xlsx"

print(f"Lendo arquivo CSV V2 (forçado): {CSV_PATH}")
try:
    df = pd.read_csv(CSV_PATH)
    print("CSV V2 (forçado) lido com sucesso.")
except FileNotFoundError:
    print(f"Erro: Arquivo CSV V2 (forçado) {CSV_PATH} não encontrado.")
    exit()
except Exception as e:
    print(f"Erro ao ler o CSV V2 (forçado): {e}")
    exit()

print(f"Salvando como arquivo Excel V2 (forçado): {EXCEL_PATH}")
try:
    df.to_excel(EXCEL_PATH, index=False, engine='openpyxl')
    print("Arquivo Excel V2 (forçado) salvo com sucesso.")
except Exception as e:
    print(f"Erro ao salvar o Excel V2 (forçado): {e}")
    exit()

print("Conversão V2 (forçado) para Excel concluída.")

