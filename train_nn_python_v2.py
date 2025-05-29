# -*- coding: utf-8 -*-
"""
Script Python para treinar uma rede neural V2 para o projeto Chiller.

Versão 2: Modificado para prever o 'setpoint_final_otimo_c' diretamente,
sem usar 'setpoint_inicial_c' como feature para essa previsão.

Este script realiza as seguintes etapas:
1. Carrega o dataset sintético V2 gerado ('chiller_dataset_v2.csv').
2. Prepara os dados:
    - Separa as features (X - excluindo setpoint_inicial_c) e os alvos (y1: setpoint_final_otimo_c, y2: comando_habilitacao).
    - Normaliza as features numéricas.
    - Divide os dados em conjuntos de treinamento e teste.
3. Define a arquitetura da rede neural com múltiplas saídas (ajustada para 7 features).
4. Compila o modelo.
5. Treina o modelo.
6. Salva o novo modelo treinado (v2) e o novo scaler (v2).
7. Gera e salva gráficos do histórico de treinamento.
"""

# Bloco 0: Configuração do Matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Bloco 1: Importação de Bibliotecas
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import pickle
import sys

print("TensorFlow version:", tf.__version__)

# Bloco 2: Carregamento e Preparação dos Dados V2
DATASET_PATH = 'chiller_dataset_v2.csv'
MODEL_SAVE_PATH = 'chiller_model_python_v2.keras'
SCALER_SAVE_PATH = 'scaler_python_v2.pkl'
PLOT_SAVE_PATH = 'training_history_python_v2.png'

try:
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset '{DATASET_PATH}' carregado com sucesso.")
except FileNotFoundError:
    print(f"Erro: Arquivo '{DATASET_PATH}' não encontrado. Execute o script de geração de dados V2 primeiro.")
    exit()

# Definindo as features (X) - EXCLUINDO 'setpoint_inicial_c'
feature_columns = [
    'temperatura_ambiente_c',
    'temperatura_entrada_chiller_c',
    'temperatura_saida_chiller_c',
    'pressao_agua_saida_bar',
    # 'setpoint_inicial_c', # Removido das features de entrada do modelo
    'status_bomba_primaria',
    'status_bomba_secundaria',
    'status_valvula_bloqueio'
]
X = df[feature_columns]

# Definindo os alvos (y)
y1 = df['setpoint_final_otimo_c'] # Novo alvo para regressão
y2 = df['comando_habilitacao']

print(f"Shape das features (X): {X.shape}")
print(f"Shape do alvo y1 (setpoint_final_otimo_c): {y1.shape}")
print(f"Shape do alvo y2 (comando_habilitacao): {y2.shape}")

# Bloco 3: Normalização das Features e Divisão dos Dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

try:
    with open(SCALER_SAVE_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler V2 salvo em '{SCALER_SAVE_PATH}'")
except Exception as e:
    print(f"Erro ao salvar o scaler V2: {e}")

X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    X_scaled, y1, y2, test_size=0.2, random_state=42
)

print(f"Shape de X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"Shape de y1_train: {y1_train.shape}, y1_test: {y1_test.shape}")
print(f"Shape de y2_train: {y2_train.shape}, y2_test: {y2_test.shape}")


# Bloco 4: Definição da Arquitetura da Rede Neural V2
# Ajustar a camada de entrada para o novo número de features (7)
input_layer = Input(shape=(X_train.shape[1],), name='input_features') # Shape agora é 7

# Camadas compartilhadas (mantendo a estrutura anterior)
shared_dense_1 = Dense(128, activation='relu', kernel_regularizer=l2(0.001), name='shared_dense_1')(input_layer)
shared_dropout_1 = Dropout(0.3, name='shared_dropout_1')(shared_dense_1)
shared_dense_2 = Dense(64, activation='relu', kernel_regularizer=l2(0.001), name='shared_dense_2')(shared_dropout_1)
shared_dropout_2 = Dropout(0.3, name='shared_dropout_2')(shared_dense_2)

# Ramificação para a saída de regressão (setpoint_final_otimo_c)
output_regression_dense = Dense(32, activation='relu', kernel_regularizer=l2(0.001), name='regression_dense')(shared_dropout_2)
output_regression = Dense(1, activation='linear', name='output_setpoint_final')(output_regression_dense)

# Ramificação para a saída de classificação (comando_habilitacao)
output_classification_dense = Dense(32, activation='relu', kernel_regularizer=l2(0.001), name='classification_dense')(shared_dropout_2)
output_classification = Dense(1, activation='sigmoid', name='output_comando_habilitacao')(output_classification_dense)

model = Model(inputs=input_layer, outputs=[output_regression, output_classification], name='chiller_control_model_v2')
model.summary()

# Bloco 5: Compilação do Modelo V2
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss={
        'output_setpoint_final': 'mean_squared_error',
        'output_comando_habilitacao': 'binary_crossentropy'
    },
    metrics={
        'output_setpoint_final': 'mean_absolute_error',
        'output_comando_habilitacao': 'accuracy'
    },
    loss_weights={'output_setpoint_final': 1.0, 'output_comando_habilitacao': 1.0}
)
print("Modelo V2 compilado com sucesso.")

# Bloco 6: Treinamento do Modelo V2
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

print("Iniciando o treinamento do modelo V2...")
history = model.fit(
    X_train,
    {'output_setpoint_final': y1_train, 'output_comando_habilitacao': y2_train},
    epochs=200,
    batch_size=32,
    validation_data=(X_test, {'output_setpoint_final': y1_test, 'output_comando_habilitacao': y2_test}),
    callbacks=[early_stopping],
    verbose=1
)
print("Treinamento V2 concluído.")

# Bloco 7: Avaliação do Modelo V2
print("\nAvaliando o modelo V2 com dados de teste...")
results = model.evaluate(
    X_test,
    {'output_setpoint_final': y1_test, 'output_comando_habilitacao': y2_test},
    verbose=0
)
print(f"Perda total no teste: {results[0]:.4f}")
print(f"Perda (MSE) - Setpoint Final: {results[1]:.4f}")
print(f"Perda (BinaryCrossentropy) - Comando Habilitação: {results[2]:.4f}")
print(f"MAE - Setpoint Final: {results[3]:.4f}")
print(f"Acurácia - Comando Habilitação: {results[4]*100:.2f}%")

# Bloco 8: Salvamento do Modelo V2
try:
    model.save(MODEL_SAVE_PATH)
    print(f"\nModelo V2 salvo como '{MODEL_SAVE_PATH}'")
except Exception as e:
    print(f"Erro ao salvar o modelo V2: {e}")

# Bloco 9: Visualização do Histórico de Treinamento V2
try:
    print("\nGerando gráficos do histórico de treinamento V2...")
    history_dict = history.history

    # Verificar chaves (nomes podem mudar ligeiramente com base nos nomes das saídas)
    required_keys = [
        'loss', 'val_loss',
        'output_setpoint_final_mean_absolute_error', 'val_output_setpoint_final_mean_absolute_error',
        'output_comando_habilitacao_accuracy', 'val_output_comando_habilitacao_accuracy',
        'output_comando_habilitacao_loss', 'val_output_comando_habilitacao_loss'
    ]
    missing_keys = [key for key in required_keys if key not in history_dict]
    if missing_keys:
        print(f"Erro: Chaves ausentes no histórico de treinamento V2: {missing_keys}")
        print(f"Chaves disponíveis: {list(history_dict.keys())}")
        # Tentar plotar com as chaves disponíveis se possível
        mae_key = 'output_setpoint_final_mean_absolute_error'
        val_mae_key = 'val_output_setpoint_final_mean_absolute_error'
        acc_key = 'output_comando_habilitacao_accuracy'
        val_acc_key = 'val_output_comando_habilitacao_accuracy'
        loss_key = 'output_comando_habilitacao_loss'
        val_loss_key = 'val_output_comando_habilitacao_loss'
        # Ajustar se os nomes forem diferentes (ex: sem o prefixo 'output_')
        if mae_key not in history_dict and 'mean_absolute_error' in history_dict:
             mae_key = 'mean_absolute_error'
             val_mae_key = 'val_mean_absolute_error'
        if acc_key not in history_dict and 'accuracy' in history_dict:
             acc_key = 'accuracy'
             val_acc_key = 'val_accuracy'
        if loss_key not in history_dict and 'binary_crossentropy' in history_dict:
             loss_key = 'binary_crossentropy'
             val_loss_key = 'val_binary_crossentropy'
        # Verificar novamente se as chaves ajustadas existem
        adjusted_keys = ['loss', 'val_loss', mae_key, val_mae_key, acc_key, val_acc_key, loss_key, val_loss_key]
        if not all(key in history_dict for key in adjusted_keys):
             print("Erro: Mesmo após ajuste, chaves essenciais para plotagem não encontradas.")
             raise KeyError("Chaves de histórico ausentes")
    else:
        mae_key = 'output_setpoint_final_mean_absolute_error'
        val_mae_key = 'val_output_setpoint_final_mean_absolute_error'
        acc_key = 'output_comando_habilitacao_accuracy'
        val_acc_key = 'val_output_comando_habilitacao_accuracy'
        loss_key = 'output_comando_habilitacao_loss'
        val_loss_key = 'val_output_comando_habilitacao_loss'

    # Plotagem
    plt.figure(figsize=(12, 10))

    # Perda Total
    plt.subplot(2, 2, 1)
    plt.plot(history_dict['loss'], label='Perda Treino Total')
    plt.plot(history_dict['val_loss'], label='Perda Validação Total')
    plt.title('Perda Total do Modelo V2')
    plt.xlabel('Época')
    plt.ylabel('Perda')
    plt.legend()
    plt.grid(True)

    # MAE - Setpoint Final
    plt.subplot(2, 2, 2)
    plt.plot(history_dict[mae_key], label='MAE Treino (Setpoint Final)')
    plt.plot(history_dict[val_mae_key], label='MAE Validação (Setpoint Final)')
    plt.title('MAE - Setpoint Final Ótimo')
    plt.xlabel('Época')
    plt.ylabel('MAE (°C)')
    plt.legend()
    plt.grid(True)

    # Acurácia - Comando Habilitação
    plt.subplot(2, 2, 3)
    plt.plot(history_dict[acc_key], label='Acurácia Treino (Comando)')
    plt.plot(history_dict[val_acc_key], label='Acurácia Validação (Comando)')
    plt.title('Acurácia - Comando Habilitação')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.grid(True)

    # Perda - Comando Habilitação (Binary Crossentropy)
    plt.subplot(2, 2, 4)
    plt.plot(history_dict[loss_key], label='Perda Treino (Comando)')
    plt.plot(history_dict[val_loss_key], label='Perda Validação (Comando)')
    plt.title('Perda - Comando Habilitação (Binary Crossentropy)')
    plt.xlabel('Época')
    plt.ylabel('Perda')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(PLOT_SAVE_PATH)
    print(f"Gráficos V2 salvos como '{PLOT_SAVE_PATH}'")

except Exception as e:
    print(f"Erro ao gerar ou salvar os gráficos V2: {e}")

print("\nScript de treinamento Python V2 concluído.")

