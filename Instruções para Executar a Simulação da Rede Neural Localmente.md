# Instruções para Executar a Simulação da Rede Neural Localmente

Para executar a simulação da rede neural do Chiller em seu próprio computador, você precisará dos seguintes arquivos e de um ambiente Python configurado.

## Arquivos Essenciais

Você precisará dos seguintes arquivos, que foram gerados durante o projeto:

1.  **Modelo Treinado:** `chiller_model_python.keras` - Este arquivo contém a rede neural treinada.
2.  **Normalizador (Scaler):** `scaler_python.pkl` - Este arquivo é usado para normalizar os dados de entrada da mesma forma que foram normalizados durante o treinamento.
3.  **Scripts de Simulação (Escolha um conforme a necessidade):**
    *   `simulate_nn_python.py`: Para simular um único conjunto de valores de entrada (os valores são definidos dentro do próprio script).
    *   `simulate_100_scenarios.py`: Para gerar e simular 100 cenários com valores aleatórios (gera os dados internamente).
    *   `run_sbc_simulation_inference.py`: Para rodar a simulação usando um arquivo CSV como entrada (requer um arquivo como `simulacao_sbc_temperaturas_reais_input.csv`).

## Configuração do Ambiente Python

1.  **Instalar Python:** Certifique-se de ter o Python instalado em seu computador (versão 3.9 ou superior recomendada).
2.  **Instalar Bibliotecas:** Você precisará instalar as bibliotecas Python necessárias. Abra o terminal ou prompt de comando e execute:
    ```bash
    pip install tensorflow pandas scikit-learn numpy
    ```
    *Observação: Dependendo da sua configuração, pode ser necessário usar `pip3` em vez de `pip`.*

## Como Executar a Simulação

1.  **Organize os Arquivos:** Coloque os arquivos `.keras`, `.pkl` e o script `.py` que você deseja executar na mesma pasta (diretório) no seu computador.
2.  **Prepare os Dados (se necessário):**
    *   Para `simulate_nn_python.py`: Edite o script para alterar os valores de entrada diretamente na seção `input_data`.
    *   Para `simulate_100_scenarios.py`: Nenhuma preparação de dados é necessária, ele gera aleatoriamente.
    *   Para `run_sbc_simulation_inference.py`: Certifique-se de ter o arquivo CSV de entrada (ex: `simulacao_sbc_temperaturas_reais_input.csv`) na mesma pasta e que o nome do arquivo no script (`pd.read_csv(...)`) corresponda ao nome do seu arquivo.
3.  **Execute o Script:** Abra o terminal ou prompt de comando, navegue até a pasta onde você salvou os arquivos e execute o script desejado usando o Python:

    *   Para simular um único cenário:
        ```bash
        python simulate_nn_python.py
        ```
    *   Para simular 100 cenários aleatórios:
        ```bash
        python simulate_100_scenarios.py
        ```
    *   Para simular com dados de um arquivo CSV:
        ```bash
        python run_sbc_simulation_inference.py
        ```

4.  **Verifique a Saída:** Os scripts geralmente imprimem os resultados da simulação (predições da rede neural) diretamente no terminal. O script `simulate_100_scenarios.py` também pode gerar um arquivo CSV com os resultados.

## Observações Adicionais

*   Os scripts de cálculo de carga térmica (`calculate_thermal_load.py`, `calculate_sbc_thermal_load.py`) e de exportação para Excel (`export_to_excel.py`, `export_sbc_results_to_excel.py`) podem ser usados separadamente para processar os resultados das simulações, caso necessário. Eles geralmente leem os arquivos CSV gerados pelas simulações.
*   Certifique-se de que as versões das bibliotecas instaladas localmente sejam compatíveis. As versões usadas no ambiente de desenvolvimento foram TensorFlow 2.19.0, Pandas 2.2.2, Scikit-learn 1.5.1, e NumPy 2.0.0.

Seguindo estas instruções, você deverá conseguir executar as simulações no seu computador.
