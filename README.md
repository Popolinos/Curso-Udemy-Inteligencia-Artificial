# Curso-Udemy-Formacao-Completa-Inteligencia-Artificial

# Análise de Renda e Previsão com Redes Neurais

Este projeto visa construir um modelo de Machine Learning para prever a renda de indivíduos com base em diversas características demográficas e de emprego. 
Utilizando uma Rede Neural, o objetivo é classificar se a renda de um indivíduo é `>50K` ou `<=50K`.

## Sumário do Projeto

O dataset utilizado contém informações como idade, classe de trabalho, nível de educação, estado civil, ocupação, raça, gênero, ganhos de capital, perdas de capital, 
horas de trabalho por semana e país de origem. As principais etapas do projeto incluem:

1.  **Carregamento e Visualização dos Dados**: Importação dos dados de treino, teste e validação, e uma análise inicial para entender a estrutura e tipos de dados.
2.  **Pré-processamento dos Dados**: Limpeza, padronização e transformação dos dados para prepará-los para o modelo de Machine Learning.
3.  **Detecção e Tratamento de Anomalias**: Identificação e tratamento de outliers em colunas numéricas.
4.  **Balanceamento de Classes**: Redução da quantidade de exemplos da classe majoritária para evitar viés do modelo.
5.  **Criação e Treinamento do Modelo**: Implementação e treinamento de uma Rede Neural.
6.  **Avaliação do Modelo**: Análise do desempenho do modelo utilizando métricas como Acurácia, F1-Score, Recall e Matriz de Confusão.

## Pré-processamento dos Dados

Foi desenvolvida uma função `padronizar_dataset` que realiza as seguintes transformações:

*   **Conversão de Tipos**: Colunas numéricas inteiras são convertidas para float.
*   **Binning de Variáveis Categóricas**: Categorias raras ou ambíguas em `workclass`, `marital-status`, `occupation`, `native-country`, `race` e `education` são agrupadas em categorias mais amplas (ex: 'Other', 'Basic_School', 'Upper_School').
*   **Categorização de Horas de Trabalho**: A coluna `hours-per-week` é discretizada em categorias ('<40', '40', '>40').
*   **Padronização de Colunas Numéricas**: `MinMaxScaler` é aplicado para normalizar as colunas numéricas.
*   **Codificação de Variáveis Categóricas**: `LabelEncoder` é usado para colunas como workclass, education, marital-status. `OneHotEncoder` é aplicado a native-country e race.

### Tratamento de Outliers e Balanceamento de Classes

*   **`fnlwgt`**: Outliers identificados via Z-Score (>3.2 desvios padrão) foram substituídos pela média da coluna.
*   **Balanceamento de `income`**: Para lidar com o desequilíbrio de classes na variável alvo (`income`), um número significativo de linhas da classe `<=50K` foi removido aleatoriamente do conjunto de treino.

## Modelo de Rede Neural

Foi construído um modelo de Rede Neural Sequencial usando Keras, composto por:

*   Duas camadas `Dense` com ativação `relu` (128 e 32 neurônios, respectivamente).
*   Camadas `Dropout` (0.2 e 0.3) para prevenir overfitting.
*   Uma camada de saída `Dense` com ativação `sigmoid` para classificação binária.

O modelo foi compilado com o otimizador `adam`, função de perda `binary_crossentropy` e as métricas de `accuracy`, `Precision` e `Recall`.

## Treinamento e Avaliação

O modelo foi treinado com `EarlyStopping` monitorando a `val_loss` com paciência de 10 épocas para evitar overfitting. Diversas configurações de pré-processamento foram testadas, e os resultados foram comparados usando:

*   **Acurácia**: Proporção de previsões corretas.
*   **F1-Score**: Média harmônica da precisão e recall.
*   **Recall**: Capacidade do modelo de identificar corretamente todas as instâncias positivas.
*   **Matriz de Confusão**: Visualização do desempenho do modelo, mostrando True Positives, True Negatives, False Positives e False Negatives.

### Comparação dos Testes

| Teste            | Acurácia | F1-Score | Recall  |
| :--------------- | :------- | :------- | :------ |
| Primeiro Teste   | 0.8403   | 0.5883   | 0.4816  |
| Segundo Teste    | 0.8320   | 0.5583   | 0.4482  |
| Terceiro Teste   | 0.8533   | 0.6586   | 0.5974  |
| Quarto Teste     | 0.8331   | 0.6937   | 0.7978  |

**Descrição dos Testes:**

*   **Primeiro Teste**: Linha de base, dados originais com algumas conversões.
*   **Segundo Teste**: Outliers tratados com o valor da moda (ao invés da média).
*   **Terceiro Teste**: Sem tratamento de outliers.
*   **Quarto Teste**: Tratamento de outlier na coluna 'fnlwgt', binning na coluna 'education', e remoção de linhas em income para balanceamento.

O `Quarto Teste` apresentou o melhor `F1-Score` e `Recall`, indicando um bom equilíbrio entre precisão e a capacidade de identificar corretamente a classe positiva, o que é crucial em cenários de classes desbalanceadas. A acurácia geral permaneceu competitiva.

### Avaliação Final no Conjunto de Validação (Quarto Teste)

*   **Acurácia**: 0.8179
*   **F1-Score**: 0.6599
*   **Recall**: 0.7684

**Matriz de Confusão:**

|           | Predito <=50K | Predito >50K |
| :-------- | :------------ | :----------- |
| Real <=50K | 4698          | 944          |
| Real >50K  | 390           | 1294         |

