# Projeto Aplicado XPE

## Título
Análise de Exoplanetas com Dados da NASA

## Autor
Gustavo Barreto

## Descrição
Este repositório contém os códigos utilizados para o projeto de Análise de Exoplanetas com Dados da NASA.
O objetivo do projeto é desenvolver um modelo de machine learning para identificar exoplanetas potencialmente habitáveis com base em dados coletados da NASA Exoplanet Archive.

## Estrutura do Repositório

- **Scripts**: Scripts Python utilizados para coleta de dados, preparação e treinamento de modelos.
- **Scripts/data/**: Contém os dados brutos e tratados utilizados no projeto.
- **Scripts/modules/**: Contém as bibliotecas utilizadas no projeto.
- **data/models/**: Modelos treinados e salvos em formato `.pkl`.
- **README.md**: Este arquivo.

## Requisitos

- Python 3.8+
- Bibliotecas: 
  - pandas
  - numpy
  - scikit-learn
  - plotly
  - requests
  - boto3

## Como Usar

1. Clone o repositório:
    ```bash
    git clone https://github.com/seuusuario/projeto-exoplanetas-nasa.git
    ```
2. Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```
3. Execute os notebooks ou scripts para replicar as análises e modelos.

## Passos do Projeto

1. **Coleta de Dados**: Utilização da API pública da NASA Exoplanet Archive.
2. **Preparação de Dados**: Tratamento de valores ausentes, normalização e codificação.
3. **Modelagem e Machine Learning**:
   - Seleção de modelos (Random Forest, SVM, etc.)
   - Treinamento e ajuste de hiperparâmetros
   - Avaliação com métricas (precisão, recall, AUC)
4. **Interpretação e Aplicação**: Análise das características dos exoplanetas e aplicação do modelo.
5. **Disponibilização**: Armazenamento dos modelos treinados na AWS S3.
