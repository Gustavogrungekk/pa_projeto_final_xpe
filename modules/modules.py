#==============================================================================================================#
# Author: Gustavo Barreto
# Date: 2024-07-06
# Description: Funções auxiliares para a criação do pipeline para do projeto final 
# Análise de Exoplanetas com Dados da NASA. Para gerar a chave da API e coletar dados de exoplanetas sera 
# necessario acessar o site oficial da NASA https://api.nasa.gov/ e fazer a requisição.
# Github: https://github.com/Gustavogrungekk
# Versão: 1.0
#===============================================================================================================#

import boto3
import requests
import pandas as pd
from time import sleep
import os
import re
import random
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import pickle

def get_exoplanets(api_key):
    """
    Função para acessar a NASA Exoplanet Archive API e coletar dados de exoplanetas.
    
    Parâmetros:
    api_key (str): Sua chave de API da NASA.

    Retorna:
    pandas.DataFrame: DataFrame contendo os dados dos exoplanetas.
    """
    # URL da API para acessar os dados do NASA Exoplanet Archive
    base_url = 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI'
    query_params = {
        'table': 'cumulative',
        'format': 'json',
        'api_key': api_key
    }

    # Salva os dados coletados em um arquivo CSV na raiz do projeto
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Caminho da raiz do projeto
    output_dir = os.path.join(root_dir, 'data')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    while True:
        try:
            # Fazendo a requisição para a API
            response = requests.get(base_url, params=query_params)
            response.raise_for_status()  # Levanta uma exceção para códigos de status 4xx/5xx
            # Lendo os dados da requisição
            exoplanets_data = pd.DataFrame(response.json())
            
            # Salvando os dados em um arquivo CSV
            output_file = os.path.join(output_dir, 'exoplanets_data.csv')
            exoplanets_data.to_csv(output_file, index=False)
            print(f'Requisição concluída. Dados salvos em {output_file}')
            
            return  f'Dados coletados com sucesso em {output_file}'        
        except requests.exceptions.HTTPError as http_err:
            print(f'HTTP error occurred: {http_err}')
            if response.status_code in [401, 403, 404]:
                print('Erro fatal. Verifique sua chave de API ou a URL.')
                return None
            else:
                print('Tentando novamente em 10 segundos...')
                sleep(10)
        except requests.exceptions.RequestException as req_err:
            print(f'Erro de requisição: {req_err}')
            print('Tentando novamente em 10 segundos...')
            sleep(10)
        except ValueError as json_err:
            print(f'Erro ao processar JSON: {json_err}')
            return None
        except Exception as err:
            print(f'Erro inesperado: {err}')
            return None

def dataframe():
    '''
    Função para carregar o dataframe contendo os dados dos exoplanetas.
    '''
    df = pd.read_csv('data/exoplanets_data.csv')
    return df

def celcius(temp:float, from_unit:str) -> float:
    '''
    Converte temperatura para Celsius a partir de diferentes unidades.

    Parâmetros:
    temp (float): Valor da temperatura a ser convertido.
    from_unit (str): Unidade de temperatura de entrada. Pode ser 'K' (Kelvin), 'F' (Fahrenheit) ou 'R' (Rankine). Default é 'K'.

    Retorna:
    float: Temperatura convertida para Celsius.
    '''
    if from_unit.upper() == 'K':
        return temp - 273.15
    elif from_unit.upper() == 'F':
        return (temp - 32) * 5/9
    elif from_unit.upper() == 'R':
        return (temp - 491.67) * 5/9
    else:
        raise ValueError("Unidade de temperatura não suportada. Use 'K' para Kelvin, 'F' para Fahrenheit ou 'R' para Rankine.")
    
def nulos(dataframe, fill: bool = False):
    '''
    Gera um relatório em formato DataFrame sobre a contagem de valores nulos e preenchidos em cada coluna do DataFrame fornecido.
    
    Parâmetros:
    dataframe (pd.DataFrame): O DataFrame contendo os dados a serem analisados.
    fill (bool, opcional): Se True, preenche os valores nulos com a média da coluna. Padrão é False.

    Retorna:
    pd.DataFrame: DataFrame contendo as seguintes colunas:
        - 'Coluna': Nome da coluna do DataFrame original.
        - 'Nulos': Número de valores nulos na coluna.
        - 'Preenchidos': Número de valores preenchidos na coluna após opcionalmente aplicar o preenchimento.
    '''
    resultados = []

    for coluna in dataframe.columns:
        nulos = dataframe[coluna].isna().sum()
        preenchidos = dataframe[coluna].count()

        if fill and nulos > 0:
            dataframe[coluna].fillna(dataframe[coluna].mean(), inplace=True)

        resultados.append({
            'Coluna': coluna,
            'Nulos': nulos,
            'Preenchidos': preenchidos
            })

    return pd.DataFrame(resultados)


def normalizar_treinar_modelo(df, colunas:list, prever:str, salvar:bool = False):
    '''
    Descrição: Função para normalizar os dados e aplicar o modelo de ML. Atualmente conta com 7 modelos de ML.
    Sendo eles Random Forest, Decision Tree, Logistic Regression, K-Nearest Neighbors, SVM e Naive Bayes.
    Argumentos:
        df: Pandas DataFrame
        colunas: Lista de colunas a serem normalizadas
        prever: Coluna dependente a ser prevista
        salvar: Se True, salva o modelo normalizado em um arquivo pickle. Padrão é False.
    Retorna:
    Imprime os resultados da normalização e avaliação do modelo de ML
    '''

    # Inicializando o MinMaxScaler e StandardScaler
    minmax_scaler = MinMaxScaler()
    standard_scaler = StandardScaler()

    # Normalizando os dados com MinMaxScaler
    df[colunas] = minmax_scaler.fit_transform(df[colunas])
    
    # Definindo as variáveis independentes (X) e a variável dependente (y)
    X = df[colunas]
    y = df[prever]  # Supondo que esta coluna indica a habitabilidade

    # Dividindo os dados em treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Criando um dicionário para armazenar os modelos
    modelos = {
        'Random Forest': RandomForestClassifier(), # Random Forest modelo de classificação
        'Decision Tree': DecisionTreeClassifier(), # Decision Tree modelo de classificação
        'Logistic Regression': LogisticRegression(max_iter=1000), # Logistic Regression modelo de regressão
        'K-Nearest Neighbors': KNeighborsClassifier(), # K-Nearest Neighbors modelo de classificação
        'SVM': SVC(probability=True), # SVM modelo de classificação
        'Naive Bayes': GaussianNB(), # Naive Bayes modelo de classificação
    }

    # Treinando e avaliando os modelos
    for nome, modelo in modelos.items():
        try:
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)
            y_pred_proba = modelo.predict_proba(X_test)[:, 1] if hasattr(modelo, "predict_proba") else modelo.decision_function(X_test)
            
            auc = roc_auc_score(y_test, y_pred_proba)

            print(f"Modelo: {nome}")
            print(f"Acurácia: {accuracy_score(y_test, y_pred)}")
            print(f"AUC: {auc}")
            print(classification_report(y_test, y_pred))
            print("\n")
        except ZeroDivisionError:
            print(f"ZeroDivisionError encontrado no modelo: {nome}")
            print("\n")

    # Se for para salvar os modelos
    if salvar:
        # Criando uma pasta de modelos
        if not os.path.exists("data/modelos"):
            os.makedirs("data/modelos")

        # Salvando o modelo em pickle
        for nome, modelo in modelos.items():
            pickle.dump(modelo, open(f"data/modelos/{nome}.pkl", "wb"))


def carregar_modelo(nome_modelo):
    """
    Função para carregar um modelo treinado a partir de um arquivo .pkl. Modelos disponíveis em data/modelos.

    Parâmetros:
    nome_modelo (str): Nome do modelo (sem a extensão .pkl).

    Retorna:
    sklearn model: Modelo carregado pronto para fazer previsões.
    """
    try:
        # Carregando o modelo a partir do arquivo .pkl
        modelo = pickle.load(open(f"data/modelos/{nome_modelo}.pkl", "rb"))
        return modelo
    except FileNotFoundError:
        print(f"Arquivo do modelo {nome_modelo}.pkl não encontrado.")
        return None
    except Exception as e:
        print(f"Erro ao carregar o modelo {nome_modelo}.pkl:", str(e))
        return None

def fazer_previsao(input_exoplaneta, nome_modelo):
    """
    Função para fazer previsões usando um modelo carregado.

    Parâmetros:
    input_exoplaneta (dict or pd.DataFrame): Dados do exoplaneta para fazer a previsão.
    nome_modelo (str): Nome do modelo (sem a extensão .pkl).

    Retorna:
    dict: Dicionário contendo o resultado da previsão ('previsao'), as probabilidades das classes ('probabilidades'),
          e informações adicionais sobre o modelo ('info_modelo').
          Retorna None se o modelo não puder ser carregado.
    """
    # Carregando o modelo
    modelo = carregar_modelo(nome_modelo)
    
    if modelo:
        # Fazendo a previsão
        if isinstance(input_exoplaneta, dict):
            input_exoplaneta = pd.DataFrame([input_exoplaneta])
        elif isinstance(input_exoplaneta, pd.DataFrame):
            pass
        else:
            raise ValueError("O input_exoplaneta deve ser um dicionário ou DataFrame pandas.")

        # Fazendo a previsão e obtendo as probabilidades associadas
        probabilidades = modelo.predict_proba(input_exoplaneta)
        previsao = modelo.predict(input_exoplaneta)[0]
        
        # Informações adicionais sobre o modelo
        info_modelo = {
            'tipo_modelo': type(modelo).__name__,
            'configuracao_modelo': modelo.get_params(),
            'metricas_modelo': None
        }
        
        print(f'Resultado da previsão: {previsao}')
        resultado = {
            'previsao': previsao,
            'probabilidades': probabilidades,
            'info_modelo': info_modelo
        }

        return resultado
    else:
        return None
    
# Agora iremos enviar o modelo para um bucket s3.
def model_to_s3(modelo: str, bucket: str, prefix: str, region: str, aws_access_key_id: str, aws_secret_access_key: str):
    '''
    Função simples para enviar um modelo para um bucket s3.

    Parâmetros:
    modelo (str): Nome do modelo (sem a extensão .pkl).
    bucket (str): Nome do bucket.
    prefix (str): Prefixo para o nome do arquivo no bucket.
    region (str): Região da AWS.
    aws_access_key_id (str): Chave de acesso da AWS.
    aws_secret_access_key (str): Chave de acesso da AWS.
    '''
    s3 = boto3.client('s3', region_name=region, aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
    try:
        s3.upload_file(f'data/modelos/{modelo}.pkl', bucket, f'{prefix}/{modelo}.pkl')
        print(f'Modelo {modelo}.pkl enviado com sucesso para o bucket {bucket}.')
    except Exception as e:
        print(f'Erro ao enviar o modelo {modelo}.pkl para o bucket {bucket}:', str(e))