import pandas as pd
import numpy as np
import json

def load_data():
    # faz a leitura do conjunto de dados
    dados = pd.read_csv('./data/titanic.csv')
    dados = dados.drop(['PassengerId'], axis=1)
    return dados

def age_calculator(dados, p_class, p_sex):
    # Impute a idade pela média da classe
    class_age_means = dados.groupby(['Pclass','Sex'])['Age'].mean()
    dict_class_age_means = dict(class_age_means)
    avg_age = dict_class_age_means[(p_class, p_sex)]
    return avg_age

# retorna todos os dados já armazenados
# TODO: verificar se o arquivo existe antes de abrir
def get_all_predictions():
    data = None
    with open('predictions.json', 'r') as f:
        data = json.load(f)
        
    return data

# salva as predições em um arquivo JSON
# TODO: verificar se já não está salvo no arquivo antes de salvar de novo
def save_prediction(passageiro):
    data = get_all_predictions()
        
    data.append(passageiro)
        
    with open('predictions.json', 'w') as f:
        json.dump(data, f)
        


