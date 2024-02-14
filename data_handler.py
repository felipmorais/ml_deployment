import pandas as pd
import numpy as np
import json

# arquivo voltado para acesso e manipulação dos dados do titanic e das predições

# data mapping

# # Mapeia o 'Sexo' e 'Embarcado' para valores numéricos.
# dados['Sex'] = dados['Sex'].map({'male':0, 'female':1})
# dados['Embarked'] = dados['Embarked'].map({'C':0, 'Q':1, 'S':2}

P_CLASS_MAP = {
    '1st': 1, 
    '2nd': 2, 
    '3rd': 3
}

SEX_MAP = {
    'Male': 0,
    'Female': 1,
}

EMBARKED_MAP = {
    'Cherbourg': 0, 
    'Queenstown': 1, 
    'Southampton': 2
}

# realiza a carga dos dados do arquivo CSV do titanic para um dataframe pandas
def load_data():
    # faz a leitura do conjunto de dados
    dados = pd.read_csv('./data/titanic.csv')
    # depois de ler os dados, remove a coluna PassengerId
    dados = dados.drop(['PassengerId'], axis=1)
    return dados


# retorna todos os dados já armazenados das predições realizadas e validadas pelo usuário
# TODO: verificar se o arquivo existe antes de abrir
def get_all_predictions():
    data = None
    with open('predictions.json', 'r') as f:
        data = json.load(f)
    return data

# salva as predições em um arquivo JSON
# TODO: verificar se já não está salvo no arquivo antes de salvar de novo
def save_prediction(passageiro):
    try:
        # le todos as predições
        data = get_all_predictions()
        
        # adiciona a nova predição nos dados já armazenados
        data.append(passageiro)
        
        # salva todas as predições no arquivo json
        with open('predictions.json', 'w') as f:
            json.dump(data, f)
            
        return True
    except Exception as e:
        print(f'Exception during save_prediction: {e}')
        return False
        


