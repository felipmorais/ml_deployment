import pandas as pd
import numpy as np
import json
import pickle

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

# método para realizar a predição de sobrevivencia do passageiro
def survival_predictor(passageiro):
    # mapeia os valores enviados no json para valores que o modelo ML irá entender
    passageiro['Pclass'] = P_CLASS_MAP[passageiro['Pclass']]
    passageiro['Sex'] = SEX_MAP[passageiro['Sex']]
    passageiro['Embarked'] = EMBARKED_MAP[passageiro['Embarked']]
    
    # transforma o dict do passageiro em um dataframe
    values = pd.DataFrame([passageiro])
    
    # carrega o modelo de predição de sobrevivencia
    model = pickle.load(open('./models/model.pkl', 'rb')) 
    # realiza a predição com base no modelo carregado e nos dados recebidos como parametro da API
    results = model.predict(values)
    
    # futuro: calcular a idade caso o usuário não passe
    # media_idade = data_handler.age_calculator(dados, p_class=1, p_sex='male')
    # print(media_idade)
    
    result = None
    
    # se existir somente um resultado, já o transforma em inteiro e retorna
    if len(results) == 1:
        result = int(results[0])
    
    return result

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
        


