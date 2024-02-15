from fastapi import Body, FastAPI
from pydantic import BaseModel
from typing import Union, Any
import data_handler
import json
import pandas as pd
import pickle

# esse arquivo representa a API que vai ficar rodando para servir a aplicação principal

# para rodar esse arquivo:
# uvicorn main:api --reload

api = FastAPI()

# método simples de exemplo para verificar se a API está rodando e funcionando
@api.get("/hello_world")
async def root():
    return {"message": "Hello World"}

# método especifico para realizar a predição de sobrevivencia de um passageiro
# recebe um json com as informações do passageiro
@api.post("/predict")
def predict(passageiro_json: Any = Body(None)):
    # carrega o json em um python dict
    passageiro = json.loads(passageiro_json)
    
    # mapeia os valores enviados no json para valores que o modelo ML irá entender
    passageiro['Pclass'] = data_handler.P_CLASS_MAP[passageiro['Pclass']]
    passageiro['Sex'] = data_handler.SEX_MAP[passageiro['Sex']]
    passageiro['Embarked'] = data_handler.EMBARKED_MAP[passageiro['Embarked']]
    
    # transforma o dict do passageiro em um dataframe
    values = pd.DataFrame([passageiro])
    
    # carrega o modelo de predição de sobrevivencia
    model = pickle.load(open('./models/model.pkl', 'rb')) 
    # realiza a predição com base no modelo carregado e nos dados recebidos como parametro da API
    results = model.predict(values)
    
    # futuro: calcular a idade caso o usuário não passe
    # media_idade = data_handler.age_calculator(dados, p_class=1, p_sex='male')
    # print(media_idade)
    
    # se existir somente um resultado, já o transforma em inteiro e retorna
    if len(results) == 1:
        return int(results[0])
    return None

# método para retornar todos os dados dos passageiros do titanic
@api.get("/get_titanic_data")
async def get_titanic_data():
    # carrega os dados
    dados = data_handler.load_data()
    # transforma os dados em json
    dados_json = dados.to_json(orient='records')
    # retorna o json para ser enviado via API
    return dados_json

# método para salvar a predição
# recebe como parametro um json com todas as informações do passageiro e do resultado da predição
@api.post("/save_prediction")
async def save_prediction(passageiro_json: Any = Body(None)):
    # transforma o json em um python dict
    passageiro = json.loads(passageiro_json)
    # salva a predição no arquivo já existente de predições realizadas
    result = data_handler.save_prediction(passageiro)
    # retorna o resultado
    return result

# método para retornar todas as predições já realizadas e salvas
@api.get("/get_all_predictions")
async def get_all_predictions():
    # carrega a lista de predições
    all_predictions = data_handler.get_all_predictions()
    # retorna essa lista
    return all_predictions