from fastapi import Body, FastAPI
from pydantic import BaseModel
from typing import Union, Any
import data_handler
import json
import pandas as pd
import pickle

# to run this file:
# uvicorn main:api --reload

api = FastAPI()

# @api.get("/")
# async def root():
#     return {"message": "Hello World"}

@api.post("/predict")
def predict(passageiro_json: Any = Body(None)):
    passageiro = json.loads(passageiro_json)
    
    passageiro['Pclass'] = data_handler.P_CLASS_MAP[passageiro['Pclass']]
    passageiro['Sex'] = data_handler.SEX_MAP[passageiro['Sex']]
    passageiro['Embarked'] = data_handler.EMBARKED_MAP[passageiro['Embarked']]
    
    values = pd.DataFrame([passageiro])
    
    model = pickle.load(open('./models/model.pkl', 'rb')) 
    results = model.predict(values)
    
    # futuro: calcular a idade caso o usuário não passe
    # media_idade = data_handler.age_calculator(dados, p_class=1, p_sex='male')
    # print(media_idade)
    
    if len(results) == 1:
        return int(results[0])
    return None

@api.get("/get_titanic_data")
async def get_titanic_data():
    dados = data_handler.load_data()
    dados_json = dados.to_json(orient='records')
    return dados_json

@api.post("/save_prediction")
async def save_prediction(passageiro_json: Any = Body(None)):
    passageiro = json.loads(passageiro_json)
    result = data_handler.save_prediction(passageiro)
    return result

@api.get("/get_all_predictions")
async def get_all_predictions():
    all_predictions = data_handler.get_all_predictions()
    return all_predictions