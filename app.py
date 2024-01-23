import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import util
import requests
import json

# verifica se a senha de acesso está correta
if not util.check_password():
    # se a senha estiver errada, para o processamento do app
    st.stop()


# Aqui começa a estrutura do App que vai ser executado em produção (nuvem AWS)
# dados = data_handler.load_data()
response = requests.get("http://localhost:8000/get_titanic_data") 
dados = None
if response.status_code == 200:
    dados_json = json.loads(response.json())
    dados = pd.DataFrame(dados_json)
else: 
    print("Error fetching the data")
    



st.title('App dos dados do Titanic')

data_analyses_on = st.toggle('Exibir análise dos dados')

if data_analyses_on:
    st.header('Dados do Titanic - Dataframe')
    st.dataframe(dados)

    st.header('Histograma das idades')
    fig = plt.figure()
    plt.hist(dados['Age'], bins=10)
    plt.xlabel('Idade')
    plt.ylabel('Quantidade')
    st.pyplot(fig)


    st.header('Sobreviventes')
    st.bar_chart(dados.Survived.value_counts())
    
st.header('Preditor de sobrevivência')

# ler as seguintes informações de input:
# Pclass - '1st', '2nd', or '3rd'
# Sex - 'male' or 'female'
# Age - int
# SibSp - int
# Parch - int
# Fare - float
# Embarked - C = Cherbourg, Q = Queenstown, S = Southampton

# define a linha 1 de inputs
col1, col2, col3 = st.columns(3)
with col1:
    # pclass: A proxy for socio-economic status (SES)
    # 1st = Upper
    # 2nd = Middle
    # 3rd = Lower
    classes = ['1st', '2nd', '3rd']
    p_class = st.selectbox('Ticket class', classes)
    
with col2:
    classes = ['Male', 'Female']
    sex = st.selectbox('Sex', classes)
    
with col3:
    # age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
    age = st.number_input('Age in years', step=1)
    

# define a linha 2 de inputs
col1, col2, col3 = st.columns(3)
with col1:
    # sibsp: The dataset defines family relations in this way...
    # Sibling = brother, sister, stepbrother, stepsister
    # Spouse = husband, wife (mistresses and fiancés were ignored)
    sib_sp = st.number_input('Number of siblings / spouses aboard the Titanic', step=1)
    
with col2:
    # parch: The dataset defines family relations in this way...
    # Parent = mother, father
    # Child = daughter, son, stepdaughter, stepson
    # Some children travelled only with a nanny, therefore parch=0 for them.
    par_ch = st.number_input('Number of parents / children aboard the Titanic', step=1)

with col3:
    fare = st.number_input('Passenger fare')
    
    
# define a linha 3 de inputs
col1, col2 = st.columns(2)
with col1:
    classes = ['Cherbourg', 'Queenstown', 'Southampton']
    embarked = st.selectbox('Port of Embarkation', classes)
    
with col2:
    submit = st.button('Verificar')


passageiro = {}

    
if submit or 'survived' in st.session_state:

    passageiro = {
        'Pclass': p_class,
        'Sex': sex,
        'Age': age,
        'SibSp': sib_sp,
        'Parch': par_ch,
        'Fare': fare,
        'Embarked': embarked
    }

    # values = pd.DataFrame([passageiro])
    # results = model.predict(values)
    
    passageiro_json = json.dumps(passageiro)
    response = requests.post("http://localhost:8000/predict", json=passageiro_json) 
    result = None
    if response.status_code == 200:
        result = response.json()
    else: 
        print("Error fetching the data")
        
    if result is not None:
        survived = result
        if survived == 1:
            st.subheader('Passageiro SOBREVIVEU! 😃🙌🏻')
            if 'survived' not in st.session_state:
                st.balloons() 
        else:
            st.subheader('Passageiro NÃO sobreviveu! 😢')
            if 'survived' not in st.session_state:
                st.snow()
            
        st.session_state['survived'] = survived
            
    if passageiro and 'survived' in st.session_state:
        st.write("A predição está correta?")
        col1, col2, col3 = st.columns([1,1,5])
        with col1:
            correct_prediction = st.button('👍🏻')
        with col2:
            wrong_prediction = st.button('👎🏻')
            
        if correct_prediction or wrong_prediction:
            message = "Muito obrigado pelo feedback"
            if wrong_prediction:
                message += ", iremos usar esses dados para melhorar as predições"
            message += "."
            
            if correct_prediction:
                passageiro['CorrectPrediction'] = True
            elif wrong_prediction:
                passageiro['CorrectPrediction'] = False
                
            passageiro['Survived'] = st.session_state['survived']
            
            st.write(message)
            
            # data_handler.save_prediction(passageiro)
            passareiro_json = json.dumps(passageiro)
            response = requests.post("http://localhost:8000/save_prediction", json=passareiro_json)
            if response.status_code == 200:
                print("Predictions saved")
            else: 
                print("Error fetching the data")
            
    st.write('')
    col1, col2, col3 = st.columns(3)
    with col2:
        new_test = st.button('Iniciar Nova Análise')
        
        if new_test and 'survived' in st.session_state:
            del st.session_state['survived']
            st.rerun()

accuracy_predictions_on = st.toggle('Exibir acurácia')

if accuracy_predictions_on:
    # predictions = data_handler.get_all_predictions()
    response = requests.get("http://localhost:8000/get_all_predictions") 
    predictions = None
    if response.status_code == 200:
        predictions = response.json()
    else: 
        print("Error fetching the data")
        
    num_total_predictions = len(predictions)
    
    accuracy_hist = [0]
    correct_predictions = 0
    for index, passageiro in enumerate(predictions):
        total = index + 1
        if passageiro['CorrectPrediction'] == True:
            correct_predictions += 1
            
        temp_accuracy = correct_predictions / total if total else 0
        accuracy_hist.append(round(temp_accuracy, 2)) 
    
    accuracy = correct_predictions / num_total_predictions if num_total_predictions else 0
    
    st.write(f'Accuracy: {round(accuracy, 2)}')
    
    st.subheader("Histórico de acurácia")
    st.line_chart(accuracy_hist)
    
