import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import util
import data_handler
import pickle

# verifica se a senha de acesso está correta
if not util.check_password():
    # se a senha estiver errada, para o processamento do app
    st.stop()


# Aqui começa a estrutura do App que vai ser executado em produção (nuvem AWS)

# primeiro de tudo, carrega os dados do titanic para um dataframe
dados = data_handler.load_data()

# carrega o modelo de predição já treinado e validado
model = pickle.load(open('./models/model.pkl', 'rb')) 

# começa a estrutura da interface do sistema
st.title('App dos dados do Titanic')

data_analyses_on = st.toggle('Exibir análise dos dados')

if data_analyses_on:
    # essa parte é só um exmplo de que é possível realizar diversas visualizações e plotagens com o streamlit
    st.header('Dados do Titanic - Dataframe')
    
    # exibe todo o dataframe dos dados do titanic
    st.dataframe(dados)

    # plota um histograma das idades dos passageiros
    st.header('Histograma das idades')
    fig = plt.figure()
    plt.hist(dados['Age'], bins=10)
    plt.xlabel('Idade')
    plt.ylabel('Quantidade')
    st.pyplot(fig)

    # plota um gráfico de barras com a contagem dos sobreviventes
    st.header('Sobreviventes')
    st.bar_chart(dados.Survived.value_counts())
    
# daqui em diante vamos montar a inteface para capturar os dados de input do usuário para realizar a predição
# que vai identificar se um passageiro sobreviveu ou não
st.header('Preditor de sobrevivência')

# ler as seguintes informações de input:
# Pclass - int
# Sex - 'male' or 'female'
# Age - int
# SibSp - int
# Parch - int
# Fare - float
# Embarked - C = Cherbourg, Q = Queenstown, S = Southampton

# essas foram as informações utilizadas para treinar o modelo
# assim, todas essas informações também devem ser passadas para o modelo realizar a predição

# define a linha 1 de inputs com 3 colunas
col1, col2, col3 = st.columns(3)

# captura a p_class do passageiro, com base na lista de classes dispobilizadas 
with col1:
    # pclass: A proxy for socio-economic status (SES)
    # 1st = Upper
    # 2nd = Middle
    # 3rd = Lower
    classes = ['1st', '2nd', '3rd']
    p_class = st.selectbox('Ticket class', classes)

# captura o sex do passageiro, com base na lista de classes disponibilizdas
with col2:
    classes = ['Male', 'Female']
    sex = st.selectbox('Sex', classes)
    
# captura a idade do passageiro, como o step é 1, ele considera a idade como inteira
with col3:
    # age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
    age = st.number_input('Age in years', step=1)
    

# define a linha 2 de inputs, também com 3 colunas
col1, col2, col3 = st.columns(3)

# captura o número de irmãos e esposa(o)
with col1:
    # sibsp: The dataset defines family relations in this way...
    # Sibling = brother, sister, stepbrother, stepsister
    # Spouse = husband, wife (mistresses and fiancés were ignored)
    sib_sp = st.number_input('Number of siblings / spouses aboard the Titanic', step=1)

# captura o número de pais e filhos
with col2:
    # parch: The dataset defines family relations in this way...
    # Parent = mother, father
    # Child = daughter, son, stepdaughter, stepson
    # Some children travelled only with a nanny, therefore parch=0 for them.
    par_ch = st.number_input('Number of parents / children aboard the Titanic', step=1)

# captura o valor pago pela passagem, agora em float
with col3:
    fare = st.number_input('Passenger fare')
    
    
# define a linha 3 de inputs, com 2 colunas
col1, col2 = st.columns(2)

# captura o porto de embarque do passageiro, com base na lista de classes disponibilizadas
with col1:
    classes = ['Cherbourg', 'Queenstown', 'Southampton']
    embarked = st.selectbox('Port of Embarkation', classes)
    
# define o botão de verificar, que deverá ser pressionado para o sistema realizar a predição
with col2:
    submit = st.button('Verificar')

# data mapping
# essa parte do código realiza o mapeamento dos campos p_class, sex e embarked para valores numéricos
# o mesmo procedimento foi realizado durante o treinamento do modelo
# assim, isso também deve ser feito aqui para haver compatibilidade nos dados

# # Mapeia o 'Sexo' e 'Embarcado' para valores numéricos.
# dados['Sex'] = dados['Sex'].map({'male':0, 'female':1})
# dados['Embarked'] = dados['Embarked'].map({'C':0, 'Q':1, 'S':2}

p_class_map = {
    '1st': 1, 
    '2nd': 2, 
    '3rd': 3
}
sex_map = {
    'Male': 0,
    'Female': 1,
}
embarked_map = {
    'Cherbourg': 0, 
    'Queenstown': 1, 
    'Southampton': 2
}

# armazena todos os dados do passageiro nesse dict
passageiro = {}

    
# verifica se o botão submit foi pressionado e se o campo survived está em cache
if submit or 'survived' in st.session_state:
    
    # TODO: verificar se o usuário informou todas as informações do passageiro antes de realizar o processamento dos dados e predição
    # TODO: no dataset possuiam vários dados sem a idade do passageiro, assim deveriamos permitir que a idade não fosse informada e tratar essa falta de informação

    # seta todos os attrs do passsageiro e já realiza o mapeamento dos attrs que não são numéricos
    passageiro = {
        'Pclass': p_class_map[p_class],
        'Sex': sex_map[sex],
        'Age': age,
        'SibSp': sib_sp,
        'Parch': par_ch,
        'Fare': fare,
        'Embarked': embarked_map[embarked]
    }
    print(passageiro)
    
    # converte o passageiro para um pandas dataframe
    # isso é feito para igualar ao tipo de dado que foi utilizado para treinar o modelo
    values = pd.DataFrame([passageiro])
    print(values) 

    # realiza a predição de sobrevivência do passageiro com base nos dados inseridos pelo usuário
    results = model.predict(values)
    print(results)
    
    # o modelo foi treinado para retornar uma lista com 0 ou 1, onde cada posição da lista indica se o passageiro sobreviveu (1) ou não (0)
    # como estamos realizando a predição de somente um passageiro por vez, o modelo deverá retornar somente um elemento na lista
    if len(results) == 1:
        # converte o valor retornado para inteiro
        survived = int(results[0])
        
        # verifica se o passageiro sobreviveu
        if survived == 1:
            # se sim, exibe uma mensagem que o passageiro sobreviveu
            st.subheader('Passageiro SOBREVIVEU! 😃🙌🏻')
            if 'survived' not in st.session_state:
                st.balloons()
        else:
            # se não, exibe uma mensagem que o passageiro não sobreviveu
            st.subheader('Passageiro NÃO sobreviveu! 😢')
            if 'survived' not in st.session_state:
                st.snow()
        
        # salva no cache da aplicação se o passageiro sobreviveu
        st.session_state['survived'] = survived
    
    # verifica se existe um passageiro e se já foi verificado se ele sobreviveu ou não
    if passageiro and 'survived' in st.session_state:
        # se sim, pergunta ao usuário se a predição está certa e salva essa informação
        st.write("A predição está correta?")
        col1, col2, col3 = st.columns([1,1,5])
        with col1:
            correct_prediction = st.button('👍🏻')
        with col2:
            wrong_prediction = st.button('👎🏻')
        
        # exibe uma mensagem para o usuário agradecendo o feedback
        if correct_prediction or wrong_prediction:
            message = "Muito obrigado pelo feedback"
            if wrong_prediction:
                message += ", iremos usar esses dados para melhorar as predições"
            message += "."
            
            # adiciona no dict do passageiro se a predição está correta ou não
            if correct_prediction:
                passageiro['CorrectPrediction'] = True
            elif wrong_prediction:
                passageiro['CorrectPrediction'] = False
                
            # adiciona no dict do passageiro se ele sobreviveu ou não
            passageiro['Survived'] = st.session_state['survived']
            
            # escreve a mensagem na tela
            st.write(message)
            print(message)
            
            # salva a predição no JSON para cálculo das métricas de avaliação do sistema
            data_handler.save_prediction(passageiro)
            
    st.write('')
    # adiciona um botão para permitir o usuário realizar uma nova análise
    col1, col2, col3 = st.columns(3)
    with col2:
        new_test = st.button('Iniciar Nova Análise')
        
        # se o usuário pressionar no botão e já existe um passageiro, remove ele do cache
        if new_test and 'survived' in st.session_state:
            del st.session_state['survived']
            st.rerun()

# calcula e exibe as métricas de avaliação do modelo
# aqui, somente a acurária está sendo usada
# TODO: adicionar as mesmas métricas utilizadas na disciplina de treinamento e validação do modelo (recall, precision, F1-score)
accuracy_predictions_on = st.toggle('Exibir acurácia')

if accuracy_predictions_on:
    # pega todas as predições salvas no JSON
    predictions = data_handler.get_all_predictions()
    # salva o número total de predições realizadas
    num_total_predictions = len(predictions)
    
    # calcula o número de predições corretas e salva os resultados conforme as predições foram sendo realizadas
    accuracy_hist = [0]
    # salva o numero de predições corretas
    correct_predictions = 0
    # percorre cada uma das predições, salvando o total móvel e o número de predições corretas
    for index, passageiro in enumerate(predictions):
        total = index + 1
        if passageiro['CorrectPrediction'] == True:
            correct_predictions += 1
            
        # calcula a acurracia movel
        temp_accuracy = correct_predictions / total if total else 0
        # salva o valor na lista de historico de acuracias
        accuracy_hist.append(round(temp_accuracy, 2)) 
    
    # calcula a acuracia atual
    accuracy = correct_predictions / num_total_predictions if num_total_predictions else 0
    
    # exibe a acuracia atual para o usuário
    st.metric(label='Acurácia', value=round(accuracy, 2))
    # TODO: usar o attr delta do st.metric para exibir a diferença na variação da acurácia
    
    # exibe o histórico da acurácia
    st.subheader("Histórico de acurácia")
    st.line_chart(accuracy_hist)
    
