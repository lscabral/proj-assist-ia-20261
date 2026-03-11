#deploy da aplicação

#imports
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

#funcao para carregar o dataset
#@st.cache
def get_data():
    #carregar os dados
    data = pd.read_csv('dados.csv')
    return data

#funcao para treinar o modelo
def train_model():
    data = get_data()
    X = data.drop('MEDV', axis=1)
    y = data['MEDV']
    model = RandomForestRegressor()
    model.fit(X, y)
    return model

#criando dataframe
data = get_data()

#treinando o modelo
model = train_model()

#criando o website da aplicacao com streamlit
st.title('DataApp - Prevendo Preços de Casas Imóveis')

#descricao
st.markdown("Este é um Data App utilizado para exibir a solução " \
"de Machine Learnin para o problema de predição de valores de imóveis " \
"com o dataset Boston House Prices do MIT.")

#visao do dataset
st.subheader('Selecionando as colunas para visualização')

#selecionando as colunas para visualização
default_cols = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'RM', 'PTRATIO', 'MEDV']

#multiselect para selecionar as colunas
cols = st.multiselect('Selecione as colunas para visualização', 
                      data.columns.tolist(), default=default_cols)

#exibindo o dataframe com as colunas selecionadas
st.dataframe(data[cols].head(10))

#visualizacao dos dados
st.subheader('Distribuição de imóveis por preço')

#filtrando os dados por faixa de valores
faixa_valores = st.slider('Faixa de preço', float(data.MEDV.min()), 150., (10.0, 100.0))

#filtrando os dados com base na faixa de valores selecionada
dados = data[data['MEDV'].between(faixa_valores[0], faixa_valores[1])]

#plot da distribuição dos dados
f = px.histogram(dados, x='MEDV', nbins=100, title='Distribuição de Preços')
f.update_xaxes(title='MEDV')
f.update_yaxes(title='Total imóveis')
st.plotly_chart(f)

#criacao da aba lateral
st.sidebar.header('Defina os atributos de Preços de Imóveis')

#mapeando dados do usuario para cada atributo da base de dados
CRIM = st.sidebar.number_input('Taxa de criminalidade', value=data.CRIM.mean())
ZN = st.sidebar.number_input('Área de terreno residencial', value=data.ZN.mean())
INDUS = st.sidebar.number_input('Área de indústria', value=data.INDUS.mean())
CHAS = st.sidebar.selectbox('Faz limite com rio sim (1) ou não (0)', [0, 1])
RM = st.sidebar.number_input('Número médio de quartos', value=data.RM.mean())
PTRATIO = st.sidebar.number_input('Relação aluno-professor', value=data.PTRATIO.mean())

#botao de predicao
btn_predict = st.sidebar.button('Prever Preço')

#verificar se o botao foi acionado
if btn_predict:
    result = model.predict([[CRIM, ZN, INDUS, CHAS, RM, PTRATIO]])
    st.subheader('O valor previsto para o imóvel é: ')
    result = 'US $ '+str(round(result[0]*1000,2))
    st.write(result)