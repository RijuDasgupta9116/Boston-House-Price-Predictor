import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

pipe = pickle.load(open('pipe.pkl','rb'))

st.title('House Price Predictor Web App')

CRIM = st.number_input('Per capita crime rate by town')
ZN = st.number_input('proportion of residential land zoned for lots over 25,000 sq.ftp')
INDUS = st.number_input('proportion of non-retail business acres per town')
CHAS = st.number_input('Charles River dummy variable')
NOX = st.number_input('Nitric oxides concentration')
RM = st.number_input('Average number of rooms per dwelling')
AGE = st.number_input('Per its built prior to 1940')
DIS = st.number_input('Weighted distances to five Boston employment centres')
RAD = st.number_input('Index of accessibility to radial highways')
TAX = st.number_input('Full-value property-tax rate per $10,000')
PTRATIO = st.number_input('Pupil-teacher ratio by town')
B = st.number_input('proportion of blacks by town')
LSTAT = st.number_input('% lower status of the population')

if st.button('Predict Price'):
    input = np.array([[CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT]])
    input = pd.DataFrame(input,columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'])
    y_pred = pipe.predict(input)
    # st.title("Rs " + str(np.round(y_pred[0])) + '[/-]')
    st.title("Rs. "+str(np.round(y_pred[0]))+"/-")