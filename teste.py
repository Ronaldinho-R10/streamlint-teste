import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import streamlit as st
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# título
st.write("Prevendo Diabetes")

df = pd.read_csv("diabetes.csv")

# Mapeia nomes das colunas em português para nomes em inglês
column_mapping = {
    'Gravidez': 'Pregnancies',
    'Glicose': 'Glucose',
    'Pressão Sanguínea': 'BloodPressure',
    'Espessura da pele': 'SkinThickness',
    'Insulina': 'Insulin',
    'Índice de massa corporal': 'BMI',
    'Histórico familiar de diabetes': 'DiabetesPedigreeFunction',
    'Idade': 'Age'
}

# Renomeia as colunas do DataFrame de acordo com o mapeamento
df.rename(columns=column_mapping, inplace=True)

# cabeçalho
st.subheader("Informações dos dados")

# nomedousuário
user_input = st.sidebar.text_input("Digite seu nome")
st.write("Paciente:", user_input)

# dados de entrada
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# dados dos usuários com a função
def get_user_data():
    pregnancies = st.sidebar.slider("Pregnancies", 0, 15, 1)
    glucose = st.sidebar.slider("Glucose", 0, 200, 110)
    blood_pressure = st.sidebar.slider("BloodPressure", 0, 122, 72)
    skin_thickness = st.sidebar.slider("SkinThickness", 0, 99, 20)
    insulin = st.sidebar.slider("Insulin", 0, 900, 30)
    bmi = st.sidebar.slider("BMI", 0.0, 70.0, 15.0)
    diabetes_pedigree_function = st.sidebar.slider("DiabetesPedigreeFunction", 0.0, 3.0, 0.0)
    age = st.sidebar.slider("Age", 15, 100, 21)

    user_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': diabetes_pedigree_function,
        'Age': age
    }

    user_features = pd.DataFrame(user_data, index=[0])
    return user_features

user_input_variables = get_user_data()

# Treinamento do modelo
dtc = DecisionTreeClassifier(criterion='entropy', max_depth=3)
dtc.fit(x_train, y_train)

# Acurácia do modelo
st.subheader('Acurácia do modelo')
accuracy = accuracy_score(y_test, dtc.predict(x_test)) * 100
st.write(accuracy)

# Previsão do resultado
prediction = dtc.predict(user_input_variables)
st.subheader('Previsão:')
st.write(prediction)

# Gráfico de barras dos dados do usuário
st.subheader('Dados do Usuário')
st.bar_chart(user_input_variables)

# Gráfico de barras dos dados do usuário
st.subheader('Dados do Usuário')
st.bar_chart(user_input_variables)