import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Título do aplicativo
st.title('Efeitos da Microgravidade nas Variáveis de Saúde')

# Texto introdutório
st.write('Ao entrar em µg, a ausência do vetor de gravidade diminui a pressão hidrostática e os fluidos corporais são redistribuídos para a parte superior do corpo e cabeça (deslocamento do fluido cefálico induzido por µg) [12]. Além disso, µg reduz drasticamente os níveis de atividade física no espaço. O deslocamento ascendente de fluido resulta em um aumento do volume vascular e do volume sistólico (VS) que distende a vasculatura central, acionando os receptores centrais da carótida, aórtica e cardíaco que permitem mecanismos para reduzir a sobrecarga de fluido percebida [36]. A distensão do coração aumenta a liberação do peptídeo natriurético atrial (ANP) e estimula os barorreceptores nas artérias carótidas e aórticas que, por sua vez, inibem o sistema renina-angiotensina-aldosterona [37]. Juntas, essas respostas resultam em uma redução de 10 a 15% no volume plasmático sanguíneo [2]. O ANP também induz vasodilatação e, portanto, a exposição de curto prazo em µg (até 10 dias) causa vasodilatação e uma alteração aguda na permeabilidade vascular que contribui para a diminuição do volume plasmático e ajuda a diminuir as pressões atriais [38]. Outros mecanismos que visam diminuir a sobrecarga hídrica percebida são o aumento da diurese ou natriurese. A diurese aparente não é observada durante o voo espacial, e acredita-se que as reduções no volume do plasma sanguíneo não sejam o resultado do aumento da diurese e da natriurese, mas resultem da mudança transitória de fluido dos compartimentos intravasculares para os espaços intracelulares [39]. Isto resulta da redução das pressões intersticiais e do aumento das pressões vasculares na parte superior do corpo, que são caracterizadas por sintomas típicos como rostos “inchados”, narizes “entupidos” e “pernas de frango” [2]. Uma adaptação crônica a µg é o déficit contínuo no volume sanguíneo efetivo [38,40].')

# Entrada de Microgravidade
microgravidade = st.slider('Microgravidade (0-1)', 0.0, 1.0, 0.5)

# Definir as variáveis e seus valores de referência
variaveis = {
    'Blood Volume': 5.0,
    'Haematocrit': 40,
    'CO (Cardiac Output)': 5.0,
    'SV (Stroke Volume)': 70,
    'Ventricular Size': 100,
    'CVP (Central Venous Pressure)': 8,
    'MAP (Mean Arterial Pressure)': 90,
    'SBP (Systolic Blood Pressure)': 120,
    'DBP (Diastolic Blood Pressure)': 80,
    'SVR (Systemic Vascular Resistance)': 1200,
    'HR (Heart Rate)': 70,
    'cIMT (Carotid Intima-Media Thickness)': 0.7,
    'Femoral IMT': 0.6,
    'Arterial Stiffness': 5.0
}

# Calcular os efeitos da microgravidade nas variáveis
variaveis_afetadas = {var: valor * (1 - microgravidade) for var, valor in variaveis.items()}

# Mostrar os resultados
st.subheader('Efeitos da Microgravidade nas Variáveis de Saúde')
df_variaveis = pd.DataFrame.from_dict(variaveis_afetadas, orient='index', columns=['Valor Afetado'])
st.write(df_variaveis)

# Gráfico das Variáveis Afetadas
st.subheader('Gráfico das Variáveis Afetadas')
fig, ax = plt.subplots(figsize=(8, 6))
df_variaveis.plot(kind='bar', ax=ax)
ax.set_ylabel('Valor Afetado')
plt.xticks(rotation=45)
st.pyplot(fig)

# Modelo de Regressão Simples (apenas para ilustração)
# Isso é fictício e não representa uma relação real
regression_model = LinearRegression()
X = np.array(list(variaveis.values())).reshape(-1, 1)
y = np.array(list(variaveis_afetadas.values()))
regression_model.fit(X, y)

# Previsão com base na microgravidade
microgravidade_input = np.array([1 - microgravidade]).reshape(-1, 1)
previsao_variavel = regression_model.predict(microgravidade_input)
st.subheader('Previsão da Variável Afetada pela Microgravidade')
st.write(f'A previsão da variável afetada pela microgravidade é aproximadamente {previsao_variavel[0]:.2f}.')

# Imagem relacionada
image = "biomed.jpg"
st.image(image, caption='Um resumo das doenças cardiovasculares induzidas por voos espaciais. Abreviaturas: débito cardíaco (DC); volume sistólico (VS); pressão arterial média (PAM); pressão arterial sistólica (PAS); pressão arterial diastólica (PAD); intolerância ortostática (IO); aumento (↑); diminuir (↓).', use_column_width=True)

# Rodapé e Referências
st.write("Feito com Streamlit")
st.write("Referências")
st.write("https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8773383/")
st.write("https://www.nature.com/articles/s41591-021-01637-7")