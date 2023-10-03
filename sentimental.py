import streamlit as st
from transformers import pipeline

# Carregando o modelo pré-treinado para análise de sentimentos
sentiment_classifier = pipeline("sentiment-analysis")

# Título do aplicativo
st.title('Análise de Sentimentos com Streamlit')

# Caixa de texto para inserir a frase
sentence = st.text_input('Insira uma frase para análise de sentimento:')

# Botão para analisar a frase
if st.button('Analisar Sentimento'):
    if sentence:
        # Realiza a análise de sentimento
        result = sentiment_classifier(sentence)

        # Exibe o resultado
        sentiment = result[0]['label']
        score = result[0]['score']

        st.write(f'Sentimento: {sentiment}')
        st.write(f'Confiança: {score:.2f}')
    else:
        st.warning('Por favor, insira uma frase para análise de sentimento.')