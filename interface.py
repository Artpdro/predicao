import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from datetime import time, date
from preditor_ofc import AccidentPredictor  # Importa a classe correta

# Carregar o modelo treinado e seus componentes
@st.cache_resource
def load_model():
    try:
        with open("modelo_acidentes.pkl", "rb") as f:
            data = pickle.load(f)
        
        predictor = AccidentPredictor()
        predictor.modelo = data["modelo"]
        predictor.encoders = data["encoders"]
        predictor.feature_names = data["features"]
        predictor.best_params = data["params"]
        predictor.r2_score = data["r2"]
        predictor.rmse_score = data["rmse"]
        predictor.treinado = True
        return predictor

    except FileNotFoundError:
        st.error("O arquivo do modelo \'modelo_acidentes.pkl\' não foi encontrado. Por favor, treine o modelo primeiro.")
        return None
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None

# Carregar as opções de UF, Município e Condição Climática
@st.cache_data
def load_options():
    # Estas opções não são salvas no modelo, então vamos usar um conjunto padrão ou inferir do datatran_consolidado.json
    # Para simplificar, vamos usar algumas opções comuns. Em um cenário real, estas seriam geradas a partir dos dados de treinamento.
    uf_options = ["MG", "SP", "RJ", "ES", "PR", "SC", "RS"]
    municipios_por_uf = {
        "MG": ["BELO HORIZONTE", "UBERLÂNDIA", "CONTAGEM"],
        "SP": ["SÃO PAULO", "CAMPINAS", "GUARULHOS"],
        "RJ": ["RIO DE JANEIRO", "NITERÓI", "DUQUE DE CAXIAS"],
        "ES": ["VITÓRIA", "VILA VELHA", "SERRA"],
        "PR": ["CURITIBA", "LONDRINA", "MARINGÁ"],
        "SC": ["FLORIANÓPOLIS", "JOINVILLE", "BLUMENAU"],
        "RS": ["PORTO ALEGRE", "CAXIAS DO SUL", "CANOAS"]
    }
    condicoes_metereologicas_options = ["Bom", "Chuva", "Nublado", "Vento", "Nevoeiro/Neblina", "Outro"]
    
    return uf_options, municipios_por_uf, condicoes_metereologicas_options

predictor = load_model()
uf_options, municipios_por_uf, condicoes_metereologicas_options = load_options()

st.title("Preditor de Acidentes de Trânsito 🚗")

if predictor:
    uf = st.selectbox("UF", uf_options)
    municipios_filtrados = municipios_por_uf.get(uf, ["DESCONHECIDO"])
    municipio = st.selectbox("Município", municipios_filtrados)
    horario = st.time_input("Horário", time(0, 0))
    condicao_metereologica = st.selectbox("Condição Climática", condicoes_metereologicas_options)

    if st.button("Prever Acidentes"):
        try:
            # Garantir formato do horário
            horario_str = horario.strftime("%H:%M:%S") if hasattr(horario, "strftime") else str(horario)
            
            # Data para previsão (pode ser a data atual ou uma data futura)
            data_previsao = date.today() # Usar a data atual para a previsão
            data_ddmmyyyy = data_previsao.strftime("%d/%m/%Y")

            # Criar um DataFrame com os novos dados para previsão
            novos_dados_df = pd.DataFrame([
                {
                    "data_inversa": data_ddmmyyyy,
                    "horario": horario_str,
                    "uf": uf,
                    "municipio": municipio,
                    "tipo_acidente": "COLISÃO", # Valor padrão, pois não é uma entrada do usuário
                    "condicao_metereologica": condicao_metereologica
                }
            ])

            # Faz a previsão usando o método 'prever' da classe AccidentPredictor
            previsoes_df = predictor.prever(novos_dados_df)
            
            # A classe AccidentPredictor.prever retorna um DataFrame com 'data' e 'previsoes_acidentes'
            # Pegamos o primeiro (e único) valor de previsão
            predicao = int(previsoes_df["previsoes_acidentes"].iloc[0])

            st.success(f"A previsão de acidentes para as condições informadas é: **{predicao}**")

        except Exception as e_all:
            st.error("Ocorreu um erro ao fazer a predição.")
            st.exception(e_all)

else:
    st.warning("O modelo ainda não foi treinado.")

