import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from datetime import time, date, datetime
from sklearn.preprocessing import LabelEncoder
from preditor_ofc import AdvancedAccidentPredictor

# -------------------------
# Funções utilitárias
# -------------------------
@st.cache_resource
def load_model():
    """
    Carrega o arquivo pickle gerado por AdvancedAccidentPredictor.salvar_modelo()
    Nota: o preditor salva as chaves: "modelo", "encoders", "feature_names", "best_params", "r2_score", "rmse_score".
    """
    try:
        with open("modelo_acidentes.pkl", "rb") as f:
            data = pickle.load(f)

        predictor = AdvancedAccidentPredictor()
        # Ajuste conforme o que foi salvo no preditor (preditor_ofc.py)
        predictor.modelo = data.get("modelo") or data.get("model")  # tenta duas formas por segurança
        predictor.encoders = data.get("encoders", {})
        predictor.feature_names = data.get("feature_names", [])
        predictor.best_params = data.get("best_params", {})
        predictor.r2_score = data.get("r2_score", None)
        predictor.rmse_score = data.get("rmse_score", None)

        # Se o objeto modelo foi carregado, considera o preditor como treinado/salvo
        predictor.treinado = predictor.modelo is not None

        # Garantir que holidays_br está setado (AdvancedAccidentPredictor já inicializa por padrão)
        predictor.holidays_br = getattr(predictor, "holidays_br", None) or {}

        return predictor

    except FileNotFoundError:
        st.error("O arquivo do modelo 'modelo_acidentes.pkl' não foi encontrado. Por favor, treine e salve o modelo primeiro.")
        return None
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None


@st.cache_data
def load_options():
    """Carrega arquivos JSON de opções (UFs, municípios, condições)"""
    try:
        with open("uf_options.json", "r", encoding="utf-8") as f:
            uf_options = json.load(f)
    except Exception:
        uf_options = []

    try:
        with open("municipios_por_uf.json", "r", encoding="utf-8") as f:
            municipios_por_uf = json.load(f)
    except Exception:
        municipios_por_uf = {}

    try:
        with open("condicoes_metereologicas_options.json", "r", encoding="utf-8") as f:
            condicoes_metereologicas_options = json.load(f)
    except Exception:
        condicoes_metereologicas_options = []

    return uf_options, municipios_por_uf, condicoes_metereologicas_options


def build_feature_vector(predictor, data_fixa: date, horario: time, uf: str, municipio: str,
                         condicao_metereologica: str):
    """
    Monta um vetor de features compatível (na medida do possível) com feature_names geradas no treino.
    Preenche valores ausentes com defaults (0) para lags/janelas móveis quando não houver histórico disponível.
    """
    # Data -> pd.Timestamp (mesma forma que o preditor usa)
    dt = pd.to_datetime(data_fixa)

    ano = dt.year
    mes = dt.month
    dia_semana = dt.dayofweek
    dia_ano = dt.dayofyear
    semana_ano = int(dt.isocalendar().week)
    fim_semana = 1 if dia_semana >= 5 else 0

    # horario pode ser datetime.time ou string
    hora_media = None
    if hasattr(horario, "hour"):
        hora_media = horario.hour
    else:
        try:
            parsed = pd.to_datetime(str(horario), format="%H:%M:%S", errors="coerce")
            hora_media = parsed.hour if not pd.isna(parsed) else 0
        except Exception:
            hora_media = 0

    dia_semana_sin = np.sin(2 * np.pi * dia_semana / 7)
    dia_semana_cos = np.cos(2 * np.pi * dia_semana / 7)
    dia_ano_sin = np.sin(2 * np.pi * dia_ano / 365.25)
    dia_ano_cos = np.cos(2 * np.pi * dia_ano / 365.25)

    feriado = int(dt in getattr(predictor, "holidays_br", []))
    feriado_fim_semana = feriado * fim_semana

    # Defaults para lags e janelas móveis (quando não temos histórico no app)
    defaults = {
        "acidentes_lag_1": 0, "acidentes_lag_2": 0, "acidentes_lag_7": 0, "acidentes_lag_14": 0,
        "media_movel_7d": 0, "std_movel_7d": 0, "media_movel_14d": 0, "std_movel_14d": 0,
        "media_movel_28d": 0, "std_movel_28d": 0
    }

    # Categóricas codificadas (usar encoders presentes no predictor)
    encoded = {}
    # Os nomes usados pelo preditor: uf_principal, municipio_principal, condicao_metereologica_principal
    mapping = {
        "uf_principal": uf,
        "municipio_principal": municipio,
        "condicao_metereologica_principal": condicao_metereologica
    }

    for col, val in mapping.items():
        enc = predictor.encoders.get(col)
        encoded_col_name = f"{col}_encoded"
        if enc is None:
            # sem encoder, marca 0
            encoded[encoded_col_name] = 0
        else:
            try:
                encoded_value = int(enc.transform([val])[0])
            except Exception:
                # valor desconhecido — tenta usar a classe mais comum (primeiro valor) ou 0
                try:
                    encoded_value = int(enc.transform([enc.classes_[0]])[0])
                except Exception:
                    encoded_value = 0
            encoded[encoded_col_name] = encoded_value

    # Monta dicionário completo
    row = {
        "ano": ano,
        "mes": mes,
        "dia_semana": dia_semana,
        "dia_ano": dia_ano,
        "semana_ano": semana_ano,
        "fim_semana": fim_semana,
        "dia_semana_sin": dia_semana_sin,
        "dia_semana_cos": dia_semana_cos,
        "dia_ano_sin": dia_ano_sin,
        "dia_ano_cos": dia_ano_cos,
        "hora_media": hora_media,
        "feriado": feriado,
        "feriado_fim_semana": feriado_fim_semana,
        # lags/janelas móveis
        **defaults,
        # encoded cats
        **encoded
    }

    # Se houver feature_names do modelo, ordena e garante todas as colunas necessárias
    if predictor.feature_names:
        X = {}
        for feat in predictor.feature_names:
            X[feat] = row.get(feat, 0)  # default 0 se não foi calculado
        return pd.DataFrame([X])
    else:
        # Sem feature_names: retorna DataFrame a partir do row (pode não casar com o modelo)
        return pd.DataFrame([row])


# -------------------------
# Inicialização
# -------------------------
predictor = load_model()
uf_options, municipios_por_uf, condicoes_metereologicas_options = load_options()

st.title("Preditor de Acidentes de Trânsito")

if predictor is None or not getattr(predictor, "treinado", False):
    st.warning("Modelo não encontrado ou não carregado corretamente. Treine e salve o modelo antes de usar a interface.")
else:
    # Inputs do usuário (removido campo de tipo de acidente conforme solicitado)
    uf = st.selectbox("UF", uf_options)
    municipios_filtrados = municipios_por_uf.get(uf, ["DESCONHECIDO"])
    municipio = st.selectbox("Município", municipios_filtrados)
    horario = st.time_input("Horário", time(0, 0))
    condicao_metereologica = st.selectbox("Condição Climática", condicoes_metereologicas_options)

    if st.button("Prever Acidentes"):
        try:
            # usa data fixa para evitar pegar horário atual — conforme sua instrução
            data_fixa = date(2020, 1, 1)
            # monta features (não inclui tipo de acidente)
            X = build_feature_vector(
                predictor=predictor,
                data_fixa=data_fixa,
                horario=horario,
                uf=uf,
                municipio=municipio,
                condicao_metereologica=condicao_metereologica
            )

            # Verifica compatibilidade com o modelo
            if predictor.modelo is None:
                st.error("Modelo válido não foi carregado.")
            else:
                try:
                    pred_raw = predictor.modelo.predict(X)
                    # se for array, pega primeiro elemento
                    if hasattr(pred_raw, "__len__"):
                        pred_val = float(pred_raw[0])
                    else:
                        pred_val = float(pred_raw)

                    pred_final = int(np.round(pred_val).clip(0))
                    st.success(f"A previsão de acidentes para as condições informadas é: **{pred_final}**")
                    if predictor.r2_score is not None and predictor.rmse_score is not None:
                        st.info(f"(R² treino: {predictor.r2_score:.3f} | RMSE treino: {predictor.rmse_score:.2f})")
                except Exception as e_pred:
                    st.error("Erro ao executar a predição com o modelo carregado.")
                    st.exception(e_pred)

        except Exception as e_all:
            st.error("Ocorreu um erro ao fazer a predição.")
            st.exception(e_all)
