import pandas as pd
import numpy as np
import json
import pickle
import lightgbm as lgb
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error
import holidays
import warnings
import itertools

warnings.filterwarnings("ignore")


class AccidentPredictor:
    def __init__(self):
        self.modelo = lgb.LGBMRegressor(random_state=42)
        self.encoders = {}
        self.feature_names = []
        self.treinado = False
        self.best_params = {}
        self.r2_score = None
        self.rmse_score = None
        self.holidays_br = holidays.Brazil()

    def _simplificar_clima(self, cond):
        if any(k in cond for k in ["Chuva", "Garoa"]):
            return "Chuva"
        if "Nublado" in cond:
            return "Nublado"
        if any(k in cond for k in ["Céu Claro", "Sol"]):
            return "Bom"
        if "Vento" in cond:
            return "Vento"
        if any(k in cond for k in ["Nevoeiro", "Neblina"]):
            return "Nevoeiro/Neblina"
        return "Outro"

    def _processar_dados(self, df):
        df["data"] = pd.to_datetime(df["data_inversa"], format="%d/%m/%Y", errors="coerce")
        df = df[df["data"].dt.year >= 2019].dropna(subset=["data", "horario", "uf", "municipio", "tipo_acidente", "condicao_metereologica"])
        df["hora"] = pd.to_datetime(df["horario"], format="%H:%M:%S", errors="coerce").dt.hour
        df.dropna(subset=["hora"], inplace=True)
        df["condicao_metereologica"] = df["condicao_metereologica"].apply(self._simplificar_clima)

        agg = df.groupby("data").agg(
            acidentes=("data_inversa", "count"),
            uf=("uf", lambda x: x.mode()[0]),
            municipio=("municipio", lambda x: x.mode()[0]),
            tipo_acidente=("tipo_acidente", lambda x: x.mode()[0]),
            condicao_metereologica=("condicao_metereologica", lambda x: x.mode()[0]), # Manter o nome original
            hora_media=("hora", "mean")
        ).reset_index()
        # Aplicar _simplificar_clima após a agregação para a coluna 'clima'
        agg["clima"] = agg["condicao_metereologica"].apply(self._simplificar_clima)
        agg = agg.drop(columns=["condicao_metereologica"]) # Remover a coluna original após simplificação

        return agg.sort_values("data").reset_index(drop=True)

    def _criar_features(self, df):
        df["ano"] = df["data"].dt.year
        df["mes"] = df["data"].dt.month
        df["dia_semana"] = df["data"].dt.dayofweek
        df["dia_ano"] = df["data"].dt.dayofyear
        df["semana"] = df["data"].dt.isocalendar().week.astype(int)
        df["fim_semana"] = (df["dia_semana"] >= 5).astype(int)
        df["feriado"] = df["data"].apply(lambda x: int(x in self.holidays_br))
        df["feriado_fim_semana"] = df["feriado"] * df["fim_semana"]

        df["dia_semana_sin"] = np.sin(2 * np.pi * df["dia_semana"] / 7)
        df["dia_semana_cos"] = np.cos(2 * np.pi * df["dia_semana"] / 7)
        df["dia_ano_sin"] = np.sin(2 * np.pi * df["dia_ano"] / 365.25)
        df["dia_ano_cos"] = np.cos(2 * np.pi * df["dia_ano"] / 365.25)

        # Lags e médias móveis só fazem sentido se houver a coluna 'acidentes'
        # Durante a previsão, df_processado terá 'acidentes' = 0, então estas features serão 0
        if 'acidentes' in df.columns:
            for lag in [1, 2, 7, 14]:
                df[f"lag_{lag}"] = df["acidentes"].shift(lag)
            for w in [7, 14, 28]:
                df[f"media_{w}"] = df["acidentes"].shift(1).rolling(w, min_periods=1).mean()
                df[f"std_{w}"] = df["acidentes"].shift(1).rolling(w, min_periods=1).std()
        else:
            for lag in [1, 2, 7, 14]:
                df[f"lag_{lag}"] = 0
            for w in [7, 14, 28]:
                df[f"media_{w}"] = 0
                df[f"std_{w}"] = 0

        df.fillna(0, inplace=True)

        for col in ["uf", "municipio", "tipo_acidente", "clima"]:
            if col in df.columns:
                if col in self.encoders:
                    enc = self.encoders[col]
                    df.loc[:, f"{col}_enc"] = df[col].apply(lambda x: enc.transform([x])[0] if x in enc.classes_ else -1)
                else:
                    # Isso só deve acontecer durante o treinamento inicial
                    enc = LabelEncoder()
                    df.loc[:, f"{col}_enc"] = enc.fit_transform(df[col])
                    self.encoders[col] = enc
            else:
                # Se a coluna não estiver presente no df, mas o encoder existir (modo de previsão),
                # preencher com um valor padrão (e.g., -1 para 'desconhecido')
                if col in self.encoders:
                    df.loc[:, f"{col}_enc"] = -1
                else:
                    # Se a coluna não existe e não há encoder, criar com 0 (caso de treinamento com dados incompletos)
                    df.loc[:, f"{col}_enc"] = 0

        features = [
            "ano", "mes", "dia_semana", "dia_ano", "semana", "fim_semana",
            "dia_semana_sin", "dia_semana_cos", "dia_ano_sin", "dia_ano_cos",
            "hora_media", "feriado", "feriado_fim_semana"
        ] + [f"lag_{i}" for i in [1, 2, 7, 14]] + \
            [f"media_{i}" for i in [7, 14, 28]] + \
            [f"std_{i}" for i in [7, 14, 28]] + \
            [f"{c}_enc" for c in ["uf", "municipio", "tipo_acidente", "clima"]]

        y = df["acidentes"] if "acidentes" in df.columns else None
        return df[features], y

    def _otimizar_parametros(self, X, y, grid):
        tscv = TimeSeriesSplit(n_splits=5)
        best_rmse, best = np.inf, None

        for combo in itertools.product(*grid.values()):
            params = dict(zip(grid.keys(), combo))
            rmses = []
            for tr, val in tscv.split(X):
                X_tr, X_val = X.iloc[tr], X.iloc[val]
                y_tr, y_val = y.iloc[tr], y.iloc[val]

                m = lgb.LGBMRegressor(**params, random_state=42, verbosity=-1)
                m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
                y_pred = np.clip(np.round(m.predict(X_val)), 0, None)
                rmses.append(np.sqrt(mean_squared_error(y_val, y_pred)))

            mean_rmse = np.mean(rmses)
            if mean_rmse < best_rmse:
                best_rmse, best = mean_rmse, params

        return best

    def treinar(self, arquivo_json):
        with open(arquivo_json, "r", encoding="utf-8") as f:
            df = pd.DataFrame(json.load(f))

        df = self._processar_dados(df)
        X, y = self._criar_features(df)

        if X.empty:
            raise ValueError("Erro: DataFrame vazio após processamento.")

        self.feature_names = X.columns.tolist()
        grid = {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "num_leaves": [20, 31],
            "max_depth": [-1, 10]
        }

        self.best_params = self._otimizar_parametros(X, y, grid)
        self.modelo = lgb.LGBMRegressor(**self.best_params, random_state=42)
        self.modelo.fit(X, y)

        y_pred = np.clip(np.round(self.modelo.predict(X)), 0, None)
        self.r2_score = r2_score(y, y_pred)
        self.rmse_score = np.sqrt(mean_squared_error(y, y_pred))
        self.treinado = True

        print(f"Treinamento concluído | R²: {self.r2_score:.4f} | RMSE: {self.rmse_score:.2f}")

    def prever(self, df_novos_dados):
        if not self.treinado:
            raise RuntimeError("Treine o modelo antes de fazer previsões.")

        df_processado = self._processar_dados(df_novos_dados.copy())
        
        # Para a previsão, precisamos garantir que as colunas categóricas originais
        # estejam presentes no df_processado antes de _criar_features ser chamado.
        # O _processar_dados já retorna as colunas agregadas (uf, municipio, etc.).
        # No entanto, para a codificação, _criar_features precisa delas.
        # O problema anterior era que _criar_features tentava acessar df[col] onde col era 'uf' etc.
        # mas df_processado já tinha essas colunas agregadas.
        # A solução é garantir que _criar_features possa lidar com a ausência de 'acidentes'
        # e que os encoders sejam aplicados corretamente.

        X_prever, _ = self._criar_features(df_processado)

        # Garantir que as colunas de previsão correspondam às colunas de treinamento
        missing_cols = set(self.feature_names) - set(X_prever.columns)
        for c in missing_cols:
            X_prever[c] = 0
        X_prever = X_prever[self.feature_names]

        previsoes = np.clip(np.round(self.modelo.predict(X_prever)), 0, None)
        df_processado["previsoes_acidentes"] = previsoes
        return df_processado[["data", "previsoes_acidentes"]]

    def salvar_modelo(self, nome="modelo_acidentes.pkl"):
        if not self.treinado:
            raise RuntimeError("Treine o modelo antes de salvar.")

        with open(nome, "wb") as f:
            pickle.dump({
                "modelo": self.modelo,
                "encoders": self.encoders,
                "features": self.feature_names,
                "params": self.best_params,
                "r2": self.r2_score,
                "rmse": self.rmse_score
            }, f)
        print(f"Modelo salvo: {nome}")


if __name__ == "__main__":
    predictor = AccidentPredictor()
    predictor.treinar("datatran_consolidado.json")
    predictor.salvar_modelo()