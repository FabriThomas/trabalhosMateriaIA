import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder


arquivo = "Inteligência Artificial - Base Gripe (respostas).xlsx"
df = pd.read_excel(arquivo)


df.columns = [
    "carimbo_data_hora",
    "ficou_gripado",
    "vacina_gripe",
    "ambientes_cheios",
    "viajou_100km",
    "alergia_vias_aereas",
    "horas_sono",
    "atividade_fisica",
    "alimentacao_balanceada",
    "lavagem_maos",
    "estresse"
]

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].astype(str).str.strip()

df["viajou_100km"] = df["viajou_100km"].replace("Nuca", "Nunca")

df["estresse"] = df["estresse"].astype(float).astype(int)


def prever_gripe_if_else(viajou_100km, alergia_vias_aereas, estresse, vacina_gripe, lavagem_maos):
    """
    Classificação manual baseada na árvore de decisão gerada anteriormente.
    Retorna 'Sim' ou 'Não' para a pergunta:
    'Você ficou gripado no ano passado?'
    """

    if viajou_100km == "Poucas vezes por ano":
        if estresse == 1:
            return "Não"
        elif estresse == 2:
            return "Sim"
        elif estresse == 4:
            if vacina_gripe == "Não":
                return "Sim"
            else:
                return "Não"
        elif estresse == 5:
            return "Sim"
        else:
            return "Não"

    elif viajou_100km == "Pelo menos uma vez por mês":
        if alergia_vias_aereas == "Muito":
            return "Não"

        elif alergia_vias_aereas == "Não":
            return "Não"

        elif alergia_vias_aereas == "Médio":
            if estresse == 2:
                return "Não"
            elif estresse in [3, 4, 5]:
                return "Sim"
            else:
                return "Não"

        elif alergia_vias_aereas == "Pouco":
            if lavagem_maos == "2 vezes ou menos":
                return "Sim"
            else:
                return "Não"

        else:
            return "Não"

    elif viajou_100km == "Nunca":
        if alergia_vias_aereas == "Muito":
            return "Sim"
        elif alergia_vias_aereas == "Não":
            return "Sim"
        elif alergia_vias_aereas == "Médio":
            return "Não"
        elif alergia_vias_aereas == "Pouco":
            return "Não"
        else:
            return "Não"

    else:
        return "Dados inválidos"



X = df[
    [
        "vacina_gripe",
        "ambientes_cheios",
        "viajou_100km",
        "alergia_vias_aereas",
        "horas_sono",
        "atividade_fisica",
        "alimentacao_balanceada",
        "lavagem_maos",
        "estresse"
    ]
].copy()

y = df["ficou_gripado"].copy()

encoders = {}

for col in X.columns:
    if X[col].dtype == "object":
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le


target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

modelo = DecisionTreeClassifier(random_state=42, max_depth=4)
modelo.fit(X, y_encoded)

regras = export_text(modelo, feature_names=list(X.columns))
print("=== ÁRVORE GERADA PELO SCIKIT-LEARN ===")
print(regras)



def prever_gripe_sklearn(
    vacina_gripe,
    ambientes_cheios,
    viajou_100km,
    alergia_vias_aereas,
    horas_sono,
    atividade_fisica,
    alimentacao_balanceada,
    lavagem_maos,
    estresse
):
    """
    Faz a previsão usando o modelo treinado com scikit-learn.
    Retorna 'Sim' ou 'Não'.
    """

    entrada = {
        "vacina_gripe": vacina_gripe,
        "ambientes_cheios": ambientes_cheios,
        "viajou_100km": viajou_100km,
        "alergia_vias_aereas": alergia_vias_aereas,
        "horas_sono": horas_sono,
        "atividade_fisica": atividade_fisica,
        "alimentacao_balanceada": alimentacao_balanceada,
        "lavagem_maos": lavagem_maos,
        "estresse": estresse
    }

    entrada_df = pd.DataFrame([entrada])

    
    for col in entrada_df.columns:
        if col in encoders:
            entrada_df[col] = encoders[col].transform(entrada_df[col])

    previsao = modelo.predict(entrada_df)[0]
    return target_encoder.inverse_transform([previsao])[0]




print("\n=== TESTE COM AS DUAS ABORDAGENS ===")

resultado_if_else = prever_gripe_if_else(
    viajou_100km="Poucas vezes por ano",
    alergia_vias_aereas="Pouco",
    estresse=4,
    vacina_gripe="Não",
    lavagem_maos="3 a 5 vezes"
)

resultado_sklearn = prever_gripe_sklearn(
    vacina_gripe="Não",
    ambientes_cheios="Sim",
    viajou_100km="Poucas vezes por ano",
    alergia_vias_aereas="Pouco",
    horas_sono="entre 4 e 6 horas",
    atividade_fisica="Sim",
    alimentacao_balanceada="Às vezes",
    lavagem_maos="3 a 5 vezes",
    estresse=4
)

print("Resultado com if/else:", resultado_if_else)
print("Resultado com scikit-learn:", resultado_sklearn)
