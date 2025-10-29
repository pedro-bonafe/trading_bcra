import requests
import pandas as pd
import json


# Endpoint
url_ficha = "https://open.bymadata.com.ar/vanoms-be-core/rest/api/bymadata/free/bnown/fichatecnica/especies/general"
#df = pd.read_csv("public_bonds_lebacs.csv")
df = pd.read_csv("public_lebacs.csv")
# Supongamos que ya tenés df con la columna "symbol"
symbols = df["symbol"].unique().tolist()

# Lista para acumular resultados
fichas = []

for sym in symbols:
    payload = {"symbol": sym, "Content-Type": "application/json"}
    try:
        resp = requests.post(url_ficha, json=payload, verify=False)
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            if data:
                ficha = data[0]
                ficha["symbol"] = sym   # 👈 inyectamos el símbolo
                fichas.append(ficha)
        else:
            print(f"Error {resp.status_code} para {sym}")
    except Exception as e:
        print(f"Excepción para {sym}: {e}")

# Pasamos a DataFrame
df_ficha = pd.DataFrame(fichas)

# Guardar en CSV para trabajar después


print("Fichas técnicas descargadas:", df_ficha.shape)
print(df_ficha[["symbol", "denominacion", "fechaVencimiento"]].head())

df_ficha.to_csv("fichas_tecnicas_lebacs.csv", index=False, sep=";")

file_name = "ficha_lebacs.json"

try:
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(fichas, f, indent=4, ensure_ascii=False)
    print(f"Datos guardados exitosamente en '{file_name}'")
except IOError as e:
    print(f"Error al escribir en el archivo: {e}")