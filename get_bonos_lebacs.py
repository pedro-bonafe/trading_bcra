import requests
import pandas as pd
import certifi

# Endpoints
URL_BONDS = "https://open.bymadata.com.ar/vanoms-be-core/rest/api/bymadata/free/public-bonds"
URL_LEBACS = "https://open.bymadata.com.ar/vanoms-be-core/rest/api/bymadata/free/lebacs"

# Headers
headers = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
    "Content-Type": "application/json"
}

# Columnas de referencia
BASE_COLS = [
    "symbol", "trade", "bidPrice", "offerPrice",
    "volume", "tradeVolume", "maturityDate",
    "denominationCcy", "market", "tradeHour"
]

def normalize_dataframe(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Asegura que el DataFrame tenga todas las columnas esperadas."""
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols].copy()

def fetch_bonds(T0: bool, T1: bool) -> pd.DataFrame:
    payload = {
        "T0": T0,
        "T1": T1,
        "Content-Type": "application/json, text/plain"
    }
    resp = requests.post(URL_BONDS, headers=headers, json=payload, verify=False)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data["data"])
    df = normalize_dataframe(df, BASE_COLS)
    df["settlement"] = "T0" if T0 else "T1"
    df["type"] = "bond"
    return df

def fetch_lebacs() -> pd.DataFrame:
    payload = {
        "T0": True,
        "T1": True,
        "Content-Type": "application/json, text/plain"
    }
    resp = requests.post(URL_LEBACS, headers=headers, json=payload, verify=False)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data["data"])
    df = normalize_dataframe(df, BASE_COLS)
    df["settlement"] = "T0/T1"
    df["type"] = "lebac"
    return df

def main():
    # Bonos
    df_T1 = fetch_bonds(T0=False, T1=True)
    df_T0 = fetch_bonds(T0=True, T1=False)
    df_bonds = pd.concat([df_T1, df_T0], ignore_index=True)

    # LEBACs
    df_lebacs = fetch_lebacs()

    # Unir todo
    df_all = pd.concat([df_bonds, df_lebacs], ignore_index=True)

    # Guardar en CSV
    df_all.to_csv("public_bonds_lebacs.csv", index=False, encoding="utf-8")

    print("✅ Datos guardados en public_bonds_lebacs.csv")
    print(df_all.head())

if __name__ == "__main__":
    main()
