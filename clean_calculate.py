import pandas as pd
import numpy as np
import json 
import matplotlib as mlp
import re 

try:
    with open('ficha_lebacs.json', 'r', encoding='utf-8') as archivo:
        fichas_tecnicas_lebacs = json.load(archivo)
        
    print("Archivo .json leído exitosamente.")
    
    # 'datos' ahora es un diccionario o lista de Python
    print(type(fichas_tecnicas_lebacs)) 
    print(fichas_tecnicas_lebacs)
    
except FileNotFoundError:
    print("Error: El archivo 'ficha_lebacs.json' no se encontró.")
except json.JSONDecodeError:
    print("Error: El archivo contiene JSON no válido.")
except Exception as e:
    print(f"Ocurrió un error: {e}")


# ASUMIMOS QUE df YA ESTÁ CARGADO con los datos originales.
# df = pd.DataFrame(datos_originales) 
df = pd.DataFrame(fichas_tecnicas_lebacs)

# ==============================================================================
# 0. MAPEO DE CLAVES Y CONFIGURACIÓN
# ==============================================================================

# Claves a normalizar genéricamente
claves_texto_limpio = ['paisLey', 'tipoEspecie', 'emisor', 'ley', 'moneda', 'tipoObligacion', 'tipoGarantia', 'insType']
clave_forma_amortizacion = 'formaAmortizacion'
clave_interes = 'interes' # Clave usada en el paso 3

# Lista de acentos/tildes a remover
REPLACEMENTS = {
    'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
    'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U'
}
def remove_accents(text):
    """Función auxiliar para quitar tildes."""
    if pd.isna(text): return text
    text = str(text) 
    for acc, norm in REPLACEMENTS.items():
        text = text.replace(acc, norm)
    return text

print(f"Total de registros a procesar: {len(df)}")

# ==============================================================================
# 1. LIMPIEZA UNIVERSAL DE CARACTERES ESPECIALES EN COLUMNAS OBJECT
# ==============================================================================

print("\n1. Aplicando limpieza universal de caracteres especiales (manteniendo . y ,)...")

# Identificar columnas de tipo 'object' (texto)
cols_to_clean = df.select_dtypes(include=['object']).columns

for col in cols_to_clean:
    df[col] = df[col].astype(str).str.strip().apply(remove_accents).str.lower()
    
    # Limpieza de caracteres universales (saltos de línea, comillas, etc.)
    df[col] = (
        df[col]
        .str.replace(r'\s+', ' ', regex=True)  # Unificar espacios
        # Se mantiene el punto y la coma
        .str.replace(r'[;:`´"\-()\[\]¿?¡!*#$@]', '', regex=True)
        .str.strip()
    )
    
    # Reemplazar valores vacíos o 'nan' resultantes por NA de Pandas
    df[col] = df[col].replace(['nan', 'none', ''], pd.NA)

print("Limpieza universal de texto completada.")

# ==============================================================================
# 2. NORMALIZACIÓN DE CLAVES DE TEXTO GENERALES Y AGRUPACIÓN
# ==============================================================================

print("\n2. Normalizando claves de texto generales (paisLey, emisor, etc.) y formaAmortizacion...")

# Normalización Genérica (basada en el texto limpio del Paso 1)
for col in claves_texto_limpio:
    if col in df.columns:
        # La limpieza profunda ya fue hecha en el Paso 1, aquí solo se copia y renombra.
        df[f'{col}_norm'] = df[col] 
        df[f'{col}_norm'] = df[f'{col}_norm'].replace(['nan', 'none', ''], pd.NA)

# Normalización avanzada: formaAmortizacion (Agrupación)
if clave_forma_amortizacion in df.columns:
    col = clave_forma_amortizacion
    
    # Función de Agrupación Lógica
    def agrupar_amortizacion(texto_limpio):
        if pd.isna(texto_limpio):
            return pd.NA
        if 'cuotas trimestrales' in texto_limpio or 'cuotas iguales y consecutivas' in texto_limpio:
            return 'Amortizacion en Cuotas'
        elif 'tipo de cambio aplicable' in texto_limpio or 'moneda de pago pesos' in texto_limpio:
            return 'Integra al Vencimiento (TC Aplicable)'
        elif 'integra al vencimiento' in texto_limpio or 'integramente al vencimiento' in texto_limpio:
            return 'Integra al Vencimiento (Simple/Unica)'
        else:
            return 'Otras/Pendiente de Revision'

    # Se aplica sobre la columna original, que ya fue limpiada en el Paso 1.
    df['formaAmortizacion_agrupada'] = df[clave_forma_amortizacion].apply(agrupar_amortizacion)

print("Normalización genérica y 'formaAmortizacion' completada.")

# ==============================================================================
# 3. NORMALIZACIÓN AVANZADA: interes (Extracción y TNA/Frecuencia) - CORREGIDO
# ==============================================================================

print("\n3. Normalización de interes (Extracción y TNA/Frecuencia) - CORREGIDO (BADLAR/TAMAR Priorizados)...")
clave_interes = 'interes'

if clave_interes in df.columns:
    
    # FUNCIÓN AUXILIAR DE LIMPIEZA Y CONVERSIÓN DE DECIMALES
    def limpiar_y_convertir_valor(valor_str):
        if pd.isna(valor_str) or not isinstance(valor_str, str):
            return np.nan
        
        valor_str = valor_str.replace(' ', '')
        
        # Caso 1: Formato Latino (X,XXX.YY o X.XXX,YY) -> Convertir todo a '.'
        if ',' in valor_str and '.' in valor_str:
             if valor_str.index('.') < valor_str.index(','):
                 valor_str = valor_str.replace('.', '').replace(',', '.')
             else:
                 valor_str = valor_str.replace(',', '')

        # Caso 2: Formato Latino puro (X,YY) -> Convertir a '.'
        elif ',' in valor_str:
            valor_str = valor_str.replace(',', '.')

        # Eliminar cualquier punto extra que no sea el decimal final (separadores de miles)
        valor_str = re.sub(r'(?<=\d)\.(?=\d{3}(?:\.|$))', '', valor_str)
        
        try:
            return float(valor_str)
        except ValueError:
            return np.nan

    # CREACIÓN DE LA COLUMNA 'interes_limpio'
    df['interes_limpio'] = df[clave_interes].astype(str)
    df['interes_limpio'] = (
        df['interes_limpio']
        .str.replace('\n', ' ', regex=False)
        .str.replace('\r', ' ', regex=False)
        .str.replace(r'\s+', ' ', regex=True)
        .str.replace('¿', '', regex=False)
        .apply(remove_accents)
        .str.lower()
        .str.strip()
        # Estandarización de frases que complican la extracción:
        .str.replace('devengaran intereses a una ', 'tasa ', regex=False)
        .str.replace('tasa de interes', 'tasa', regex=False)
        .str.replace('los intereses seran pagados al vencimiento', 'pagaderos al vencimiento', regex=False)
    )

    # Definiciones de patrones clave
    patron_frecuencia = r'(capitalizable|capitalizacion|mensual|trimestral|semestral|anual|diaria)'
    patron_valor_interes = r'(?:\s|^)([0-9.,]+)\s*%' 
    patron_tasa_variable = r'(tamar|badlar|tem)'
    patron_cupon_cero = r'(descuento|cupón cero|cero cupón|no devengarán|no devengaran)'
    
    # 💥 PATRONES PARA TASA FIJA (CORREGIDOS)
    patron_tasa_tipo_complejo = r'(nominal|efectiva)\s+(anual|mensual)' # e.g. "nominal anual"
    patron_tasa_tipo_simple = r'(tna|tea|tnm|tem)' # Acrónimos
    
    # Función para clasificación, extracción, referencia y frecuencia
    def clasificar_interes(texto):
        tipo = 'OTRO'
        referencia = pd.NA
        valor = pd.NA
        frecuencia = pd.NA
        
        if pd.isna(texto):
             return pd.NA, pd.NA, pd.NA, pd.NA

        # 1. Extracción de Frecuencia de Capitalización 
        match_frecuencia = re.search(patron_frecuencia, texto)
        if match_frecuencia:
            freq_str = match_frecuencia.group(1)
            if 'diaria' in freq_str: frecuencia = 'DIARIA'
            elif 'mensual' in freq_str: frecuencia = 'MENSUAL'
            elif 'trimestral' in freq_str: frecuencia = 'TRIMESTRAL'
            elif 'semestral' in freq_str: frecuencia = 'SEMESTRAL'
            elif 'anual' in freq_str: frecuencia = 'ANUAL'
            elif 'capitaliza' in freq_str: frecuencia = 'OTRA CAPITALIZACION'


        # 2. Extracción del Valor Numérico 
        match_valor = re.search(patron_valor_interes, texto)
        if match_valor:
            valor_str = match_valor.group(1)
            valor = limpiar_y_convertir_valor(valor_str)
            
        # 3. Clasificación del Tipo y Referencia (Prioridad: CERO, VARIABLE, FIJA)
        
        # A. CUPÓN CERO
        if re.search(patron_cupon_cero, texto):
            tipo = 'CUPON_CERO'
        
        # B. TASA VARIABLE (Captura BADLAR/TAMAR)
        elif re.search(patron_tasa_variable, texto):
            tipo = 'VARIABLE'
            if 'tamar' in texto: referencia = 'TAMAR'
            elif 'badlar' in texto: referencia = 'BADLAR'
            elif 'tem' in texto: referencia = 'TEM' 
            
            # Revisa si es margen 
            if 'margen' in texto: tipo = 'VARIABLE_MARGEN'
        
        # C. TASA FIJA / OTRAS REFERENCIAS (Lógica corregida)
        elif valor is not np.nan:
            tipo = 'FIJA'
            
            # 💥 1. Búsqueda de la estructura "Nominal/Efectiva + Anual/Mensual"
            match_complejo = re.search(patron_tasa_tipo_complejo, texto)
            
            if match_complejo:
                tipo_tasa = match_complejo.group(1)
                periodo = match_complejo.group(2)
                
                if tipo_tasa == 'nominal' and periodo == 'anual':
                    referencia = 'TNA'
                elif tipo_tasa == 'nominal' and periodo == 'mensual':
                    referencia = 'TNM'
                elif tipo_tasa == 'efectiva' and periodo == 'anual':
                    referencia = 'TEA'
                elif tipo_tasa == 'efectiva' and periodo == 'mensual':
                    referencia = 'TEM' # Nota: Puede ser TEM, pero también puede referirse a Tasa Efectiva Mensual
            
            # 💥 2. Búsqueda de acrónimos como FALLBACK (Si el complejo no encontró nada)
            elif referencia is pd.NA:
                match_simple = re.search(patron_tasa_tipo_simple, texto)
                if match_simple:
                    acronimo = match_simple.group(1)
                    if 'tna' in acronimo: referencia = 'TNA'
                    elif 'tea' in acronimo: referencia = 'TEA'
                    elif 'tnm' in acronimo: referencia = 'TNM'
                    elif 'tem' in acronimo: referencia = 'TEM'
        
        return tipo, referencia, valor, frecuencia

    # Aplicar la función de clasificación/extracción
    df[['interes_tipo', 'interes_referencia', 'interes_valor_num', 'interes_capitalizacion_freq']] = df['interes_limpio'].apply(
        lambda x: pd.Series(clasificar_interes(x))
    )
    
    # Opcional: Eliminar la columna intermedia de limpieza
    df = df.drop(columns=['interes_limpio'])

print("Extracción de interes (BADLAR/TAMAR priorizados, TNA/TNM/TEA/TEM refinados) completada.")

# ==============================================================================
# 4. NORMALIZACIÓN DE CLAVES DE FECHA Y NUMÉRICAS
# ==============================================================================

print("\n4. Normalizando claves de fecha y numéricas...")

claves_fecha_norm = ['fechaEmision', 'fechaVencimiento', 'fechaDevenganIntereses']
claves_numericas_norm = ['montoNominal', 'montoResidual', 'denominacionMinima']

# Normalizar Fechas
for col in claves_fecha_norm:
    if col in df.columns:
        df[f'{col}_dt'] = pd.to_datetime(df[col], errors='coerce', dayfirst=False)
        df[f'{col}_dt'] = df[f'{col}_dt'].apply(lambda x: pd.NaT if pd.isna(x) else x)
        
# Normalizar Numéricos
for col in claves_numericas_norm:
    if col in df.columns:
        # Utilizamos la columna original, que ya fue limpiada de símbolos de texto en el Paso 1
        df[f'{col}_float'] = pd.to_numeric(df[col], errors='coerce')

print("Normalización de fechas y números completada.")

print("\n--- PROCESO DE NORMALIZACIÓN TERMINADO ---")

# ==============================================================================
# 5. CONSTRUCCIÓN DEL DATAFRAME FINAL CON COLUMNAS NORMALIZADAS
# ==============================================================================

print("\n5. Creando el nuevo DataFrame consolidado (df_norm)...")

# Columnas originales que conservaremos (ya limpias por el Paso 1, pero sin sufijo)
cols_originales_a_mantener = ['default', 'denominacion', 'symbol', 'codigoIsin']

# El DF final se construye con las columnas seleccionadas
final_cols = (
    # Claves Normalizadas (Categóricas/Texto)
    ['paisLey_norm', 'tipoEspecie_norm', 'emisor_norm', 'ley_norm', 'moneda_norm', 
     'tipoObligacion_norm', 'tipoGarantia_norm', 'insType_norm',
     'formaAmortizacion_agrupada'] + 
    # Claves Normalizadas (Intereses)
    ['interes_tipo', 'interes_referencia', 'interes_valor_num', 'interes_capitalizacion_freq'] + 
    # Claves Normalizadas (Numéricas/Fechas)
    ['montoNominal_float', 'montoResidual_float', 'denominacionMinima_float',
     'fechaEmision_dt', 'fechaVencimiento_dt', 'fechaDevenganIntereses_dt'] +
    # Claves Originales Limpias que no se normalizan en otra versión
    cols_originales_a_mantener
)

# Filtramos las columnas que realmente existen en el DF después de la normalización
cols_existentes = [col for col in final_cols if col in df.columns]

# Creamos el DataFrame final
df_norm = df[cols_existentes].copy()

print(f"DataFrame 'df_norm' creado con {len(df_norm.columns)} columnas.")


# ==============================================================================
# 6. CÁLCULO DE VALOR FINAL (TEM) - METODOLOGÍA COMPUESTO/SIMPLE
# ==============================================================================

# 1. Función auxiliar para calcular meses enteros y días restantes (Date Offset)
def calcular_meses_y_dias(fecha_inicio, fecha_fin):
    """
    Calcula los meses enteros transcurridos y los días restantes.
    
    Args:
        fecha_inicio (pd.Timestamp): Fecha de Emisión.
        fecha_fin (pd.Timestamp): Fecha de Vencimiento.
        
    Returns:
        tuple: (meses_enteros, dias_restantes)
    """
    if pd.isna(fecha_inicio) or pd.isna(fecha_fin):
        return np.nan, np.nan
    
    # Asegurar que las fechas sean de tipo Timestamp
    fecha_inicio = pd.to_datetime(fecha_inicio)
    fecha_fin = pd.to_datetime(fecha_fin)
    
    # Inicializar contadores
    meses_enteros = 0
    fecha_actual = fecha_inicio
    
    # Avanzar por meses enteros
    while True:
        try:
            # Intentar avanzar un mes (usando DateOffset para manejar fin de mes)
            proximo_mes = fecha_actual + pd.DateOffset(months=1)
        except ValueError:
            # Manejar casos extremos de fechas (raro, pero seguro)
            break
            
        if proximo_mes <= fecha_fin:
            meses_enteros += 1
            fecha_actual = proximo_mes
        else:
            break
            
    # Calcular días restantes (simple diferencia de días)
    dias_restantes = (fecha_fin - fecha_actual).days
    
    return meses_enteros, dias_restantes


def calcular_valor_final_tem(df):
    """
    Calcula el valor final (FV) a vencimiento para letras con Tasa Fija TEM
    usando la convención Compuesto (meses enteros) y Simple (días restantes / 30).
    
    FV = 100 * (1 + TEM)^Meses_Enteros * (1 + TEM * Días_Restantes / 30)
    
    Args:
        df (pd.DataFrame): DataFrame que contiene las columnas normalizadas.
        
    Returns:
        pd.DataFrame: DataFrame con la columna 'valorFinal_calculado' actualizada.
    """
    
    # Condición de Filtrado: Tasa Fija y Referencia TEM
    condicion_tem = (df['interes_tipo'] == 'FIJA') & \
                    (df['interes_referencia'] == 'TEM') & \
                    df['interes_valor_num'].notna() & \
                    df['fechaEmision_dt'].notna() & \
                    df['fechaVencimiento_dt'].notna()

    # Inicializar la columna de Valor Final
    if 'valorFinal_calculado' not in df.columns:
        df['valorFinal_calculado'] = np.nan
    
    # 1. Filtrar el subconjunto que cumple la condición
    df_tem = df[condicion_tem].copy()
    
    if df_tem.empty:
        print("No se encontraron registros válidos con la condición: Tipo FIJA y Referencia TEM para cálculo.")
        return df

    print(f"Calculando Valor Final para {len(df_tem)} registros (Tasa Fija TEM, Método Compuesto/Simple)...")

    # 2. Calcular Meses Enteros y Días Restantes
    fechas = df_tem[['fechaEmision_dt', 'fechaVencimiento_dt']].apply(
        lambda x: calcular_meses_y_dias(x['fechaEmision_dt'], x['fechaVencimiento_dt']),
        axis=1,
        result_type='expand'
    )
    fechas.columns = ['meses_enteros', 'dias_restantes']
    df_tem = pd.concat([df_tem.reset_index(drop=True), fechas.reset_index(drop=True)], axis=1)

    # 3. Preparar componentes de la fórmula
    tasa_decimal = df_tem['interes_valor_num'] / 100
    
    # 4. Calcular los dos factores
    
    # Factor Compuesto (Meses enteros)
    factor_compuesto = (1 + tasa_decimal) ** df_tem['meses_enteros']
    
    # Factor Simple (Días restantes / 30)
    # Se usa la convención (1 + TASA * DÍAS/30)
    factor_simple = 1 + (tasa_decimal * (df_tem['dias_restantes'] / 30))
    
    # 5. Aplicar la fórmula completa
    valor_final_series = 100 * factor_compuesto * factor_simple
    
    # 6. Asignar los resultados al DataFrame original
    df.loc[condicion_tem, 'valorFinal_calculado'] = valor_final_series.values
    
    print("Cálculo de Valor Final para TEM completado con método Compuesto/Simple.")
    return df

# Ejecutar la función sobre el DataFrame normalizado
#df_norm = calcular_valor_final_tem(df_norm) 
df = calcular_valor_final_tem(df)
print("\nLa función 'calcular_valor_final_tem' ha sido actualizada con la metodología Compuesto/Simple.")
print("Asegúrate de ejecutar esta función después de los Pasos 1-5.")


# ==============================================================================
# 7. CÁLCULO DE VALOR FINAL (TNA)
# ==============================================================================

def calcular_valor_final_tna(df):
    """
    Calcula el valor final (FV) para letras con Tasa Fija TNA (Nominal Anual),
    asumiendo capitalización mensual si no se especifica lo contrario.
    
    Fórmula: FV = 100 * (1 + (TNA/12)) ^ Meses * (1 + (TNA/12) * Días/30)
    
    Args:
        df (pd.DataFrame): DataFrame que contiene las columnas normalizadas.
        
    Returns:
        pd.DataFrame: DataFrame con la columna 'valorFinal_calculado' actualizada.
    """
    
    # Condición de Filtrado: Tasa Fija y Referencia TNA
    condicion_tna = (df['interes_tipo'] == 'FIJA') & \
                    (df['interes_referencia'] == 'TNA') & \
                    df['interes_valor_num'].notna() & \
                    df['fechaEmision_dt'].notna() & \
                    df['fechaVencimiento_dt'].notna()

    # 1. Filtrar el subconjunto que cumple la condición
    # Excluimos las filas que ya fueron calculadas en el paso TEM
    condicion_no_calculada = df['valorFinal_calculado'].isna()
    
    df_tna = df[condicion_tna & condicion_no_calculada].copy()
    
    if df_tna.empty:
        print("No se encontraron registros válidos con la condición: Tipo FIJA y Referencia TNA para cálculo.")
        return df

    print(f"Calculando Valor Final para {len(df_tna)} registros (Tasa Fija TNA, Capitalización Mensual)...")

    # 2. Calcular Meses Enteros y Días Restantes (Usamos la función auxiliar del paso anterior)
    # Nota: Asegúrate de que 'calcular_meses_y_dias' esté definida y disponible en el entorno.
    fechas = df_tna[['fechaEmision_dt', 'fechaVencimiento_dt']].apply(
        lambda x: calcular_meses_y_dias(x['fechaEmision_dt'], x['fechaVencimiento_dt']),
        axis=1,
        result_type='expand'
    )
    fechas.columns = ['meses_enteros', 'dias_restantes']
    df_tna = df_tna.reset_index(drop=True).join(fechas.reset_index(drop=True))

    # 3. Calcular la Tasa Efectiva Mensual (TEM) a partir de la TNA
    tasa_tna_decimal = df_tna['interes_valor_num'] / 100
    tasa_tem_decimal = tasa_tna_decimal / 12  # TNA / 12 meses

    # 4. Aplicar la fórmula Compuesto-Simple
    
    # Factor Compuesto (Meses enteros)
    factor_compuesto = (1 + tasa_tem_decimal) ** df_tna['meses_enteros']
    
    # Factor Simple (Días restantes / 30)
    # Usamos la TEM para prorratear los días restantes
    factor_simple = 1 + (tasa_tem_decimal * (df_tna['dias_restantes'] / 30))
    
    # 5. Calcular el Valor Final
    valor_final_series = 100 * factor_compuesto * factor_simple
    
    # 6. Asignar los resultados al DataFrame original
    # Usamos la condición original, ya que el filtro lo hicimos con .copy()
    df.loc[condicion_tna & condicion_no_calculada, 'valorFinal_calculado'] = valor_final_series.values
    
    print("Cálculo de Valor Final para TNA completado.")
    return df

# INTEGRACIÓN EN EL FLUJO:
df = calcular_valor_final_tem(df)
df = calcular_valor_final_tna(df)

df.to_csv("lebacs_limpias_calculadas_v1.csv", sep=";", encoding='utf-8', index=False)
