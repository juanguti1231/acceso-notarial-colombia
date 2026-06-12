import pandas as pd
import sqlite3
import os

# ─────────────────────────────────────────
# 1. POBLACIÓN
# ─────────────────────────────────────────
pob_raw = pd.read_excel(
    '01_SNR/fuentes/dane_poblacion/poblacion_departamentos.xlsx',
    sheet_name='PobDepartamentalxÁrea', header=7)
pob_raw.columns = ['DP', 'Departamento', 'Año', 'Area', 'Poblacion']
pob = pob_raw[pob_raw['Area'] == 'Total'][
    ['Departamento', 'Año', 'Poblacion']].copy()
pob['Departamento'] = pob['Departamento'].str.strip()
pob['Año'] = pob['Año'].astype(int)
pob['Poblacion'] = pd.to_numeric(pob['Poblacion'], errors='coerce')
pob = pob[pob['Año'].between(2021, 2025)]

# ─────────────────────────────────────────
# 2. PIB
# ─────────────────────────────────────────
pib_raw = pd.read_excel(
    '01_SNR/fuentes/dane_pib/pib_departamentos.xlsx',
    sheet_name='Cuadro 1', header=9)
pib_raw = pib_raw.rename(columns={
    'Código Departamento (DIVIPOLA)': 'Codigo',
    'DEPARTAMENTOS': 'Departamento',
    '2023p': 2023, '2024pr': 2024
})
pib_raw = pib_raw[
    pib_raw['Departamento'] != 'COLOMBIA'].dropna(subset=['Departamento'])
year_cols = [c for c in pib_raw.columns
             if str(c).replace('.0', '').strip().isdigit()
             and 2021 <= int(str(c).replace('.0', '').strip()) <= 2024]
pib = pib_raw[['Departamento'] + year_cols].melt(
    id_vars='Departamento', var_name='Año', value_name='PIB_miles_millones')
pib['Año'] = pib['Año'].astype(int)
pib['Departamento'] = pib['Departamento'].str.strip()
pib['PIB_miles_millones'] = pd.to_numeric(
    pib['PIB_miles_millones'], errors='coerce')

# ─────────────────────────────────────────
# 3. IPM
# ─────────────────────────────────────────
ipm_raw = pd.read_excel(
    '01_SNR/fuentes/dnp_ipm/ipm_departamentos.xlsx',
    sheet_name='IPM_Departamentos', header=11)
años_ipm = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
new_cols = ['Departamento']
for a in años_ipm:
    new_cols += [f'IPM_{a}_Total', f'IPM_{a}_Cabeceras', f'IPM_{a}_Rural']
ipm_raw.columns = new_cols[:len(ipm_raw.columns)]
ipm_raw = ipm_raw.dropna(subset=['Departamento'])
ipm_raw = ipm_raw[~ipm_raw['Departamento'].astype(str).str.contains(
    'NaN|Departamento', na=True)]
ipm_cols = ['Departamento'] + [
    f'IPM_{a}_Total' for a in años_ipm
    if f'IPM_{a}_Total' in ipm_raw.columns]
ipm = ipm_raw[ipm_cols].melt(
    id_vars='Departamento', var_name='var', value_name='IPM')
ipm['Año'] = ipm['var'].str.extract(r'IPM_(\d{4})').astype(int)
ipm = ipm[['Departamento', 'Año', 'IPM']].copy()
ipm['Departamento'] = ipm['Departamento'].str.strip()
ipm['IPM'] = pd.to_numeric(ipm['IPM'], errors='coerce')
ipm = ipm[ipm['Año'].between(2021, 2025)]

# ─────────────────────────────────────────
# 4. NORMALIZAR NOMBRES
# ─────────────────────────────────────────
nombre_map = {
    'Bogotá D.C.': 'Bogotá, D.C.',
    'Bogotá D. C.': 'Bogotá, D.C.',
    'Norte de Santander': 'Norte De Santander',
    'Valle del Cauca': 'Valle Del Cauca',
    'Archipiélago de San Andrés, Providencia y Santa Catalina':
        'Archipiélago De San Andrés, Providencia Y Santa Catalina',
}

def normalizar(df, col):
    df[col] = df[col].str.strip().replace(nombre_map)
    return df

pob = normalizar(pob, 'Departamento')
pib = normalizar(pib, 'Departamento')
ipm = normalizar(ipm, 'Departamento')

# ─────────────────────────────────────────
# 5. SNR 2025
# ─────────────────────────────────────────
snr = pd.read_csv(
    '01_SNR/fuentes/snr_actos/Actos_Jurídicos_Notariales_20260609.csv',
    encoding='latin-1', low_memory=False)
snr['Departamento'] = snr['Departamento'].str.encode(
    'latin-1').str.decode('utf-8', errors='ignore')
snr['Fecha'] = pd.to_datetime(snr['Fecha'], errors='coerce')
snr['Año'] = snr['Fecha'].dt.year

cols_excluir = [c for c in snr.columns
                if any(x in c for x in ['Fecha', 'Departamento', 'digo', 'Año'])]
cols_num = snr.columns.drop(cols_excluir)
for c in cols_num:
    snr[c] = pd.to_numeric(
        snr[c].astype(str).str.replace(',', ''), errors='coerce')
snr[cols_num] = snr[cols_num].fillna(0)

snr_2025 = snr[snr['Año'] == 2025].groupby(
    'Departamento')[cols_num].sum().reset_index()
snr_2025['total_actos'] = snr_2025[cols_num].sum(axis=1)
snr_2025 = normalizar(snr_2025, 'Departamento')

# ─────────────────────────────────────────
# 5b. COMPOSICIÓN DE ACTOS (grupos funcionales)
# ─────────────────────────────────────────
def _cols(df, *keywords):
    """Suma las columnas cuyo nombre contiene alguna de las palabras clave."""
    matched = [c for c in df.columns
               if any(k.lower() in c.lower() for k in keywords)]
    return df[matched].sum(axis=1) if matched else pd.Series(0, index=df.index)

snr_2025['grp_inmobiliario']    = _cols(snr_2025,
    'Compraventa', 'Hipotec', 'Cancelaci', 'Vivienda',
    'Permuta', 'Bien Inmueble', 'Remate', 'Arrendamiento')
snr_2025['grp_autenticaciones'] = _cols(snr_2025,
    'Autenticaci', 'Actas de Comparecencia',
    'Declaraci', 'Conciliaci')
snr_2025['grp_registro_civil']  = _cols(snr_2025,
    'Registro Civil', 'Copias del Registro',
    'Legitimaci', 'Cambio de Nombre', 'Correci')
snr_2025['grp_familia']         = _cols(snr_2025,
    'Matrimoni', 'Divorcio', 'Disoluci', 'Capitulaci', 'Uniones')
snr_2025['grp_sociedades']      = _cols(snr_2025,
    'Sociedad', 'Fiducia', 'Leasing', 'Insolvencia')

# ─────────────────────────────────────────
# 6. CRUCE Y CÁLCULO DEL ÍNDICE
# ─────────────────────────────────────────
pob_2025 = pob[pob['Año'] == 2025][['Departamento', 'Poblacion']]
pib_2024 = pib[pib['Año'] == 2024][['Departamento', 'PIB_miles_millones']]
ipm_2025 = ipm[ipm['Año'] == 2025][['Departamento', 'IPM']]

comp_cols = ['grp_inmobiliario', 'grp_autenticaciones',
             'grp_registro_civil', 'grp_familia', 'grp_sociedades']

df = snr_2025[['Departamento', 'total_actos', 'Compraventa'] + comp_cols].merge(
    pob_2025, on='Departamento', how='left').merge(
    pib_2024, on='Departamento', how='left').merge(
    ipm_2025, on='Departamento', how='left')

df['actos_per_capita'] = (df['total_actos'] / df['Poblacion']).round(4)
df['compraventas_per_capita'] = (
    df['Compraventa'] / df['Poblacion']).round(6)
df['PIB_per_capita'] = (
    df['PIB_miles_millones'] * 1e9 / df['Poblacion']).round(0)

# Participaciones porcentuales de composición
for g in comp_cols:
    df[g.replace('grp_', 'pct_')] = (
        df[g] / df['total_actos'] * 100).round(2)

# ─────────────────────────────────────────
# ÍNDICE A — todos los actos (IIAN completo)
# ÍNDICE B — sin autenticaciones/declaraciones
# ÍNDICE C — solo actos patrimoniales
# ─────────────────────────────────────────
df['iian_a_pc'] = (df['total_actos'] / df['Poblacion']).round(4)

total_sin_aut = df['total_actos'] - df['grp_autenticaciones']
df['iian_b_pc'] = (total_sin_aut / df['Poblacion']).round(4)

df['iian_c_pc'] = (df['grp_inmobiliario'] / df['Poblacion']).round(6)

# Calcular desviación y clasificación para los tres índices
def desviacion_y_clasificacion(serie, prefijo, df_out):
    prom = serie.mean()
    desv = ((serie - prom) / prom * 100).round(2)
    df_out[f'{prefijo}_desv_pct'] = desv
    df_out[f'{prefijo}_pos'] = serie.rank(ascending=False).astype(int)
    return desv

def clasificar(d):
    if d > 50:     return 'Alta intensidad'
    elif d > 10:   return 'Sobre promedio'
    elif d >= -10: return 'Promedio'
    elif d >= -50: return 'Baja intensidad'
    else:          return 'Muy baja intensidad'

desv_a = desviacion_y_clasificacion(df['iian_a_pc'], 'iian_a', df)
desv_b = desviacion_y_clasificacion(df['iian_b_pc'], 'iian_b', df)
desv_c = desviacion_y_clasificacion(df['iian_c_pc'], 'iian_c', df)

df['clasificacion']   = desv_a.apply(clasificar)
df['clasificacion_b'] = desv_b.apply(clasificar)
df['clasificacion_c'] = desv_c.apply(clasificar)

# Mantener compatibilidad con código existente
df['actos_per_capita'] = df['iian_a_pc']
df['desviacion_pct']   = df['iian_a_desv_pct']

df = df.sort_values('iian_a_pc', ascending=False).reset_index(drop=True)
df['posicion'] = df.index + 1

print("=== ÍNDICE DE INTENSIDAD DE ACTIVIDAD NOTARIAL (IIAN) 2025 ===")
print("\n--- IIAN-A (todos los actos) ---")
print(df[['posicion', 'Departamento', 'iian_a_pc',
          'iian_a_desv_pct', 'clasificacion']].head(10).to_string())

print("\n--- IIAN-B (sin autenticaciones) — top 10 ---")
df_b = df.sort_values('iian_b_pc', ascending=False)
print(df_b[['Departamento', 'iian_b_pc', 'iian_b_desv_pct',
            'clasificacion_b']].head(10).to_string(index=False))

print("\n--- IIAN-C (solo patrimonial) — top 10 ---")
df_c = df.sort_values('iian_c_pc', ascending=False)
print(df_c[['Departamento', 'iian_c_pc', 'iian_c_desv_pct',
            'clasificacion_c']].head(10).to_string(index=False))

print("\n--- Comparación de rankings (¿cambia quién es top?) ---")
comp = df[['Departamento', 'iian_a_pos', 'iian_b_pos', 'iian_c_pos']].copy()
comp['var_ab'] = (comp['iian_a_pos'] - comp['iian_b_pos']).abs()
comp['var_ac'] = (comp['iian_a_pos'] - comp['iian_c_pos']).abs()
print(comp.sort_values('var_ac', ascending=False).head(10).to_string(index=False))

# ─────────────────────────────────────────
# 7. EXPORTAR
# ─────────────────────────────────────────
os.makedirs('06_Consolidado', exist_ok=True)

df.to_csv(
    '06_Consolidado/indice_acceso_notarial_2025.csv',
    index=False, encoding='utf-8-sig')

conn = sqlite3.connect('06_Consolidado/acceso_notarial_colombia.db')
df.to_sql('indice_iian_2025', conn, if_exists='replace', index=False)
df.to_sql('indice_acceso',    conn, if_exists='replace', index=False)
pob.to_sql('poblacion_departamentos', conn,
           if_exists='replace', index=False)
pib.to_sql('pib_departamentos', conn,
           if_exists='replace', index=False)
ipm.to_sql('ipm_departamentos', conn,
           if_exists='replace', index=False)
conn.close()

print("\n✓ 06_Consolidado/indice_acceso_notarial_2025.csv")
print("✓ 06_Consolidado/acceso_notarial_colombia.db")