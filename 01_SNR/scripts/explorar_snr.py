import pandas as pd
import sqlite3
import os

# ─────────────────────────────────────────
# 1. CARGAR Y LIMPIAR
# ─────────────────────────────────────────
df = pd.read_csv(
    '01_SNR/fuentes/snr_actos/Actos_Jurídicos_Notariales_20260609.csv',
    encoding='latin-1',
    low_memory=False
)

# Corregir encoding
df['Departamento'] = df['Departamento'].str.encode(
    'latin-1').str.decode('utf-8', errors='ignore')

# Detectar columnas numéricas
cols_excluir = [c for c in df.columns
                if any(x in c for x in ['Fecha', 'Departamento', 'digo'])]
cols_numericas = df.columns.drop(cols_excluir)

# Limpiar columnas numéricas
for col in cols_numericas:
    df[col] = pd.to_numeric(
        df[col].astype(str).str.replace(',', ''),
        errors='coerce'
    )

# Normalizar fechas
df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
df['Año'] = df['Fecha'].dt.year
df['Mes'] = df['Fecha'].dt.month

# Rellenar NaN con 0
df[cols_numericas] = df[cols_numericas].fillna(0)

print("=== DATOS LIMPIOS ===")
print(f"Filas: {len(df)}")
print(f"Rango: {df['Fecha'].min()} → {df['Fecha'].max()}")
print(f"Años: {sorted(df['Año'].dropna().unique())}")
print(f"Departamentos: {df['Departamento'].nunique()}")

# ─────────────────────────────────────────
# 2. AGREGAR POR DEPARTAMENTO Y AÑO
# ─────────────────────────────────────────
resumen = df.groupby(['Departamento', 'Año'])[
    cols_numericas].sum().reset_index()

# ─────────────────────────────────────────
# 3. PARTICIPACIÓN DE BOGOTÁ
# ─────────────────────────────────────────
bogota = resumen[resumen['Departamento'].str.contains(
    'Bogot', na=False)]

total_cv = resumen.groupby('Año')['Compraventa'].sum()
bogota_cv = bogota.set_index('Año')['Compraventa']
participacion = (bogota_cv / total_cv * 100).round(2)

print("\n=== PARTICIPACIÓN BOGOTÁ EN COMPRAVENTAS ===")
for año, pct in participacion.items():
    print(f"  {int(año)}: {pct}%")

# ─────────────────────────────────────────
# 4. RANKING 2025
# ─────────────────────────────────────────
ultimo_año = resumen[resumen['Año'] == 2025].copy()
ultimo_año['total_actos'] = ultimo_año[cols_numericas].sum(axis=1)
ultimo_año = ultimo_año.sort_values(
    'total_actos', ascending=False).reset_index(drop=True)
ultimo_año['posicion'] = ultimo_año.index + 1

print("\n=== TOP 10 DEPARTAMENTOS 2025 ===")
print(ultimo_año[['posicion', 'Departamento',
                   'total_actos', 'Compraventa']].head(10).to_string())

pos_bogota = ultimo_año[ultimo_año['Departamento'].str.contains(
    'Bogot', na=False)]
print(f"\n>>> Bogotá posición: {pos_bogota['posicion'].values}")

# ─────────────────────────────────────────
# 5. TOP ACTOS EN BOGOTÁ 2025
# ─────────────────────────────────────────
bogota_2025 = bogota[bogota['Año'] == 2025][cols_numericas].sum()
bogota_2025 = bogota_2025.sort_values(ascending=False)

print("\n=== TOP 10 ACTOS BOGOTÁ 2025 ===")
print(bogota_2025.head(10).to_string())

# ─────────────────────────────────────────
# 6. EXPORTAR
# ─────────────────────────────────────────
os.makedirs('06_Consolidado', exist_ok=True)

resumen.to_csv(
    '06_Consolidado/resumen_departamento.csv',
    index=False, encoding='utf-8-sig')
ultimo_año.to_csv(
    '06_Consolidado/ranking_2025.csv',
    index=False, encoding='utf-8-sig')

conn = sqlite3.connect('06_Consolidado/acceso_notarial_colombia.db')
df.to_sql('actos_notariales', conn,
          if_exists='replace', index=False)
resumen.to_sql('resumen_departamento', conn,
               if_exists='replace', index=False)
ultimo_año.to_sql('ranking_departamentos', conn,
                  if_exists='replace', index=False)
conn.close()

print("\n=== ARCHIVOS GENERADOS ===")
print("✓ 06_Consolidado/resumen_departamento.csv")
print("✓ 06_Consolidado/ranking_2025.csv")
print("✓ 06_Consolidado/acceso_notarial_colombia.db")