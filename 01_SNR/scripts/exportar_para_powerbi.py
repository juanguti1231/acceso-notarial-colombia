"""
exportar_para_powerbi.py
Lee desde SQLite (almacén central) usando consultas SQL y exporta
un archivo Excel multipestaña listo para subir a Power BI Service.
"""

import sqlite3
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
DB = BASE / "06_Consolidado" / "acceso_notarial_colombia.db"
OUT_DIR = BASE / "07_Dashboard"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "datos_powerbi.xlsx"

con = sqlite3.connect(DB)
print(f"Conectado a: {DB}\n")

# Consultas SQL que extraen las tablas limpias para el dashboard
indice = pd.read_sql_query(
    "SELECT * FROM indice_con_notarias ORDER BY departamento", con)

panel = pd.read_sql_query(
    "SELECT * FROM panel_temporal ORDER BY departamento, año", con)

clusters = pd.read_sql_query(
    "SELECT * FROM clusters ORDER BY departamento", con)

ranking = pd.read_sql_query("SELECT * FROM ranking", con)

notarias = pd.read_sql_query(
    "SELECT * FROM notarias_por_departamento ORDER BY departamento", con)

con.close()

# Mostrar columnas para verificación
print("Columnas de cada tabla extraída:")
for nombre, df in [
    ("indice_con_notarias", indice),
    ("panel_temporal", panel),
    ("clusters", clusters),
    ("ranking", ranking),
    ("notarias", notarias),
]:
    print(f"  {nombre:25s} {len(df):4d} filas  cols: {list(df.columns)}")

# Exportar a Excel multipestaña
with pd.ExcelWriter(OUT_FILE, engine="openpyxl") as writer:
    indice.to_excel(writer, sheet_name="indice_2025", index=False)
    panel.to_excel(writer, sheet_name="panel_temporal", index=False)
    clusters.to_excel(writer, sheet_name="clusters", index=False)
    ranking.to_excel(writer, sheet_name="ranking", index=False)
    notarias.to_excel(writer, sheet_name="notarias", index=False)

print(f"\nArchivo exportado:")
print(f"  {OUT_FILE}")
print(f"\nListo para subir a app.powerbi.com")