"""
consolidar_sqlite.py
Carga todos los CSVs finales del análisis a la base SQLite central.
Idempotente: si la tabla existe, la reemplaza.
"""

import sqlite3
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent  # raíz del proyecto
DB = BASE / "06_Consolidado" / "acceso_notarial_colombia.db"

CARGAS = [
    # (ruta_csv, nombre_tabla_sqlite)
    ("06_Consolidado/indice_acceso_notarial_2025.csv", "indice_2025"),
    ("06_Consolidado/indice_con_notarias_2025.csv", "indice_con_notarias"),
    ("06_Consolidado/panel_2021_2025.csv", "panel_temporal"),
    ("06_Consolidado/clusters_departamentos.csv", "clusters"),
    ("06_Consolidado/ranking_2025.csv", "ranking"),
    ("01_SNR/fuentes/snr_notarias/notarias_por_departamento.csv", "notarias_por_departamento"),
]

con = sqlite3.connect(DB)
print(f"Conectado a: {DB}\n")

for ruta_rel, tabla in CARGAS:
    ruta = BASE / ruta_rel
    if not ruta.exists():
        print(f"  [SALTADO] No existe: {ruta_rel}")
        continue
    df = pd.read_csv(ruta)
    df.to_sql(tabla, con, if_exists="replace", index=False)
    print(f"  [OK] {tabla:30s}  {len(df):4d} filas  ←  {ruta_rel}")

# Verificación final
print("\nEstado final de la BD:")
cur = con.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
for (nombre,) in cur.fetchall():
    cur.execute(f"SELECT COUNT(*) FROM {nombre};")
    n = cur.fetchone()[0]
    print(f"  {nombre:30s}  {n:5d} filas")

con.close()
print("\nListo.")