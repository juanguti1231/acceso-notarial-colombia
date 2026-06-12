"""
explorar_geojson.py
Descarga el GeoJSON de departamentos de Colombia y muestra
su estructura, para diseñar correctamente el choropleth de Plotly.
"""

import json
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
OUT_DIR = BASE / "07_Dashboard"
OUT_DIR.mkdir(parents=True, exist_ok=True)
GEOJSON_PATH = OUT_DIR / "colombia_departamentos.geojson"

# Fuente: gist público de John Guerra (verificado, ampliamente usado)
URL = "https://gist.githubusercontent.com/john-guerra/43c7656821069d00dcbc/raw/be6a6e239cd5b5b803c6e7c2ec405b793a9064dd/Colombia.geo.json"

print(f"Descargando GeoJSON de Colombia...\n  {URL}\n")
try:
    urllib.request.urlretrieve(URL, GEOJSON_PATH)
    tam_kb = GEOJSON_PATH.stat().st_size / 1024
    print(f"Guardado en: {GEOJSON_PATH}")
    print(f"Tamaño: {tam_kb:.1f} KB\n")
except Exception as e:
    print(f"Error al descargar: {e}")
    raise

with open(GEOJSON_PATH, encoding="utf-8") as f:
    geojson = json.load(f)

print(f"Tipo: {geojson.get('type')}")
print(f"Número de departamentos en el GeoJSON: {len(geojson['features'])}\n")

print("Propiedades del primer feature (estructura):")
print(json.dumps(geojson["features"][0]["properties"], indent=2, ensure_ascii=False))

print("\nLista completa de departamentos en el GeoJSON:")
for i, feat in enumerate(geojson["features"], 1):
    props = feat["properties"]
    print(f"  {i:2d}. {props}")