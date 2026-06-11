"""
grafica_oferta.py
Genera el scatter de densidad notarial vs utilizacion por departamento.
Entrada:  06_Consolidado/indice_con_notarias_2025.csv  (lo crea integrar_notarias.py)
Salida:   imagenes/scatter_densidad_utilizacion.png
Uso:      python 01_SNR/scripts/grafica_oferta.py   (ejecutar desde la raiz)
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

ENTRADA = "06_Consolidado/indice_con_notarias_2025.csv"
SALIDA  = "imagenes/scatter_densidad_utilizacion.png"

# --- 1. Cargar datos ---
if not os.path.exists(ENTRADA):
    raise FileNotFoundError(
        f"No encuentro {ENTRADA}. Corre primero integrar_notarias.py "
        "para generarlo.")
df = pd.read_csv(ENTRADA)

# --- 2. Estadisticos para el subtitulo (asi siempre cuadran con los datos) ---
sub = df[["notarias_per_100k", "actos_per_capita"]].dropna()
r, p = stats.pearsonr(sub["notarias_per_100k"], sub["actos_per_capita"])
n = len(sub)

# --- 3. Figura ---
fig, ax = plt.subplots(figsize=(10.5, 6.5))
ax.scatter(df["notarias_per_100k"], df["actos_per_capita"],
           s=72, color="#4682B4", edgecolor="white",
           linewidth=1.2, alpha=0.85, zorder=3)

# Etiquetar solo los casos extremos / interesantes (no todos, para no saturar)
DESTACAR = {
    "Bogotá, D.C.", "Boyacá", "Guaviare", "Atlántico", "La Guajira",
    "Vichada", "Chocó", "Quindío", "Antioquia", "Meta",
    "Archipiélago De San Andrés, Providencia Y Santa Catalina",
}
ALIAS = {  # nombres mas cortos para anotacion
    "Archipiélago De San Andrés, Providencia Y Santa Catalina": "San Andrés",
    "Bogotá, D.C.": "Bogotá D.C.",
}
for _, x in df.iterrows():
    if x["Departamento"] in DESTACAR:
        etiqueta = ALIAS.get(x["Departamento"], x["Departamento"])
        ax.annotate(etiqueta,
                    (x["notarias_per_100k"], x["actos_per_capita"]),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=8.5, color="#333")

# --- 4. Estetica ---
ax.set_xlabel("Notarías por 100.000 habitantes (oferta)", fontsize=11)
ax.set_ylabel("Actos notariales per cápita (utilización)", fontsize=11)
ax.set_title(
    "Densidad notarial vs utilización por departamento (2025)\n"
    f"Pearson r = {r:+.2f}, p = {p:.2f}, n = {n}  "
    f"({'sin asociación significativa' if p > 0.05 else 'asociación significativa'})",
    fontsize=12, pad=12)
ax.grid(alpha=0.25, linestyle="--", linewidth=0.6, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
ax.set_axisbelow(True)

# Lineas de mediana como referencia visual
ax.axvline(df["notarias_per_100k"].median(),
           color="#888", linestyle=":", linewidth=0.8, alpha=0.7)
ax.axhline(df["actos_per_capita"].median(),
           color="#888", linestyle=":", linewidth=0.8, alpha=0.7)

# --- 5. Guardar ---
os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
plt.tight_layout()
plt.savefig(SALIDA, dpi=200, bbox_inches="tight")
plt.close()
print(f"Generado: {SALIDA}")
print(f"  n = {n}, Pearson r = {r:+.3f}, p = {p:.4f}")