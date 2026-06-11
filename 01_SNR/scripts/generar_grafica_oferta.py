import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv("06_Consolidado/indice_con_notarias_2025.csv")
fig, ax = plt.subplots(figsize=(10.5, 6.5))
ax.scatter(df["notarias_per_100k"], df["actos_per_capita"], s=72, color="#4682B4", edgecolor="white", linewidth=1.2, alpha=0.85, zorder=3)
destacar = {"Bogotá, D.C.","Boyacá","Guaviare","Atlántico","La Guajira","Vichada","Chocó","Quindío","Antioquia","Meta"}
for _, x in df.iterrows():
    if x["Departamento"] in destacar:
        ax.annotate(x["Departamento"], (x["notarias_per_100k"], x["actos_per_capita"]), xytext=(6,4), textcoords="offset points", fontsize=8.5)
ax.set_xlabel("Notarías por 100.000 habitantes (oferta)"); ax.set_ylabel("Actos notariales per cápita (utilización)")
ax.set_title("Densidad notarial vs utilización por departamento (2025)\nSin asociación significativa: Pearson r = -0.19, p = 0.29, n = 33")
ax.grid(alpha=0.25, linestyle="--"); ax.spines[["top","right"]].set_visible(False)
ax.axvline(df["notarias_per_100k"].median(), color="#888", linestyle=":", alpha=0.7); ax.axhline(df["actos_per_capita"].median(), color="#888", linestyle=":", alpha=0.7)
plt.tight_layout(); plt.savefig("imagenes/scatter_densidad_utilizacion.png", dpi=200, bbox_inches="tight")