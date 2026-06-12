import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# ─────────────────────────────────────────
# CARGAR Y PREPARAR
# ─────────────────────────────────────────
df = pd.read_csv('06_Consolidado/indice_acceso_notarial_2025.csv')
df = df.dropna(subset=['actos_per_capita', 'IPM', 'PIB_per_capita']).copy()

X = df[['actos_per_capita', 'IPM', 'PIB_per_capita']].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ─────────────────────────────────────────
# 1. SELECCIÓN DE k — SILHOUETTE SCORE
#    (más objetivo que el método del codo con n=33)
# ─────────────────────────────────────────
silhouette_vals = {}
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    silhouette_vals[k] = silhouette_score(X_scaled, labels)

k_optimo = max(silhouette_vals, key=silhouette_vals.get)

print("=== SELECCIÓN DE k (Silhouette Score) ===")
print(f"{'k':>4}  {'Silhouette':>12}")
for k, s in silhouette_vals.items():
    marca = " ← óptimo" if k == k_optimo else ""
    print(f"{k:>4}  {s:>12.4f}{marca}")

# Gráfico Silhouette
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor('#0F1117')
ax.set_facecolor('#0F1117')
ks   = list(silhouette_vals.keys())
sils = list(silhouette_vals.values())
bars = ax.bar(ks, sils, color='#4F8EF7', alpha=0.8, width=0.6)
bars[ks.index(k_optimo)].set_color('#3ABFA0')
ax.set_xlabel('Número de clusters (k)', color='#9AA3BC')
ax.set_ylabel('Silhouette Score', color='#9AA3BC')
ax.set_title('Selección de k óptimo — Silhouette Score\n'
             f'k={k_optimo} maximiza la cohesión interna',
             color='#E8EBF4', fontsize=13, fontweight='bold', pad=14)
ax.tick_params(colors='#9AA3BC')
ax.spines[['top', 'right']].set_visible(False)
ax.spines[['bottom', 'left']].set_color('#2A3147')
ax.grid(color='#2A3147', linewidth=0.5, alpha=0.5, axis='y')
plt.tight_layout()
plt.savefig('imagenes/silhouette_score.png', dpi=150,
            bbox_inches='tight', facecolor='#0F1117')
plt.close()
print("\n✓ imagenes/silhouette_score.png")

# ─────────────────────────────────────────
# 2. CLUSTERING JERÁRQUICO (Ward) — validación independiente
# ─────────────────────────────────────────
Z = linkage(X_scaled, method='ward')
df['cluster_jerarquico'] = fcluster(Z, t=k_optimo, criterion='maxclust')

# Dendrograma
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor('#0F1117')
ax.set_facecolor('#0F1117')
nombres = [d.split(',')[0][:14] for d in df['Departamento']]
dendrogram(Z, labels=nombres, ax=ax,
           color_threshold=Z[-(k_optimo - 1), 2],
           above_threshold_color='#9AA3BC',
           leaf_rotation=90, leaf_font_size=8)
ax.set_title(f'Dendrograma — Clustering jerárquico Ward (k={k_optimo})',
             color='#E8EBF4', fontsize=13, fontweight='bold', pad=14)
ax.set_ylabel('Distancia (Ward)', color='#9AA3BC')
ax.tick_params(colors='#9AA3BC', labelsize=7)
ax.spines[['top', 'right']].set_visible(False)
ax.spines[['bottom', 'left']].set_color('#2A3147')
for item in ax.get_xticklabels():
    item.set_color('#9AA3BC')
plt.tight_layout()
plt.savefig('imagenes/dendrograma.png', dpi=150,
            bbox_inches='tight', facecolor='#0F1117')
plt.close()
print("✓ imagenes/dendrograma.png")

# ─────────────────────────────────────────
# 3. K-MEANS CON k ÓPTIMO
# ─────────────────────────────────────────
km = KMeans(n_clusters=k_optimo, random_state=42, n_init=10)
df['cluster'] = km.fit_predict(X_scaled)

# Verificar convergencia entre métodos
acuerdo = (df['cluster_jerarquico'] - 1 == df['cluster']).mean()
print(f"\nAcuerdo K-Means / Jerárquico (mismo k): {acuerdo:.1%}")
print("(Nota: las etiquetas de cluster pueden diferir aunque los grupos coincidan)")

# Caracterización
print("\n=== CARACTERIZACIÓN DE CLUSTERS ===")
resumen = df.groupby('cluster').agg(
    n=('Departamento', 'count'),
    actos_pc_prom=('actos_per_capita', 'mean'),
    ipm_prom=('IPM', 'mean'),
    pib_pc_prom=('PIB_per_capita', 'mean'),
    departamentos=('Departamento', lambda x: ', '.join(x))
).round(2)
print(resumen.to_string())

# ─────────────────────────────────────────
# 4. VISUALIZACIÓN CLUSTERS (IPM vs actos per cápita)
# ─────────────────────────────────────────
colores_cluster = ['#4F8EF7', '#3ABFA0', '#E8A23A', '#E85A5A',
                   '#A855F7', '#EC4899'][:k_optimo]

fig, ax = plt.subplots(figsize=(11, 7.5))
fig.patch.set_facecolor('#0F1117')
ax.set_facecolor('#0F1117')

for c in sorted(df['cluster'].unique()):
    sub = df[df['cluster'] == c]
    ax.scatter(sub['IPM'], sub['actos_per_capita'],
               c=colores_cluster[c], s=110, alpha=0.9,
               edgecolors='none', zorder=3,
               label=f'Grupo {c + 1} (n={len(sub)})')

for _, row in df.iterrows():
    nombre = row['Departamento'].split(',')[0]
    if len(nombre) > 12:
        nombre = nombre[:12] + '.'
    ax.annotate(nombre, (row['IPM'], row['actos_per_capita']),
                textcoords='offset points', xytext=(6, 3),
                fontsize=7, color='#9AA3BC')

ax.set_xlabel('Índice de Pobreza Multidimensional — IPM (%)',
              color='#9AA3BC', fontsize=10)
ax.set_ylabel('Actos notariales per cápita',
              color='#9AA3BC', fontsize=10)
ax.set_title(f'Agrupación de departamentos — K-Means (k={k_optimo})\n'
             f'Silhouette Score = {silhouette_vals[k_optimo]:.3f} | '
             'Variables: actos per cápita, IPM y PIB per cápita',
             color='#E8EBF4', fontsize=12, fontweight='bold', pad=16)
ax.tick_params(colors='#9AA3BC', labelsize=9)
ax.spines[['top', 'right']].set_visible(False)
ax.spines[['bottom', 'left']].set_color('#2A3147')
ax.grid(color='#2A3147', linewidth=0.5, alpha=0.5)
ax.legend(facecolor='#181C26', edgecolor='#2A3147',
          labelcolor='#9AA3BC', fontsize=9)

plt.tight_layout()
plt.savefig('imagenes/clusters_kmeans.png', dpi=150,
            bbox_inches='tight', facecolor='#0F1117')
plt.close()
print("✓ imagenes/clusters_kmeans.png")

# ─────────────────────────────────────────
# 5. EXPORTAR
# ─────────────────────────────────────────
df[['Departamento', 'actos_per_capita', 'IPM', 'PIB_per_capita',
    'clasificacion', 'cluster', 'cluster_jerarquico']].to_csv(
    '06_Consolidado/clusters_departamentos.csv',
    index=False, encoding='utf-8-sig')
print("✓ 06_Consolidado/clusters_departamentos.csv")
