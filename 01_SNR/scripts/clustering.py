import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ─────────────────────────────────────────
# CARGAR Y PREPARAR
# ─────────────────────────────────────────
df = pd.read_csv('06_Consolidado/indice_acceso_notarial_2025.csv')
df = df.dropna(subset=['actos_per_capita', 'IPM', 'PIB_per_capita']).copy()

X = df[['actos_per_capita', 'IPM', 'PIB_per_capita']].values

# Estandarizar — K-Means es sensible a escalas
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ─────────────────────────────────────────
# 1. MÉTODO DEL CODO
# ─────────────────────────────────────────
inercias = []
K_range = range(1, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inercias.append(km.inertia_)

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor('#0F1117')
ax.set_facecolor('#0F1117')
ax.plot(K_range, inercias, marker='o', color='#4F8EF7', linewidth=2)
ax.set_xlabel('Número de clusters (k)', color='#9AA3BC')
ax.set_ylabel('Inercia', color='#9AA3BC')
ax.set_title('Método del codo — selección de k óptimo',
             color='#E8EBF4', fontsize=13, fontweight='bold', pad=14)
ax.tick_params(colors='#9AA3BC')
ax.spines[['top','right']].set_visible(False)
ax.spines[['bottom','left']].set_color('#2A3147')
ax.grid(color='#2A3147', linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.savefig('imagenes/metodo_codo.png', dpi=150,
            bbox_inches='tight', facecolor='#0F1117')
plt.close()
print("✓ imagenes/metodo_codo.png")

# ─────────────────────────────────────────
# 2. K-MEANS CON k=4
# ─────────────────────────────────────────
k_optimo = 4
km = KMeans(n_clusters=k_optimo, random_state=42, n_init=10)
df['cluster'] = km.fit_predict(X_scaled)

# Caracterizar cada cluster
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
# 3. VISUALIZACIÓN
# ─────────────────────────────────────────
colores_cluster = ['#4F8EF7', '#3ABFA0', '#E8A23A', '#E85A5A']

fig, ax = plt.subplots(figsize=(11, 7.5))
fig.patch.set_facecolor('#0F1117')
ax.set_facecolor('#0F1117')

for c in sorted(df['cluster'].unique()):
    sub = df[df['cluster'] == c]
    ax.scatter(sub['IPM'], sub['actos_per_capita'],
               c=colores_cluster[c], s=110, alpha=0.9,
               edgecolors='none', zorder=3,
               label=f'Grupo {c+1} (n={len(sub)})')

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
ax.set_title(f'Agrupación de departamentos por K-Means (k={k_optimo})\nVariables: actos per cápita, IPM y PIB per cápita',
             color='#E8EBF4', fontsize=13, fontweight='bold', pad=16)
ax.tick_params(colors='#9AA3BC', labelsize=9)
ax.spines[['top','right']].set_visible(False)
ax.spines[['bottom','left']].set_color('#2A3147')
ax.grid(color='#2A3147', linewidth=0.5, alpha=0.5)
ax.legend(facecolor='#181C26', edgecolor='#2A3147',
          labelcolor='#9AA3BC', fontsize=9)

plt.tight_layout()
plt.savefig('imagenes/clusters_kmeans.png', dpi=150,
            bbox_inches='tight', facecolor='#0F1117')
plt.close()
print("\n✓ imagenes/clusters_kmeans.png")

# Guardar resultado
df[['Departamento', 'actos_per_capita', 'IPM',
    'PIB_per_capita', 'clasificacion', 'cluster']].to_csv(
    '06_Consolidado/clusters_departamentos.csv',
    index=False, encoding='utf-8-sig')
print("✓ 06_Consolidado/clusters_departamentos.csv")