import pandas as pd
import numpy as np
import geopandas as gpd
import libpysal
from esda.moran import Moran, Moran_Local
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# ─────────────────────────────────────────
# CARGAR DATOS
# ─────────────────────────────────────────
gdf = gpd.read_file('07_Dashboard/colombia_departamentos.geojson')
df  = pd.read_csv('06_Consolidado/indice_acceso_notarial_2025.csv')

# Normalizar nombres para el join (GeoJSON usa NOMBRE_DPT en mayúsculas sin tildes)
import unicodedata

def normalizar_str(s):
    s = str(s).upper().strip()
    s = unicodedata.normalize('NFD', s)
    s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
    return s

df['Departamento_norm']  = df['Departamento'].apply(normalizar_str)
gdf['Departamento_norm'] = gdf['NOMBRE_DPT'].apply(normalizar_str)

gdf = gdf.merge(df[['Departamento_norm', 'Departamento',
                     'actos_per_capita', 'clasificacion']],
                on='Departamento_norm', how='left')

gdf = gdf.dropna(subset=['actos_per_capita']).reset_index(drop=True)
print(f"Departamentos con datos: {len(gdf)}")

# ─────────────────────────────────────────
# MATRIZ DE PESOS ESPACIALES
# Se usa KNN(k=5) sobre centroides en lugar de contigüidad Queen
# porque el GeoJSON no garantiza bordes compartidos exactos.
# KNN es más robusto y apropiado cuando la topología no está validada.
# ─────────────────────────────────────────
gdf = gdf.copy()
# Proyectar a MAGNA-SIRGAS / Colombia Bogotá (EPSG:3116) para centroides correctos
gdf_proj = gdf.to_crs(epsg=3116)
gdf['centroid_x'] = gdf_proj.geometry.centroid.x
gdf['centroid_y'] = gdf_proj.geometry.centroid.y

coords = list(zip(gdf['centroid_x'], gdf['centroid_y']))
w = libpysal.weights.KNN.from_array(coords, k=5)
w.transform = 'r'   # row-standardized

# ─────────────────────────────────────────
# MORAN'S I GLOBAL
# ─────────────────────────────────────────
moran = Moran(gdf['actos_per_capita'], w)
print(f"\n=== MORAN'S I GLOBAL ===")
print(f"  I  = {moran.I:.4f}")
print(f"  E[I] = {moran.EI:.4f}  (esperado bajo H0)")
print(f"  p-valor (simulación 999 permutaciones) = {moran.p_sim:.4f}")
if moran.p_sim < 0.05:
    if moran.I > 0:
        print("  → Autocorrelación espacial positiva significativa")
        print("    (departamentos con alta intensidad tienden a ser vecinos)")
    else:
        print("  → Autocorrelación espacial negativa significativa")
else:
    print("  → No se rechaza H0: distribución espacialmente aleatoria")

# ─────────────────────────────────────────
# MORAN SCATTERPLOT
# ─────────────────────────────────────────
y     = gdf['actos_per_capita'].values
y_std = (y - y.mean()) / y.std()
wy    = libpysal.weights.lag_spatial(w, y_std)

fig, ax = plt.subplots(figsize=(8, 7))
fig.patch.set_facecolor('#0F1117')
ax.set_facecolor('#0F1117')

ax.scatter(y_std, wy, c='#4F8EF7', s=80, alpha=0.85, zorder=3)

# Línea de regresión (pendiente ≈ I de Moran)
m, b = np.polyfit(y_std, wy, 1)
x_line = np.linspace(y_std.min(), y_std.max(), 100)
ax.plot(x_line, m * x_line + b, color='#E8A23A',
        linewidth=1.8, label=f"I = {moran.I:.3f}  p = {moran.p_sim:.3f}")

ax.axhline(0, color='#2A3147', linewidth=0.8)
ax.axvline(0, color='#2A3147', linewidth=0.8)

for i, row in gdf.iterrows():
    ys = y_std[i]
    ws = wy[i]
    if abs(ys) > 1.5 or abs(ws) > 1.0:
        nombre = row['Departamento'].split(',')[0][:12]
        ax.annotate(nombre, (ys, ws), textcoords='offset points',
                    xytext=(5, 3), fontsize=7.5, color='#9AA3BC')

ax.set_xlabel('Actos per cápita (estandarizado)',
              color='#9AA3BC', fontsize=10)
ax.set_ylabel('Lag espacial (promedio vecinos)',
              color='#9AA3BC', fontsize=10)
ax.set_title("Diagrama de dispersión de Moran\nIntensidad notarial — Colombia 2025",
             color='#E8EBF4', fontsize=13, fontweight='bold', pad=14)
ax.tick_params(colors='#9AA3BC', labelsize=9)
ax.spines[['top', 'right']].set_visible(False)
ax.spines[['bottom', 'left']].set_color('#2A3147')
ax.grid(color='#2A3147', linewidth=0.5, alpha=0.4)
ax.legend(facecolor='#181C26', edgecolor='#2A3147',
          labelcolor='#9AA3BC', fontsize=9)

plt.tight_layout()
plt.savefig('imagenes/moran_scatterplot.png', dpi=150,
            bbox_inches='tight', facecolor='#0F1117')
plt.close()
print("\n✓ imagenes/moran_scatterplot.png")

# ─────────────────────────────────────────
# LISA — Indicadores Locales de Asociación Espacial
# ─────────────────────────────────────────
lisa = Moran_Local(gdf['actos_per_capita'], w, seed=42)
gdf['lisa_q']   = lisa.q     # 1=HH, 2=LH, 3=LL, 4=HL
gdf['lisa_p']   = lisa.p_sim
gdf['lisa_sig'] = lisa.p_sim < 0.05

print("\n=== CLUSTERS LISA (p < 0.05) ===")
etiquetas = {1: 'Alto-Alto (HH)', 2: 'Bajo-Alto (LH)',
             3: 'Bajo-Bajo (LL)', 4: 'Alto-Bajo (HL)'}
for q, label in etiquetas.items():
    deps = gdf[(gdf['lisa_q'] == q) & gdf['lisa_sig']]['Departamento'].tolist()
    if deps:
        print(f"  {label}: {', '.join(deps)}")

# Mapa LISA
colores_lisa = {1: '#E85A5A', 2: '#F4A5A5', 3: '#4F8EF7', 4: '#A5C5F4', 0: '#2A3147'}

def color_lisa(row):
    if not row['lisa_sig']:
        return colores_lisa[0]
    return colores_lisa.get(row['lisa_q'], colores_lisa[0])

gdf['color_lisa'] = gdf.apply(color_lisa, axis=1)

fig, ax = plt.subplots(figsize=(9, 11))
fig.patch.set_facecolor('#0F1117')
ax.set_facecolor('#0F1117')

for color in gdf['color_lisa'].unique():
    subset = gdf[gdf['color_lisa'] == color]
    subset.plot(ax=ax, color=color, edgecolor='#0F1117', linewidth=0.4)

leyenda = [
    mpatches.Patch(color='#E85A5A', label='Alto-Alto (HH) — sig.'),
    mpatches.Patch(color='#4F8EF7', label='Bajo-Bajo (LL) — sig.'),
    mpatches.Patch(color='#F4A5A5', label='Bajo-Alto (LH) — sig.'),
    mpatches.Patch(color='#A5C5F4', label='Alto-Bajo (HL) — sig.'),
    mpatches.Patch(color='#2A3147', label='No significativo'),
]
ax.legend(handles=leyenda, loc='lower left',
          facecolor='#181C26', edgecolor='#2A3147',
          labelcolor='#9AA3BC', fontsize=9)

ax.set_title("Clusters LISA — Intensidad notarial Colombia 2025\n"
             f"Moran's I = {moran.I:.3f}  (p = {moran.p_sim:.3f})",
             color='#E8EBF4', fontsize=12, fontweight='bold', pad=14)
ax.axis('off')
fig.patch.set_facecolor('#0F1117')

plt.tight_layout()
plt.savefig('imagenes/mapa_lisa.png', dpi=150,
            bbox_inches='tight', facecolor='#0F1117')
plt.close()
print("✓ imagenes/mapa_lisa.png")

# ─────────────────────────────────────────
# EXPORTAR RESULTADOS
# ─────────────────────────────────────────
gdf[['Departamento', 'actos_per_capita', 'lisa_q',
     'lisa_p', 'lisa_sig']].to_csv(
    '06_Consolidado/analisis_espacial.csv',
    index=False, encoding='utf-8-sig')
print("✓ 06_Consolidado/analisis_espacial.csv")
print(f"\nResumen: I={moran.I:.4f}, p={moran.p_sim:.4f}")
