import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import os

# ─────────────────────────────────────────
# CARGAR DATOS
# ─────────────────────────────────────────
df = pd.read_csv('06_Consolidado/indice_acceso_notarial_2025.csv')

os.makedirs('imagenes', exist_ok=True)

# Paleta por clasificación
colores = {
    'Sobre-servido':  '#4F8EF7',
    'Sobre promedio': '#3ABFA0',
    'Promedio':       '#E8A23A',
    'Sub-servido':    '#E85A8A',
    'Brecha crítica': '#E85A5A'
}
df['color'] = df['clasificacion'].map(colores)

# ─────────────────────────────────────────
# GRÁFICO 1 — Ranking per cápita
# ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 11))
fig.patch.set_facecolor('#0F1117')
ax.set_facecolor('#0F1117')

# Acortar nombre largo
df_plot = df.copy()
df_plot['Departamento'] = df_plot['Departamento'].str.replace(
    'Archipiélago De San Andrés, Providencia Y Santa Catalina',
    'San Andrés')

bars = ax.barh(
    df_plot['Departamento'],
    df_plot['actos_per_capita'],
    color=df_plot['color'],
    height=0.7,
    edgecolor='none'
)

# Línea promedio nacional
promedio = df_plot['actos_per_capita'].mean()
ax.axvline(promedio, color='#9AA3BC', linewidth=1,
           linestyle='--', alpha=0.7)
ax.text(promedio + 0.02, -1.2, f'Promedio\n{promedio:.2f}',
        color='#9AA3BC', fontsize=8, va='top')

# Valores en las barras
for bar, val in zip(bars, df_plot['actos_per_capita']):
    ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
            f'{val:.2f}', va='center', ha='left',
            color='#E8EBF4', fontsize=7.5)

# Estilo
ax.set_xlabel('Actos notariales per cápita', color='#9AA3BC', fontsize=10)
ax.set_title('Índice de acceso notarial per cápita\nColombia 2025',
             color='#E8EBF4', fontsize=13, fontweight='bold', pad=16)
ax.tick_params(colors='#9AA3BC', labelsize=8.5)
ax.spines[['top','right','bottom','left']].set_visible(False)
ax.set_xlim(0, df_plot['actos_per_capita'].max() + 0.25)
ax.invert_yaxis()

# Leyenda
leyenda = [mpatches.Patch(color=v, label=k)
           for k, v in colores.items()]
ax.legend(handles=leyenda, loc='lower right',
          facecolor='#181C26', edgecolor='#2A3147',
          labelcolor='#9AA3BC', fontsize=8)

plt.tight_layout()
plt.savefig('imagenes/ranking_per_capita.png',
            dpi=150, bbox_inches='tight',
            facecolor='#0F1117')
plt.close()
print("✓ imagenes/ranking_per_capita.png")

# ─────────────────────────────────────────
# GRÁFICO 2 — Scatter IPM vs per cápita
# ─────────────────────────────────────────
df_scatter = df.dropna(subset=['IPM', 'actos_per_capita'])

fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor('#0F1117')
ax.set_facecolor('#0F1117')

ax.scatter(
    df_scatter['IPM'],
    df_scatter['actos_per_capita'],
    c=df_scatter['color'],
    s=90, alpha=0.9, edgecolors='none', zorder=3
)

# Etiquetas de departamentos
for _, row in df_scatter.iterrows():
    nombre = row['Departamento'].split(',')[0]
    if len(nombre) > 12:
        nombre = nombre[:12] + '.'
    ax.annotate(nombre,
                (row['IPM'], row['actos_per_capita']),
                textcoords='offset points', xytext=(6, 3),
                fontsize=7, color='#9AA3BC')

# Línea de tendencia
z = np.polyfit(df_scatter['IPM'],
               df_scatter['actos_per_capita'], 1)
p = np.poly1d(z)
x_line = pd.Series(sorted(df_scatter['IPM']))
ax.plot(x_line, p(x_line), color='#4F8EF7',
        linewidth=1.2, linestyle='--', alpha=0.6)

# Estilo
ax.set_xlabel('Índice de Pobreza Multidimensional — IPM (%)',
              color='#9AA3BC', fontsize=10)
ax.set_ylabel('Actos notariales per cápita',
              color='#9AA3BC', fontsize=10)
ax.set_title('Relación entre pobreza (IPM) y acceso notarial\nColombia 2025',
             color='#E8EBF4', fontsize=13, fontweight='bold', pad=16)
ax.tick_params(colors='#9AA3BC', labelsize=9)
ax.spines[['top','right']].set_visible(False)
ax.spines[['bottom','left']].set_color('#2A3147')
ax.grid(axis='both', color='#2A3147', linewidth=0.5, alpha=0.5)

leyenda = [mpatches.Patch(color=v, label=k)
           for k, v in colores.items()]
ax.legend(handles=leyenda, loc='upper right',
          facecolor='#181C26', edgecolor='#2A3147',
          labelcolor='#9AA3BC', fontsize=8)

plt.tight_layout()
plt.savefig('imagenes/scatter_ipm_percapita.png',
            dpi=150, bbox_inches='tight',
            facecolor='#0F1117')
plt.close()
print("✓ imagenes/scatter_ipm_percapita.png")

# ─────────────────────────────────────────
# GRÁFICO 3 — Distribución por clasificación
# ─────────────────────────────────────────
conteo = df['clasificacion'].value_counts().reindex([
    'Sobre-servido', 'Sobre promedio', 'Promedio',
    'Sub-servido', 'Brecha crítica'
])

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor('#0F1117')
ax.set_facecolor('#0F1117')

bars = ax.bar(
    conteo.index,
    conteo.values,
    color=[colores[c] for c in conteo.index],
    width=0.6, edgecolor='none'
)

for bar, val in zip(bars, conteo.values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.1,
            str(val), ha='center', va='bottom',
            color='#E8EBF4', fontsize=11, fontweight='600')

ax.set_ylabel('Número de departamentos',
              color='#9AA3BC', fontsize=10)
ax.set_title('Distribución de departamentos por nivel de acceso notarial\nColombia 2025',
             color='#E8EBF4', fontsize=13, fontweight='bold', pad=16)
ax.tick_params(colors='#9AA3BC', labelsize=9)
ax.spines[['top','right','left']].set_visible(False)
ax.spines['bottom'].set_color('#2A3147')
ax.set_ylim(0, conteo.max() + 2)
ax.yaxis.set_visible(False)

plt.tight_layout()
plt.savefig('imagenes/distribucion_clasificacion.png',
            dpi=150, bbox_inches='tight',
            facecolor='#0F1117')
plt.close()
print("✓ imagenes/distribucion_clasificacion.png")

print("\nTodos los gráficos generados en imagenes/")