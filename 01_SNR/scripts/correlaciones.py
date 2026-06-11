import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ─────────────────────────────────────────
# CARGAR DATOS
# ─────────────────────────────────────────
df = pd.read_csv('06_Consolidado/indice_acceso_notarial_2025.csv')

# Variables de interés
vars_analisis = df[['actos_per_capita', 'IPM', 
                    'PIB_per_capita', 'Poblacion',
                    'compraventas_per_capita']].dropna()

vars_analisis.columns = ['Actos per cápita', 'IPM (%)',
                          'PIB per cápita', 'Población',
                          'Compraventas per cápita']

# ─────────────────────────────────────────
# 1. CORRELACIONES PEARSON Y SPEARMAN
# ─────────────────────────────────────────
print("=" * 60)
print("CORRELACIONES CON ACTOS PER CÁPITA")
print("=" * 60)

y = vars_analisis['Actos per cápita']

for col in vars_analisis.columns[1:]:
    x = vars_analisis[col]
    r_pearson, p_pearson = stats.pearsonr(x, y)
    r_spearman, p_spearman = stats.spearmanr(x, y)
    
    sig_p = "***" if p_pearson < 0.01 else "**" if p_pearson < 0.05 else "*" if p_pearson < 0.1 else ""
    sig_s = "***" if p_spearman < 0.01 else "**" if p_spearman < 0.05 else "*" if p_spearman < 0.1 else ""
    
    print(f"\n{col}:")
    print(f"  Pearson:  r = {r_pearson:+.4f} (p = {p_pearson:.4f}) {sig_p}")
    print(f"  Spearman: ρ = {r_spearman:+.4f} (p = {p_spearman:.4f}) {sig_s}")

print("\n" + "=" * 60)
print("Significancia: *** p<0.01 | ** p<0.05 | * p<0.1")
print("=" * 60)

# ─────────────────────────────────────────
# 2. MATRIZ COMPLETA
# ─────────────────────────────────────────
corr_matrix = vars_analisis.corr(method='pearson')
print("\n=== MATRIZ DE CORRELACIONES (Pearson) ===")
print(corr_matrix.round(3).to_string())

# ─────────────────────────────────────────
# 3. HEATMAP
# ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
fig.patch.set_facecolor('#0F1117')
ax.set_facecolor('#0F1117')

mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

sns.heatmap(corr_matrix, 
            mask=mask,
            annot=True, 
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            vmin=-1, vmax=1,
            square=True,
            linewidths=1,
            linecolor='#0F1117',
            annot_kws={'fontsize': 10, 'fontweight': '600'},
            cbar_kws={'shrink': 0.8},
            ax=ax)

ax.set_title('Matriz de correlaciones — Utilización notarial\ny variables socioeconómicas, Colombia 2025',
             color='#E8EBF4', fontsize=13, fontweight='bold', pad=16)
ax.tick_params(colors='#9AA3BC', labelsize=9)
plt.setp(ax.get_xticklabels(), rotation=35, ha='right')
plt.setp(ax.get_yticklabels(), rotation=0)

cbar = ax.collections[0].colorbar
cbar.ax.tick_params(colors='#9AA3BC')

plt.tight_layout()
plt.savefig('imagenes/matriz_correlaciones.png',
            dpi=150, bbox_inches='tight', facecolor='#0F1117')
plt.close()

print("\n✓ imagenes/matriz_correlaciones.png generada")