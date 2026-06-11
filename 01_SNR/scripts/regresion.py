import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# ─────────────────────────────────────────
# CARGAR DATOS
# ─────────────────────────────────────────
df = pd.read_csv('06_Consolidado/indice_acceso_notarial_2025.csv')
df = df.dropna(subset=['actos_per_capita', 'IPM', 'PIB_per_capita']).copy()

# PIB per cápita en millones para coeficientes legibles
df['PIB_pc_millones'] = df['PIB_per_capita'] / 1e6

# ─────────────────────────────────────────
# MODELO 1 — Solo IPM
# ─────────────────────────────────────────
X1 = sm.add_constant(df['IPM'])
modelo1 = sm.OLS(df['actos_per_capita'], X1).fit()

# ─────────────────────────────────────────
# MODELO 2 — Solo PIB per cápita
# ─────────────────────────────────────────
X2 = sm.add_constant(df['PIB_pc_millones'])
modelo2 = sm.OLS(df['actos_per_capita'], X2).fit()

# ─────────────────────────────────────────
# MODELO 3 — Completo: IPM + PIB
# ─────────────────────────────────────────
X3 = sm.add_constant(df[['IPM', 'PIB_pc_millones']])
modelo3 = sm.OLS(df['actos_per_capita'], X3).fit()

# ─────────────────────────────────────────
# RESULTADOS
# ─────────────────────────────────────────
print("=" * 70)
print("MODELO 3 — COMPLETO: actos_per_capita ~ IPM + PIB_per_capita")
print("=" * 70)
print(modelo3.summary())

print("\n" + "=" * 70)
print("COMPARACIÓN DE MODELOS")
print("=" * 70)
print(f"{'Modelo':<35} {'R²':>8} {'R² adj':>8} {'AIC':>10}")
print(f"{'M1: Solo IPM':<35} {modelo1.rsquared:>8.4f} {modelo1.rsquared_adj:>8.4f} {modelo1.aic:>10.2f}")
print(f"{'M2: Solo PIB per cápita':<35} {modelo2.rsquared:>8.4f} {modelo2.rsquared_adj:>8.4f} {modelo2.aic:>10.2f}")
print(f"{'M3: IPM + PIB per cápita':<35} {modelo3.rsquared:>8.4f} {modelo3.rsquared_adj:>8.4f} {modelo3.aic:>10.2f}")

# ─────────────────────────────────────────
# DIAGNÓSTICO: multicolinealidad (VIF)
# ─────────────────────────────────────────
from statsmodels.stats.outliers_influence import variance_inflation_factor
X_vif = df[['IPM', 'PIB_pc_millones']].copy()
X_vif = sm.add_constant(X_vif)
print("\n=== VIF (multicolinealidad, >10 es problema) ===")
for i, col in enumerate(X_vif.columns):
    if col != 'const':
        print(f"  {col}: {variance_inflation_factor(X_vif.values, i):.2f}")

# ─────────────────────────────────────────
# GRÁFICO: observado vs predicho
# ─────────────────────────────────────────
df['predicho'] = modelo3.predict(X3)

fig, ax = plt.subplots(figsize=(9, 7))
fig.patch.set_facecolor('#0F1117')
ax.set_facecolor('#0F1117')

ax.scatter(df['predicho'], df['actos_per_capita'],
           c='#4F8EF7', s=90, alpha=0.85, zorder=3)

lims = [0, max(df['actos_per_capita'].max(), df['predicho'].max()) + 0.2]
ax.plot(lims, lims, color='#9AA3BC', linewidth=1,
        linestyle='--', alpha=0.7, label='Predicción perfecta')

for _, row in df.iterrows():
    residuo = abs(row['actos_per_capita'] - row['predicho'])
    if residuo > 0.35:  # etiquetar solo los atípicos
        nombre = row['Departamento'].split(',')[0]
        ax.annotate(nombre, (row['predicho'], row['actos_per_capita']),
                    textcoords='offset points', xytext=(6, 3),
                    fontsize=8, color='#E8A23A', fontweight='600')

ax.set_xlabel('Actos per cápita predichos por el modelo',
              color='#9AA3BC', fontsize=10)
ax.set_ylabel('Actos per cápita observados',
              color='#9AA3BC', fontsize=10)
ax.set_title(f'Regresión múltiple — observado vs predicho\nR² = {modelo3.rsquared:.3f} | Variables: IPM + PIB per cápita',
             color='#E8EBF4', fontsize=13, fontweight='bold', pad=16)
ax.tick_params(colors='#9AA3BC', labelsize=9)
ax.spines[['top','right']].set_visible(False)
ax.spines[['bottom','left']].set_color('#2A3147')
ax.grid(color='#2A3147', linewidth=0.5, alpha=0.5)
ax.legend(facecolor='#181C26', edgecolor='#2A3147',
          labelcolor='#9AA3BC', fontsize=9)

plt.tight_layout()
plt.savefig('imagenes/regresion_observado_predicho.png',
            dpi=150, bbox_inches='tight', facecolor='#0F1117')
plt.close()
print("\n✓ imagenes/regresion_observado_predicho.png")