import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import (
    variance_inflation_factor, OLSInfluence)

# ─────────────────────────────────────────
# CARGAR DATOS
# ─────────────────────────────────────────
df = pd.read_csv('06_Consolidado/indice_acceso_notarial_2025.csv')
df = df.dropna(subset=['actos_per_capita', 'IPM', 'PIB_per_capita']).copy()

df['PIB_pc_millones'] = df['PIB_per_capita'] / 1e6

# ─────────────────────────────────────────
# MODELOS 1, 2, 3 — base
# ─────────────────────────────────────────
X1 = sm.add_constant(df['IPM'])
modelo1 = sm.OLS(df['actos_per_capita'], X1).fit()

X2 = sm.add_constant(df['PIB_pc_millones'])
modelo2 = sm.OLS(df['actos_per_capita'], X2).fit()

X3 = sm.add_constant(df[['IPM', 'PIB_pc_millones']])
modelo3 = sm.OLS(df['actos_per_capita'], X3).fit()

# ─────────────────────────────────────────
# MODELO 4 — con % compraventas (composición de actos)
# ─────────────────────────────────────────
if 'pct_inmobiliario' in df.columns:
    df4 = df.dropna(subset=['pct_inmobiliario']).copy()
    X4 = sm.add_constant(df4[['IPM', 'PIB_pc_millones', 'pct_inmobiliario']])
    modelo4 = sm.OLS(df4['actos_per_capita'], X4).fit()
    tiene_m4 = True
else:
    tiene_m4 = False
    print("⚠ pct_inmobiliario no encontrado — ejecuta cruzar_datos.py primero")

# ─────────────────────────────────────────
# RESULTADOS
# ─────────────────────────────────────────
print("=" * 70)
print("MODELO 3 — COMPLETO: actos_per_capita ~ IPM + PIB_per_capita")
print("=" * 70)
print(modelo3.summary())

if tiene_m4:
    print("\n" + "=" * 70)
    print("MODELO 4 — con composición: ~ IPM + PIB + pct_inmobiliario")
    print("=" * 70)
    print(modelo4.summary())

# ─────────────────────────────────────────
# MODELO 5 — log-lineal: ln(actos_pc) ~ IPM + ln(PIB_pc)
# ─────────────────────────────────────────
df5 = df.dropna(subset=['actos_per_capita', 'IPM', 'PIB_per_capita']).copy()
df5 = df5[df5['actos_per_capita'] > 0].copy()
df5['ln_actos_pc'] = np.log(df5['actos_per_capita'])
df5['ln_PIB_pc']   = np.log(df5['PIB_per_capita'])

X5 = sm.add_constant(df5[['IPM', 'ln_PIB_pc']])
modelo5 = sm.OLS(df5['ln_actos_pc'], X5).fit()

print("\n" + "=" * 70)
print("MODELO 5 — LOG-LINEAL: ln(actos_pc) ~ IPM + ln(PIB_pc)")
print("(especificación alternativa para verificar robustez)")
print("=" * 70)
print(modelo5.summary())

print("\n" + "=" * 70)
print("COMPARACIÓN DE MODELOS")
print("=" * 70)
print(f"{'Modelo':<45} {'R²':>8} {'R² adj':>8} {'AIC':>10}")
print(f"{'M1: Solo IPM':<45} {modelo1.rsquared:>8.4f} "
      f"{modelo1.rsquared_adj:>8.4f} {modelo1.aic:>10.2f}")
print(f"{'M2: Solo PIB per cápita':<45} {modelo2.rsquared:>8.4f} "
      f"{modelo2.rsquared_adj:>8.4f} {modelo2.aic:>10.2f}")
print(f"{'M3: IPM + PIB per cápita':<45} {modelo3.rsquared:>8.4f} "
      f"{modelo3.rsquared_adj:>8.4f} {modelo3.aic:>10.2f}")
if tiene_m4:
    print(f"{'M4: IPM + PIB + pct_inmobiliario':<45} {modelo4.rsquared:>8.4f} "
          f"{modelo4.rsquared_adj:>8.4f} {modelo4.aic:>10.2f}")
print(f"{'M5: ln(actos_pc) ~ IPM + ln(PIB_pc)':<45} {modelo5.rsquared:>8.4f} "
      f"{modelo5.rsquared_adj:>8.4f} {modelo5.aic:>10.2f}")

# ─────────────────────────────────────────
# VIF — multicolinealidad
# ─────────────────────────────────────────
X_vif = df[['IPM', 'PIB_pc_millones']].copy()
X_vif = sm.add_constant(X_vif)
print("\n=== VIF (multicolinealidad, >10 es problema) ===")
for i, col in enumerate(X_vif.columns):
    if col != 'const':
        print(f"  {col}: {variance_inflation_factor(X_vif.values, i):.2f}")

if tiene_m4:
    X_vif4 = df4[['IPM', 'PIB_pc_millones', 'pct_inmobiliario']].copy()
    X_vif4 = sm.add_constant(X_vif4)
    print("\n=== VIF Modelo 4 ===")
    for i, col in enumerate(X_vif4.columns):
        if col != 'const':
            print(f"  {col}: {variance_inflation_factor(X_vif4.values, i):.2f}")

# ─────────────────────────────────────────
# COOK'S D — observaciones influyentes
# ─────────────────────────────────────────
influence = OLSInfluence(modelo3)
df['cook_d']    = influence.cooks_distance[0]
df['leverage']  = influence.hat_matrix_diag
df['std_resid'] = influence.resid_studentized_internal

umbral_cook = 4 / len(df)
atipicos = df[df['cook_d'] > umbral_cook].sort_values('cook_d', ascending=False)

print(f"\n=== OBSERVACIONES INFLUYENTES (Cook's D > 4/n = {umbral_cook:.4f}) ===")
print(atipicos[['Departamento', 'actos_per_capita', 'cook_d',
                'leverage', 'std_resid', 'IPM', 'PIB_per_capita']].to_string())

# Gráfico Cook's D
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor('#0F1117')
ax.set_facecolor('#0F1117')

colores_barra = ['#E85A5A' if v > umbral_cook else '#4F8EF7'
                 for v in df['cook_d']]
ax.bar(range(len(df)), df['cook_d'], color=colores_barra, alpha=0.85)
ax.axhline(umbral_cook, color='#E8A23A', linewidth=1.2,
           linestyle='--', label=f'Umbral 4/n = {umbral_cook:.3f}')

for i, row in df[df['cook_d'] > umbral_cook].iterrows():
    nombre = row['Departamento'].split(',')[0][:10]
    ax.annotate(nombre, (df.index.get_loc(i), row['cook_d']),
                textcoords='offset points', xytext=(0, 5),
                fontsize=7.5, color='#E8A23A', ha='center')

ax.set_xticks(range(len(df)))
ax.set_xticklabels(
    [d.split(',')[0][:8] for d in df['Departamento']],
    rotation=70, ha='right', fontsize=6.5, color='#9AA3BC')
ax.set_ylabel("Distancia de Cook", color='#9AA3BC', fontsize=10)
ax.set_title("Observaciones influyentes — Distancia de Cook\n"
             "Modelo M3: actos per cápita ~ IPM + PIB per cápita",
             color='#E8EBF4', fontsize=12, fontweight='bold', pad=14)
ax.tick_params(colors='#9AA3BC', labelsize=8)
ax.spines[['top', 'right']].set_visible(False)
ax.spines[['bottom', 'left']].set_color('#2A3147')
ax.grid(color='#2A3147', linewidth=0.5, alpha=0.5, axis='y')
ax.legend(facecolor='#181C26', edgecolor='#2A3147',
          labelcolor='#9AA3BC', fontsize=9)

plt.tight_layout()
plt.savefig('imagenes/cook_distance.png', dpi=150,
            bbox_inches='tight', facecolor='#0F1117')
plt.close()
print("\n✓ imagenes/cook_distance.png")

# Gráfico leverage vs residuos estandarizados (bubble = Cook's D)
fig, ax = plt.subplots(figsize=(9, 7))
fig.patch.set_facecolor('#0F1117')
ax.set_facecolor('#0F1117')

sc = ax.scatter(df['leverage'], df['std_resid'],
                s=df['cook_d'] * 2000 + 30,
                c=['#E85A5A' if v > umbral_cook else '#4F8EF7'
                   for v in df['cook_d']],
                alpha=0.8, zorder=3)

ax.axhline(0,  color='#2A3147', linewidth=0.8)
ax.axhline(2,  color='#E8A23A', linewidth=0.8, linestyle='--', alpha=0.6)
ax.axhline(-2, color='#E8A23A', linewidth=0.8, linestyle='--', alpha=0.6)

for _, row in df[df['cook_d'] > umbral_cook].iterrows():
    nombre = row['Departamento'].split(',')[0][:12]
    ax.annotate(nombre, (row['leverage'], row['std_resid']),
                textcoords='offset points', xytext=(7, 3),
                fontsize=8, color='#E8A23A', fontweight='600')

ax.set_xlabel('Leverage (hat values)', color='#9AA3BC', fontsize=10)
ax.set_ylabel('Residuos estandarizados', color='#9AA3BC', fontsize=10)
ax.set_title("Leverage vs residuos — diagnóstico de influencia\n"
             "Tamaño burbuja proporcional a Cook's D",
             color='#E8EBF4', fontsize=12, fontweight='bold', pad=14)
ax.tick_params(colors='#9AA3BC', labelsize=9)
ax.spines[['top', 'right']].set_visible(False)
ax.spines[['bottom', 'left']].set_color('#2A3147')
ax.grid(color='#2A3147', linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.savefig('imagenes/leverage_residuos.png', dpi=150,
            bbox_inches='tight', facecolor='#0F1117')
plt.close()
print("✓ imagenes/leverage_residuos.png")

# ─────────────────────────────────────────
# GRÁFICO: observado vs predicho (Modelo 4)
# ─────────────────────────────────────────
modelo_final = modelo4 if tiene_m4 else modelo3
X_final = X4 if tiene_m4 else X3
r2_final = modelo_final.rsquared

df_plot = df4.copy() if tiene_m4 else df.copy()
df_plot['predicho'] = modelo_final.predict(X_final)

# Cook's D del modelo final (puede diferir del M3)
inf_final = OLSInfluence(modelo_final)
df_plot['cook_d_final'] = inf_final.cooks_distance[0]

fig, ax = plt.subplots(figsize=(9, 7))
fig.patch.set_facecolor('#0F1117')
ax.set_facecolor('#0F1117')

colores_punto = ['#E85A5A' if v > umbral_cook else '#4F8EF7'
                 for v in df_plot['cook_d_final']]
ax.scatter(df_plot['predicho'], df_plot['actos_per_capita'],
           c=colores_punto, s=90, alpha=0.85, zorder=3)

lims = [0, max(df_plot['actos_per_capita'].max(),
               df_plot['predicho'].max()) + 0.2]
ax.plot(lims, lims, color='#9AA3BC', linewidth=1,
        linestyle='--', alpha=0.7, label='Predicción perfecta')

for _, row in df_plot.iterrows():
    residuo = abs(row['actos_per_capita'] - row['predicho'])
    if residuo > 0.30 or row['cook_d_final'] > umbral_cook:
        nombre = row['Departamento'].split(',')[0]
        ax.annotate(nombre, (row['predicho'], row['actos_per_capita']),
                    textcoords='offset points', xytext=(6, 3),
                    fontsize=8, color='#E8A23A', fontweight='600')

ax.set_xlabel('Actos per cápita predichos por el modelo',
              color='#9AA3BC', fontsize=10)
ax.set_ylabel('Actos per cápita observados',
              color='#9AA3BC', fontsize=10)
modelo_label = 'M4 (IPM + PIB + pct_inmobiliario)' if tiene_m4 else 'M3 (IPM + PIB)'
ax.set_title(f'Regresión múltiple — observado vs predicho\n'
             f'R² = {r2_final:.3f} | {modelo_label} | Rojos: Cook\'s D > 4/n',
             color='#E8EBF4', fontsize=12, fontweight='bold', pad=16)
ax.tick_params(colors='#9AA3BC', labelsize=9)
ax.spines[['top', 'right']].set_visible(False)
ax.spines[['bottom', 'left']].set_color('#2A3147')
ax.grid(color='#2A3147', linewidth=0.5, alpha=0.5)
ax.legend(facecolor='#181C26', edgecolor='#2A3147',
          labelcolor='#9AA3BC', fontsize=9)

plt.tight_layout()
plt.savefig('imagenes/regresion_observado_predicho.png',
            dpi=150, bbox_inches='tight', facecolor='#0F1117')
plt.close()
print("✓ imagenes/regresion_observado_predicho.png")

# ─────────────────────────────────────────
# DIAGNÓSTICO DE RESIDUOS — supuestos OLS
# ─────────────────────────────────────────
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan

residuos = modelo_final.resid
print("\n" + "=" * 70)
print("DIAGNÓSTICO DE RESIDUOS — Modelo final")
print("=" * 70)

# Shapiro-Wilk (cautela: n=33 le da poca potencia)
sw_stat, sw_p = stats.shapiro(residuos)
print(f"\nShapiro-Wilk: W={sw_stat:.4f}, p={sw_p:.4f}")
if sw_p > 0.05:
    print("  → No se rechaza normalidad (p > 0.05)")
else:
    print("  → Se rechaza normalidad al 5% — interpretar con cautela (n=33)")

# Breusch-Pagan (homocedasticidad)
X_bp = X_final
bp_lm, bp_p, bp_f, bp_fp = het_breuschpagan(residuos, X_bp)
print(f"\nBreusch-Pagan: LM={bp_lm:.4f}, p={bp_p:.4f}")
if bp_p > 0.05:
    print("  → No se detecta heterocedasticidad significativa (p > 0.05)")
else:
    print("  → Posible heterocedasticidad (p < 0.05) — considerar errores robustos")

# Durbin-Watson (ya en el summary, pero lo reportamos explícitamente)
from statsmodels.stats.stattools import durbin_watson
dw = durbin_watson(residuos)
print(f"\nDurbin-Watson: {dw:.4f}  (2.0 = sin autocorrelación)")

# ── Gráfico: QQ-plot + histograma (figura de 2 paneles) ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
fig.patch.set_facecolor('#0F1117')
for ax in (ax1, ax2):
    ax.set_facecolor('#0F1117')

# QQ-plot
(osm, osr), (slope, intercept, r) = stats.probplot(residuos, dist='norm')
ax1.scatter(osm, osr, c='#4F8EF7', s=60, alpha=0.85, zorder=3)
x_line = np.array([osm.min(), osm.max()])
ax1.plot(x_line, slope * x_line + intercept,
         color='#E8A23A', linewidth=1.5, linestyle='--')
ax1.set_xlabel('Cuantiles teóricos (Normal)', color='#9AA3BC', fontsize=9)
ax1.set_ylabel('Cuantiles observados', color='#9AA3BC', fontsize=9)
ax1.set_title(f'QQ-plot — residuos del modelo\nShapiro-Wilk p = {sw_p:.3f}',
              color='#E8EBF4', fontsize=11, fontweight='bold', pad=12)
ax1.tick_params(colors='#9AA3BC', labelsize=8)
ax1.spines[['top', 'right']].set_visible(False)
ax1.spines[['bottom', 'left']].set_color('#2A3147')
ax1.grid(color='#2A3147', linewidth=0.5, alpha=0.5)

# Histograma
ax2.hist(residuos, bins=10, color='#4F8EF7', alpha=0.8, edgecolor='#0F1117')
x_norm = np.linspace(residuos.min(), residuos.max(), 200)
ax2.plot(x_norm,
         stats.norm.pdf(x_norm, residuos.mean(), residuos.std()) * len(residuos)
         * (residuos.max() - residuos.min()) / 10,
         color='#E8A23A', linewidth=1.8, linestyle='--', label='Normal teórica')
ax2.set_xlabel('Residuos', color='#9AA3BC', fontsize=9)
ax2.set_ylabel('Frecuencia', color='#9AA3BC', fontsize=9)
ax2.set_title(f'Distribución de residuos\nBreusch-Pagan p = {bp_p:.3f}',
              color='#E8EBF4', fontsize=11, fontweight='bold', pad=12)
ax2.tick_params(colors='#9AA3BC', labelsize=8)
ax2.spines[['top', 'right']].set_visible(False)
ax2.spines[['bottom', 'left']].set_color('#2A3147')
ax2.grid(color='#2A3147', linewidth=0.5, alpha=0.5, axis='y')
ax2.legend(facecolor='#181C26', edgecolor='#2A3147',
           labelcolor='#9AA3BC', fontsize=8)

plt.tight_layout()
plt.savefig('imagenes/diagnostico_residuos.png', dpi=150,
            bbox_inches='tight', facecolor='#0F1117')
plt.close()
print("✓ imagenes/diagnostico_residuos.png")

# Exportar diagnóstico
df[['Departamento', 'actos_per_capita', 'predicho',
    'cook_d', 'leverage', 'std_resid']].to_csv(
    '06_Consolidado/diagnostico_influencia.csv',
    index=False, encoding='utf-8-sig')
print("✓ 06_Consolidado/diagnostico_influencia.csv")
