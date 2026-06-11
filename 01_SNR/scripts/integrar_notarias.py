"""
integrar_notarias.py
Integra el numero de notarias por departamento al consolidado:
  - notarias_per_100k  (densidad notarial = OFERTA estructural)
  - actos_por_notaria  (carga de trabajo / productividad por notaria)
Recalcula correlaciones y corre regresion OLS con la densidad como control.
Ejecutar desde la raiz del proyecto:  python 01_SNR/scripts/integrar_notarias.py
"""
import pandas as pd, numpy as np, re, unicodedata
from scipy import stats
import statsmodels.api as sm

BASE = "06_Consolidado"
RUTA_NOT = "01_SNR/fuentes/snr_notarias/notarias_por_departamento.csv"

def norm(s):
    """Normaliza nombre de depto: sin tildes, sin puntuacion, sin espacios, mayus."""
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode()
    return re.sub(r'[^A-Z0-9]', '', s.upper())

# ---------- 1. Cargar y cruzar ----------
idx  = pd.read_csv(f"{BASE}/indice_acceso_notarial_2025.csv")
nots = pd.read_csv(RUTA_NOT)
idx['_k']  = idx['Departamento'].map(norm)
nots['_k'] = nots['departamento'].map(norm)

ALIAS = {'ARCHIPIELAGODESANANDRESPROVIDENCIAYSANTACATALINA': 'SANANDRESYPROVIDENCIA'}
idx['_k'] = idx['_k'].replace(ALIAS)

m = idx.merge(nots[['_k', 'codigo_dane_dpto', 'numero_notarias']], on='_k', how='left')

print("="*60)
print("CONTROL DE CRUCE")
print("="*60)
ok = m['numero_notarias'].notna().sum()
print(f"Departamentos cruzados: {ok}/33")
sin = m.loc[m['numero_notarias'].isna(), 'Departamento'].tolist()
if sin:
    print("  SIN CRUCE (revisar):", sin)
    usadas = set(m['_k'])
    print("  Notarias sin usar:  ", nots.loc[~nots['_k'].isin(usadas), 'departamento'].tolist())
else:
    print("  Cruce perfecto, los 33 entes emparejados.")

# ---------- 2. Metricas nuevas ----------
m['notarias_per_100k'] = m['numero_notarias'] / m['Poblacion'] * 1e5
m['actos_por_notaria'] = m['total_actos'] / m['numero_notarias']

# ---------- 3. Guardar consolidado enriquecido ----------
cols_out = ['Departamento','codigo_dane_dpto','numero_notarias','Poblacion',
            'notarias_per_100k','actos_por_notaria','actos_per_capita',
            'PIB_per_capita','IPM','desviacion_pct','clasificacion','posicion']
m[cols_out].to_csv(f"{BASE}/indice_con_notarias_2025.csv", index=False, encoding='utf-8')
print(f"\nGuardado: {BASE}/indice_con_notarias_2025.csv")

# ---------- 4. Rankings rapidos ----------
print("\n" + "="*60); print("DENSIDAD NOTARIAL (notarias por 100k hab)"); print("="*60)
r = m.dropna(subset=['notarias_per_100k']).sort_values('notarias_per_100k', ascending=False)
print(f"{'DEPARTAMENTO':<26}{'NOT':>4}{'per100k':>9}{'act/not':>11}")
for _, x in r.iterrows():
    print(f"{x['Departamento']:<26}{int(x['numero_notarias']):>4}"
          f"{x['notarias_per_100k']:>9.3f}{x['actos_por_notaria']:>11,.0f}")

# ---------- 5. Correlaciones (excluyendo NaN por variable, no global) ----------
print("\n" + "="*60); print("CORRELACIONES con la utilizacion (actos_per_capita)"); print("="*60)
for var in ['notarias_per_100k', 'actos_por_notaria', 'PIB_per_capita', 'IPM']:
    sub = m[[var, 'actos_per_capita']].dropna()
    rp, pp = stats.pearsonr(sub[var], sub['actos_per_capita'])
    rs, ps = stats.spearmanr(sub[var], sub['actos_per_capita'])
    print(f"  {var:<20} Pearson r={rp:+.3f} (p={pp:.4f})   Spearman rho={rs:+.3f} (p={ps:.4f})   n={len(sub)}")

# correlacion densidad vs riqueza/pobreza (la oferta es estructural o economica?)
print("\n  Densidad notarial vs ...")
for var in ['PIB_per_capita', 'IPM']:
    sub2 = m[['notarias_per_100k', var]].dropna()
    rp, pp = stats.pearsonr(sub2['notarias_per_100k'], sub2[var])
    print(f"    notarias_per_100k ~ {var:<16} r={rp:+.3f} (p={pp:.4f})   n={len(sub2)}")

# ---------- 6. Regresion: aporta la oferta? (excluir solo filas sin PIB/IPM/notarias) ----------
print("\n" + "="*60); print("REGRESION OLS  (DV = actos_per_capita)"); print("="*60)
d = m.dropna(subset=['actos_per_capita','PIB_per_capita','IPM','notarias_per_100k']).copy()
d['PIB_pc_mill'] = d['PIB_per_capita'] / 1e6   # a millones para coef. legibles
print(f"  n efectivo en la regresion: {len(d)}  (se excluye {33-len(d)} ente(s) sin PIB/IPM)")

def ols(dv, ivs, df):
    X = sm.add_constant(df[ivs]); y = df[dv]
    return sm.OLS(y, X).fit()

mA = ols('actos_per_capita', ['PIB_pc_mill', 'IPM'], d)
mB = ols('actos_per_capita', ['PIB_pc_mill', 'IPM', 'notarias_per_100k'], d)

print(f"\nModelo A (base):   PIB_pc + IPM")
print(f"   R2={mA.rsquared:.3f}  R2_adj={mA.rsquared_adj:.3f}  AIC={mA.aic:.2f}")
print(f"\nModelo B (+oferta): PIB_pc + IPM + notarias_per_100k")
print(f"   R2={mB.rsquared:.3f}  R2_adj={mB.rsquared_adj:.3f}  AIC={mB.aic:.2f}")
print(f"\n   Aporte de la densidad notarial:")
print(f"     coef={mB.params['notarias_per_100k']:+.4f}  p={mB.pvalues['notarias_per_100k']:.4f}")
print(f"     Delta R2 = {mB.rsquared - mA.rsquared:+.3f}")
print("\n--- Modelo B completo ---")
print(mB.summary().tables[1])