import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# 1. CARGAR SNR COMPLETO (todos los años)
# ─────────────────────────────────────────
snr = pd.read_csv(
    '01_SNR/fuentes/snr_actos/Actos_Jurídicos_Notariales_20260609.csv',
    encoding='latin-1', low_memory=False)
snr['Departamento'] = snr['Departamento'].str.encode(
    'latin-1').str.decode('utf-8', errors='ignore')
snr['Fecha'] = pd.to_datetime(snr['Fecha'], format='mixed', errors='coerce')
snr['Año'] = snr['Fecha'].dt.year

cols_excluir = [c for c in snr.columns
                if any(x in c for x in ['Fecha', 'Departamento', 'digo', 'Año'])]
cols_num = snr.columns.drop(cols_excluir + ['Año'], errors='ignore')
cols_num = [c for c in cols_num if c not in ['Año']]
for c in cols_num:
    snr[c] = pd.to_numeric(
        snr[c].astype(str).str.replace(',', ''), errors='coerce')
snr[cols_num] = snr[cols_num].fillna(0)

# Total de actos por departamento y año (2021-2025, años completos)
snr_anual = snr[snr['Año'].between(2021, 2025)].groupby(
    ['Departamento', 'Año'])[cols_num].sum().reset_index()
snr_anual['total_actos'] = snr_anual[cols_num].sum(axis=1)

# ─────────────────────────────────────────
# 2. POBLACIÓN POR AÑO
# ─────────────────────────────────────────
pob_raw = pd.read_excel(
    '01_SNR/fuentes/dane_poblacion/poblacion_departamentos.xlsx',
    sheet_name='PobDepartamentalxÁrea', header=7)
pob_raw.columns = ['DP', 'Departamento', 'Año', 'Area', 'Poblacion']
pob = pob_raw[pob_raw['Area'] == 'Total'][
    ['Departamento', 'Año', 'Poblacion']].copy()
pob['Departamento'] = pob['Departamento'].str.strip()
pob['Año'] = pob['Año'].astype(int)
pob['Poblacion'] = pd.to_numeric(pob['Poblacion'], errors='coerce')
pob = pob[pob['Año'].between(2021, 2025)]

# Normalizar nombres
nombre_map = {
    'Bogotá D.C.': 'Bogotá, D.C.',
    'Bogotá D. C.': 'Bogotá, D.C.',
    'Norte de Santander': 'Norte De Santander',
    'Valle del Cauca': 'Valle Del Cauca',
    'Archipiélago de San Andrés, Providencia y Santa Catalina':
        'Archipiélago De San Andrés, Providencia Y Santa Catalina',
}
for d in [snr_anual, pob]:
    d['Departamento'] = d['Departamento'].str.strip().replace(nombre_map)

# ─────────────────────────────────────────
# 3. PANEL: índice per cápita por año
# ─────────────────────────────────────────
panel = snr_anual[['Departamento', 'Año', 'total_actos']].merge(
    pob, on=['Departamento', 'Año'], how='inner')
panel['actos_per_capita'] = panel['total_actos'] / panel['Poblacion']

# Desviación respecto al promedio de CADA año
panel['promedio_año'] = panel.groupby('Año')['actos_per_capita'].transform('mean')
panel['desviacion_pct'] = ((panel['actos_per_capita'] - panel['promedio_año'])
                            / panel['promedio_año'] * 100).round(2)

panel.to_csv('06_Consolidado/panel_2021_2025.csv',
             index=False, encoding='utf-8-sig')

# ─────────────────────────────────────────
# 4. DISPERSIÓN POR AÑO — ¿crecen las diferencias?
# ─────────────────────────────────────────
print("=== DISPERSIÓN ENTRE DEPARTAMENTOS POR AÑO ===")
print(f"{'Año':>6} {'Media':>8} {'Desv.Est':>10} {'Coef.Var':>10} {'Max/Min':>9}")
for año in sorted(panel['Año'].unique()):
    sub = panel[panel['Año'] == año]['actos_per_capita']
    cv = sub.std() / sub.mean()
    ratio = sub.max() / sub.min()
    print(f"{año:>6} {sub.mean():>8.4f} {sub.std():>10.4f} {cv:>10.4f} {ratio:>9.2f}")

# ─────────────────────────────────────────
# 5. EVOLUCIÓN DE CASOS CLAVE
# ─────────────────────────────────────────
casos = ['Bogotá, D.C.', 'La Guajira', 'Amazonas',
         'Guaviare', 'Guainía', 'Antioquia']
colores = ['#4F8EF7', '#E85A5A', '#E8A23A',
           '#3ABFA0', '#B07FE8', '#7FA8E8']

fig, ax = plt.subplots(figsize=(10, 6.5))
fig.patch.set_facecolor('#0F1117')
ax.set_facecolor('#0F1117')

# Promedio nacional
prom = panel.groupby('Año')['actos_per_capita'].mean()
ax.plot(prom.index, prom.values, color='#9AA3BC',
        linewidth=1.5, linestyle='--', label='Promedio nacional', zorder=2)

for caso, color in zip(casos, colores):
    sub = panel[panel['Departamento'] == caso].sort_values('Año')
    ax.plot(sub['Año'], sub['actos_per_capita'],
            marker='o', color=color, linewidth=2,
            markersize=5, label=caso.split(',')[0], zorder=3)

ax.set_xlabel('Año', color='#9AA3BC', fontsize=10)
ax.set_ylabel('Actos notariales per cápita', color='#9AA3BC', fontsize=10)
ax.set_title('Evolución de la utilización notarial per cápita\nCasos seleccionados, 2021–2025',
             color='#E8EBF4', fontsize=13, fontweight='bold', pad=16)
ax.tick_params(colors='#9AA3BC', labelsize=9)
ax.set_xticks(sorted(panel['Año'].unique()))
ax.spines[['top','right']].set_visible(False)
ax.spines[['bottom','left']].set_color('#2A3147')
ax.grid(color='#2A3147', linewidth=0.5, alpha=0.5)
ax.legend(facecolor='#181C26', edgecolor='#2A3147',
          labelcolor='#9AA3BC', fontsize=9, ncol=2)

plt.tight_layout()
plt.savefig('imagenes/evolucion_panel.png', dpi=150,
            bbox_inches='tight', facecolor='#0F1117')
plt.close()
print("\n✓ imagenes/evolucion_panel.png")

# ─────────────────────────────────────────
# 6. CAMBIO 2021 → 2025 POR DEPARTAMENTO
# ─────────────────────────────────────────
p21 = panel[panel['Año']==2021].set_index('Departamento')['actos_per_capita']
p25 = panel[panel['Año']==2025].set_index('Departamento')['actos_per_capita']
cambio = ((p25 - p21) / p21 * 100).round(2).sort_values(ascending=False)

print("\n=== CAMBIO % EN UTILIZACIÓN PER CÁPITA 2021 → 2025 ===")
print("Top 5 mayores aumentos:")
print(cambio.head(5).to_string())
print("\nTop 5 mayores caídas:")
print(cambio.tail(5).to_string())
print("\n✓ 06_Consolidado/panel_2021_2025.csv")