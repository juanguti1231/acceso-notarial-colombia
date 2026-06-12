"""
dashboard.py
Construye el dashboard interactivo del análisis notarial colombiano.
Lee desde SQLite (almacén central), genera visualizaciones con Plotly,
y exporta un único archivo HTML autocontenido listo para GitHub Pages.
"""

import sqlite3
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import json
import html as html_lib
from pathlib import Path

# ---------- Configuración ----------
BASE = Path(__file__).resolve().parent.parent.parent
DB = BASE / "06_Consolidado" / "acceso_notarial_colombia.db"
OUT_DIR = BASE / "07_Dashboard"
OUT_HTML = OUT_DIR / "index.html"
GEOJSON_PATH = OUT_DIR / "colombia_departamentos.geojson"

# ---------- 1. Cargar datos desde SQLite ----------
con = sqlite3.connect(DB)
indice = pd.read_sql_query("SELECT * FROM indice_con_notarias", con)
panel = pd.read_sql_query("SELECT * FROM panel_temporal", con)
con.close()

# Formatear código DANE como string de 2 dígitos para cruzar con GeoJSON
indice["DPTO"] = indice["codigo_dane_dpto"].apply(lambda x: f"{int(x):02d}")

# ---------- 2. Cargar GeoJSON ----------
with open(GEOJSON_PATH, encoding="utf-8") as f:
    geojson = json.load(f)

# ===== DIAGNÓSTICO =====
print("\n===== DIAGNÓSTICO DEL CRUCE =====")
geojson_dptos = sorted([f["properties"]["DPTO"] for f in geojson["features"]])
data_dptos = sorted(indice["DPTO"].tolist())

print(f"Departamentos en GeoJSON: {len(geojson_dptos)}")
print(f"Departamentos en datos:   {len(data_dptos)}")

faltan_en_datos = set(geojson_dptos) - set(data_dptos)
faltan_en_geojson = set(data_dptos) - set(geojson_dptos)

if faltan_en_datos:
    print(f"\nEn GeoJSON pero NO en datos: {faltan_en_datos}")
if faltan_en_geojson:
    print(f"\nEn datos pero NO en GeoJSON: {faltan_en_geojson}")
if not faltan_en_datos and not faltan_en_geojson:
    print("\n✓ Coincidencia perfecta de códigos DANE")

print("\nNulos en columnas del hover:")
for col in ["actos_per_capita", "numero_notarias", "PIB_per_capita", "Departamento"]:
    nulos = indice[col].isna().sum()
    print(f"  {col:25s}: {nulos} nulos")
    if nulos > 0:
        nulos_df = indice[indice[col].isna()][["Departamento", "codigo_dane_dpto", "DPTO"]]
        print(nulos_df.to_string(index=False))

print("================================\n")
# ===== FIN DIAGNÓSTICO =====

# ---------- 3. Visual 1: Mapa coroplético ----------
import pandas as pd
indice_map = indice.copy()
indice_map["PIB_str"] = indice_map["PIB_per_capita"].apply(
    lambda x: f"${int(x):,} COP" if pd.notna(x) else "N/D"
)
indice_map["notarias_int"] = indice_map["numero_notarias"].astype(int)
indice_map["clasif"] = indice_map["clasificacion"].fillna("N/D")

fig_mapa = px.choropleth(
    indice_map,
    geojson=geojson,
    locations="DPTO",
    featureidkey="properties.DPTO",
    color="actos_per_capita",
    color_continuous_scale="Blues",
)
fig_mapa.update_traces(
    hovertemplate=None,
    hoverinfo="skip",
)
fig_mapa.update_geos(fitbounds="locations", visible=False)
fig_mapa.update_layout(
    title="<b>Utilización notarial por departamento (Colombia, 2025)</b><br><sub>Actos jurídicos per cápita — fuente: SNR / DANE</sub>",
    margin=dict(l=0, r=0, t=80, b=0),
    height=520,
    coloraxis_colorbar=dict(title="Actos<br>per cápita"),
)

fig_mapa.add_trace(
    go.Choropleth(
        geojson=geojson,
        locations=[],
        featureidkey="properties.DPTO",
        z=[],
        colorscale=[[0, "#f59e0b"], [1, "#f59e0b"]],
        showscale=False,
        marker_line_color="#8b5e00",
        marker_line_width=2,
        hovertemplate=None,
        hoverinfo="skip",
        name="Seleccionado",
    )
)

# ---------- 4. Visual 2: Lista de departamentos ----------
indice_asc = indice.sort_values("actos_per_capita", ascending=True).copy()
indice_asc["PIB_str"] = indice_asc["PIB_per_capita"].apply(
    lambda x: f"${int(x):,} COP" if pd.notna(x) else "N/D"
)

dept_to_point = {dept: idx for idx, dept in enumerate(indice_map["Departamento"].tolist())}

dept_items = []
for _, row in indice_asc.iterrows():
    dept = row["Departamento"]
    dept_esc = html_lib.escape(dept)
    classif = html_lib.escape(str(row["clasificacion"]))
    dept_items.append(
        f"""
        <div class="dept-item"
             data-point="{dept_to_point[dept]}"
             data-dpto="{row['DPTO']}"
             data-name="{html_lib.escape(dept, quote=True)}"
             data-pos="{int(row['posicion'])}"
             data-actos="{row['actos_per_capita']:.4f}"
             data-notarias="{int(row['numero_notarias'])}"
             data-pib="{html_lib.escape(row['PIB_str'], quote=True)}"
             data-classif="{html_lib.escape(str(row['clasificacion']), quote=True)}">
          <div class="dept-top">
            <span class="dept-name">#{int(row['posicion'])} {dept_esc}</span>
            <span class="dept-value">{row['actos_per_capita']:.4f}</span>
          </div>
          <div class="dept-meta">{classif} · {int(row['numero_notarias'])} notarías</div>
        </div>
        """
    )

dept_list_html = "\n".join(dept_items)

# ---------- 5. Visual 3: Scatter PIB vs Utilización ----------
fig_pib = px.scatter(
    indice,
    x="PIB_per_capita",
    y="actos_per_capita",
    hover_name="Departamento",
    text="Departamento",
    trendline="ols",
    title="<b>PIB per cápita vs Utilización</b><br><sub>Correlación positiva fuerte — el PIB sí explica la utilización</sub>",
    labels={"PIB_per_capita": "PIB per cápita (COP)", "actos_per_capita": "Actos per cápita"},
)
fig_pib.update_traces(textposition="top center", textfont_size=9)
fig_pib.update_layout(height=550)

# ---------- 6. Visual 4: Scatter Densidad notarial vs Utilización ----------
fig_oferta = px.scatter(
    indice,
    x="notarias_per_100k",
    y="actos_per_capita",
    hover_name="Departamento",
    text="Departamento",
    trendline="ols",
    title="<b>Densidad notarial vs Utilización</b><br><sub>Sin relación significativa (r ≈ -0.19, p=0.29) — la oferta no explica</sub>",
    labels={"notarias_per_100k": "Notarías por 100k hab.", "actos_per_capita": "Actos per cápita"},
)
fig_oferta.update_traces(textposition="top center", textfont_size=9)
fig_oferta.update_layout(height=550)

# ---------- 7. Visual 5: Evolución temporal 2021-2025 ----------
top5 = indice.nlargest(5, "actos_per_capita")["Departamento"].tolist()
bot5 = indice.nsmallest(5, "actos_per_capita")["Departamento"].tolist()
seleccion = top5 + bot5
panel_sel = panel[panel["Departamento"].isin(seleccion)].sort_values(["Departamento", "Año"])

fig_temporal = px.line(
    panel_sel,
    x="Año",
    y="actos_per_capita",
    color="Departamento",
    title="<b>Evolución temporal 2021–2025</b><br><sub>Top 5 (alta utilización) vs Bottom 5 (utilización crítica)</sub>",
    markers=True,
    labels={"actos_per_capita": "Actos per cápita"},
)
fig_temporal.update_layout(height=550)

# ---------- 8. Componer HTML autocontenido ----------
def to_div(fig, include_js=False, config=None):
    return pio.to_html(fig, full_html=False, include_plotlyjs=include_js, config=config)

hover_link_js = """
<script>
window.addEventListener('load', function() {
    setTimeout(function() {
        var divs = document.querySelectorAll('.plotly-graph-div');
        if (divs.length < 1) return;
        var mapDiv = divs[0];
        var deptItems = document.querySelectorAll('.dept-item');
        var card = document.getElementById('dept-card');
        var closeBtn = document.getElementById('dept-card-close');
        if (!mapDiv || !deptItems.length) return;

        function renderCard(item) {
            if (!card || !item) return;
            var name = item.getAttribute('data-name') || '';
            var pos = item.getAttribute('data-pos') || '';
            var actos = item.getAttribute('data-actos') || '';
            var notarias = item.getAttribute('data-notarias') || '';
            var pib = item.getAttribute('data-pib') || 'N/D';
            var clasif = item.getAttribute('data-classif') || '';
            card.classList.add('is-visible');
            card.innerHTML =
                '<div class="dept-card-head">' +
                  '<div class="dept-card-title">' + name + '</div>' +
                  '<button type="button" id="dept-card-close" class="dept-card-close" aria-label="Cerrar">×</button>' +
                '</div>' +
                '<div>Posición: #' + pos + ' de 33</div>' +
                '<div>Actos per cápita: ' + actos + '</div>' +
                '<div>Notarías: ' + notarias + '</div>' +
                '<div>PIB per cápita: ' + pib + '</div>' +
                '<div>Clasificación: ' + clasif + '</div>';

            var deptCode = item.getAttribute('data-dpto') || '';
            if (deptCode) {
                Plotly.restyle(mapDiv, {
                    locations: [[deptCode]],
                    z: [[1]],
                }, [1]);
            }

            setTimeout(function() {
                var newClose = document.getElementById('dept-card-close');
                if (newClose) {
                    newClose.onclick = clearSelection;
                }
            }, 0);
        }

        function clearSelection() {
            deptItems.forEach(function(other) { other.classList.remove('is-active'); });
            if (card) {
                card.classList.remove('is-visible');
                card.innerHTML = '';
            }
            Plotly.restyle(mapDiv, {
                locations: [[]],
                z: [[]],
            }, [1]);
        }

        deptItems.forEach(function(item) {
            item.addEventListener('click', function() {
                deptItems.forEach(function(other) { other.classList.remove('is-active'); });
                item.classList.add('is-active');
                renderCard(item);
            });
        });
    }, 1500);
});
</script>
"""

html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Acceso Notarial Colombia 2025</title>
<style>
  body {{ font-family: 'Helvetica Neue', Arial, sans-serif; background: #f7f7f7; margin: 0; padding: 24px; color: #1a1a1a; }}
  .container {{ max-width: 1600px; margin: 0 auto; }}
  h1 {{ color: #1f4e79; font-size: 2em; margin-bottom: 4px; }}
  .subtitle {{ color: #6b7280; font-size: 1em; margin-bottom: 24px; }}
  .panel {{ background: white; border-radius: 8px; padding: 20px; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }}
  .grid-main {{ display: grid; grid-template-columns: 3.4fr 5.6fr; gap: 24px; margin-bottom: 24px; align-items: stretch; }}
  .grid-main > .panel {{ margin-bottom: 0; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
  .dept-list {{ max-height: 600px; overflow: auto; padding-right: 6px; }}
  .dept-item {{ border: 1px solid #e5e7eb; border-radius: 10px; padding: 12px 14px; margin-bottom: 10px; background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%); transition: transform 120ms ease, box-shadow 120ms ease, border-color 120ms ease, background 120ms ease; cursor: pointer; }}
  .dept-item:hover, .dept-item.is-active {{ border-color: #1f4e79; box-shadow: 0 8px 18px rgba(31, 78, 121, 0.12); transform: translateY(-1px); background: linear-gradient(180deg, #f7fbff 0%, #eef6ff 100%); }}
  .dept-top {{ display: flex; align-items: baseline; justify-content: space-between; gap: 12px; }}
  .dept-name {{ font-weight: 700; color: #1f2937; line-height: 1.25; }}
  .dept-value {{ font-variant-numeric: tabular-nums; color: #1f4e79; font-weight: 700; white-space: nowrap; }}
  .dept-meta {{ margin-top: 4px; color: #6b7280; font-size: 0.92em; }}
  .map-wrap {{ position: relative; }}
  .dept-card {{ position: absolute; left: 18px; bottom: 18px; z-index: 5; background: rgba(70, 70, 70, 0.96); color: white; padding: 12px 14px; border-radius: 2px; min-width: 250px; max-width: 320px; box-shadow: 0 12px 30px rgba(0,0,0,0.22); opacity: 0; transform: translateY(12px); pointer-events: none; transition: opacity 160ms ease, transform 160ms ease; }}
  .dept-card.is-visible {{ opacity: 1; transform: translateY(0); pointer-events: auto; }}
  .dept-card-head {{ display: flex; align-items: flex-start; justify-content: space-between; gap: 10px; margin-bottom: 4px; }}
  .dept-card-title {{ font-size: 1.3em; font-weight: 800; line-height: 1.05; margin: 0; }}
  .dept-card-close {{ appearance: none; border: 0; background: transparent; color: white; font-size: 1.3em; line-height: 1; cursor: pointer; padding: 0; margin: -1px 0 0; }}
  .dept-card div {{ font-size: 0.98em; line-height: 1.22; margin-top: 2px; }}
  footer {{ color: #6b7280; text-align: center; margin-top: 40px; padding: 24px; font-size: 0.9em; }}
  a {{ color: #1f4e79; }}
</style>
</head>
<body>
<div class="container">
  <h1>Acceso al servicio notarial en Colombia</h1>
  <div class="subtitle">
    Análisis de utilización por departamento — Notaría 74 del Círculo de Bogotá<br>
    Fuentes: SNR · DANE · DNP · Directorio oficial de notarías | Período: 2021–2025
  </div>

  <div class="grid-main">
    <div class="panel">
      <div class="dept-list">
        {dept_list_html}
      </div>
    </div>
    <div class="panel map-wrap">
      <div id="dept-card" class="dept-card" aria-live="polite"></div>
      {to_div(fig_mapa, include_js='cdn', config={"scrollZoom": False, "displaylogo": False, "responsive": True})}
    </div>
  </div>
  <div class="grid">
    <div class="panel">{to_div(fig_pib)}</div>
    <div class="panel">{to_div(fig_oferta)}</div>
  </div>
  <div class="panel">{to_div(fig_temporal)}</div>

  <footer>
    Dashboard generado con Python + Plotly · Fuente: SQLite (acceso_notarial_colombia.db)<br>
    Repositorio: <a href="https://github.com/juanguti1231/acceso-notarial-colombia">github.com/juanguti1231/acceso-notarial-colombia</a>
  </footer>
</div>
{hover_link_js}
</body>
</html>"""

OUT_HTML.write_text(html, encoding="utf-8")

print(f"Dashboard generado: {OUT_HTML}")
print(f"Tamaño: {OUT_HTML.stat().st_size / 1024:.1f} KB")
print(f"\nPara verlo: abre {OUT_HTML} en tu navegador.")
