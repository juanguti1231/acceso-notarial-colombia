# Análisis de la Utilización del Sistema Notarial Colombiano

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tectonic](https://img.shields.io/badge/LaTeX-Tectonic-orange.svg)](https://tectonic-typesetting.github.io/)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

Un análisis cuantitativo de las diferencias departamentales en la utilización
de servicios notariales en Colombia, cruzando datos abiertos de la
Superintendencia de Notariado y Registro (SNR), el DANE y el DNP. Construye
un índice de utilización per cápita para los 33 entes territoriales del país,
identifica los factores socioeconómicos asociados a las diferencias
observadas y propone recomendaciones de política pública basadas en la
evidencia consolidada.

---

## Tabla de contenido

1. [Pregunta de investigación](#pregunta-de-investigación)
2. [Hallazgos clave](#hallazgos-clave)
3. [Marco analítico de cuatro fases](#marco-analítico-de-cuatro-fases)
4. [Fuentes de datos](#fuentes-de-datos)
5. [Stack tecnológico](#stack-tecnológico)
6. [Estructura del repositorio](#estructura-del-repositorio)
7. [Cómo replicar el análisis](#cómo-replicar-el-análisis)
8. [Informe completo](#informe-completo)
9. [Limitaciones y trabajo futuro](#limitaciones-y-trabajo-futuro)
10. [Autor](#autor)
11. [Licencia](#licencia)

---

## Pregunta de investigación

> **¿Qué tan grandes son las diferencias en la utilización de servicios
> notariales entre los departamentos de Colombia y qué variables
> socioeconómicas se asocian con dichas diferencias?**

El sistema notarial colombiano opera en los 32 departamentos del país y en
Bogotá D.C., pero la evidencia pública sobre la distribución territorial de
su utilización es escasa. Este proyecto cuantifica esas diferencias con
datos oficiales abiertos y evalúa qué factores —económicos, de pobreza, de
oferta institucional— se asocian con ellas.

---

## Hallazgos clave

### 1. Brecha de utilización amplia y persistente

La razón entre el departamento de mayor utilización (Bogotá D.C., 1.57 actos
per cápita) y el de menor utilización (Vichada, 0.08) es de aproximadamente
**19 a 1**. Nueve departamentos —más de una cuarta parte del país— presentan
utilización crítica (más de 50 % por debajo del promedio nacional).

### 2. El PIB per cápita es el predictor más robusto

| Variable | Pearson | Spearman | p-value |
|---|---:|---:|---:|
| PIB per cápita | +0.70 | +0.79 | < 0.001 |
| IPM (pobreza) | −0.58 | −0.74 | < 0.001 |
| Población | +0.48 | — | < 0.01 |
| **Densidad notarial** | **−0.19** | **−0.15** | **0.29** |

La regresión OLS confirma al PIB per cápita como el único predictor
estadísticamente significativo del modelo (β = +0.018, p = 0.003).

### 3. La oferta institucional **no** explica las diferencias

Incorporar la densidad notarial (notarías por 100.000 habitantes) al modelo
de regresión **no mejora** la capacidad explicativa:

|  | Modelo A (base) | Modelo B (+ oferta) |
|---|---:|---:|
| R² | 0.514 | 0.521 |
| R² ajustado | 0.481 | 0.470 |
| AIC | 12.37 | 13.92 |
| ΔR² | — | +0.007 (p = 0.53) |

Boyacá tiene 5 veces más notarías por habitante que La Guajira, pero esto
no se traduce en mayor utilización. Bogotá D.C., con la segunda menor
densidad del país, lidera el índice nacional.

### 4. Las diferencias son estables en el tiempo

El coeficiente de variación entre departamentos oscila entre 0.72 y 0.80 en
el periodo 2021–2025, sin tendencia clara de convergencia ni de
ensanchamiento.

### 5. Saturación en territorios con notaría única

Guaviare, Amazonas y Vaupés operan con una sola notaría cada uno. Guaviare
registra 114.669 actos por notaría en 2025 (tercera carga más alta del
país, solo detrás de Bogotá D.C. y Atlántico).

---

## Marco analítico de cuatro fases

El proyecto aplica los cuatro tipos estándar del análisis de datos:

| Fase | Pregunta | Implementación |
|---|---|---|
| **Descriptiva** | ¿Qué sucedió? | Ranking departamental, participación de mercado, distribución por tipo de acto |
| **Diagnóstica** | ¿Por qué sucedió? | Correlaciones (Pearson/Spearman), K-Means (k=4), OLS multivariada, panel 2021–2025, integración de oferta institucional |
| **Predictiva** | ¿Qué podría pasar? | Propuesto a partir de la serie histórica 2021–2026 (en desarrollo) |
| **Prescriptiva** | ¿Qué debería hacerse? | Cinco recomendaciones de política pública dirigidas a la SNR y al Estado colombiano |

---

## Fuentes de datos

Las cinco fuentes son públicas y oficiales:

| Fuente | Entidad | Variables | Periodo |
|---|---|---|---|
| Actos Jurídicos Notariales | SNR (datos.gov.co) | 40 tipos de actos por depto/mes | 2021–2026 |
| Proyecciones de población | DANE | Población departamental | 2018–2050 |
| PIB departamental | DANE | PIB a precios corrientes | 2021–2024 |
| Índice de Pobreza Multidimensional | DNP | IPM por departamento | 2021–2025 |
| Directorio de Notarías | SNR | 920 notarías con código DANE municipal | Corte mayo 2024 |

Todos los archivos crudos están en `01_SNR/fuentes/` y son reproducibles
desde los enlaces documentados en el informe.

---

## Stack tecnológico

- **Python 3.11** — limpieza, análisis y modelado
  - `pandas` — manipulación de datos
  - `statsmodels` — regresión OLS, tests estadísticos
  - `scikit-learn` — K-Means clustering
  - `scipy.stats` — correlaciones, tests de hipótesis
  - `matplotlib` / `seaborn` — visualizaciones
  - `pdfplumber` — extracción del directorio de notarías desde PDF
- **SQLite** — base de datos central del proyecto
- **LaTeX (Tectonic)** — compilación del informe
- **Git + GitHub** — control de versiones

---

## Estructura del repositorio

```
analisis-de-datos-notarias/
├── 01_SNR/
│   ├── fuentes/              # Datos crudos (SNR, DANE, DNP, directorio notarías)
│   │   ├── snr_actos/
│   │   ├── dane_poblacion/
│   │   ├── dane_pib/
│   │   ├── dnp_ipm/
│   │   └── snr_notarias/
│   └── scripts/              # Pipeline de análisis (9 scripts)
│       ├── explorar_snr.py
│       ├── cruzar_datos.py
│       ├── correlaciones.py
│       ├── clustering.py
│       ├── regresion.py
│       ├── panel.py
│       ├── integrar_notarias.py
│       ├── grafica_oferta.py
│       └── visualizaciones.py
├── 06_Consolidado/           # Salidas procesadas y base SQLite
│   ├── acceso_notarial_colombia.db
│   ├── indice_acceso_notarial_2025.csv
│   ├── indice_con_notarias_2025.csv
│   ├── panel_2021_2025.csv
│   ├── clusters_departamentos.csv
│   └── ranking_2025.csv
├── docs/
│   └── glosario.html
├── imagenes/                 # Figuras del informe (PNG)
├── main.tex                  # Fuente LaTeX del informe
├── main.pdf                  # Informe compilado
├── README.md
├── LICENSE
└── .gitignore
```

---

## Cómo replicar el análisis

### Requisitos previos

- Python 3.11 (o superior compatible)
- Tectonic para compilar el LaTeX (opcional, solo si se quiere regenerar el PDF)
- Git

### 1. Clonar el repositorio

```bash
git clone https://github.com/juanguti1231/acceso-notarial-colombia.git
cd acceso-notarial-colombia
```

### 2. Instalar dependencias

```bash
pip install pandas openpyxl statsmodels scikit-learn scipy matplotlib seaborn pdfplumber
```

> Si usas macOS con Python instalado por Homebrew, añade `--break-system-packages` a los comandos `pip install`.

### 3. Ejecutar el pipeline completo

Los scripts deben correrse en el orden indicado, ya que cada uno depende de
los archivos generados por los anteriores:

```bash
python 01_SNR/scripts/explorar_snr.py        # Carga inicial del SNR y ranking
python 01_SNR/scripts/cruzar_datos.py        # Cruce con DANE/DNP, índice per cápita
python 01_SNR/scripts/correlaciones.py       # Matriz de correlaciones
python 01_SNR/scripts/clustering.py          # K-Means (k=4)
python 01_SNR/scripts/regresion.py           # OLS multivariada
python 01_SNR/scripts/panel.py               # Panel temporal 2021–2025
python 01_SNR/scripts/integrar_notarias.py   # Integra densidad notarial
python 01_SNR/scripts/grafica_oferta.py      # Scatter oferta vs utilización
python 01_SNR/scripts/visualizaciones.py     # Figuras del informe
```

### 4. Recompilar el informe (opcional)

```bash
tectonic main.tex
```

---

## Informe completo

El informe técnico (≈ 30 páginas) está disponible como
**[`main.pdf`](main.pdf)** en este repositorio. Incluye:

- Marco metodológico completo y justificación de KPIs
- Documentación detallada de las cinco fuentes
- Resultados descriptivos, diagnósticos y prescriptivos
- Tabla del índice para los 33 entes territoriales
- Análisis de correlaciones, clusters, regresión OLS y panel temporal
- Cinco recomendaciones de política pública
- Anexo técnico con replicabilidad completa

---

## Limitaciones y trabajo futuro

El análisis es correlacional y no permite afirmar causalidad. Otras
limitaciones explícitas:

- Los actos notariales miden volumen formal registrado, no acceso ni
  calidad del servicio.
- No se consideran tiempos de atención ni costos.
- El estudio trabaja con agregados departamentales, aun cuando el
  directorio de la SNR permitiría análisis a nivel municipal.
- Cerca del 49 % de la varianza en la utilización no se explica por las
  variables incluidas, lo que sugiere factores no observados (ruralidad,
  composición étnica, formalización empresarial).

Líneas naturales de extensión:

- Análisis a escala municipal usando el código DANE de cada notaría.
- Incorporación de variables de demanda potencial (formalización
  empresarial, bancarización, registro inmobiliario).
- Modelos de series de tiempo para proyectar la evolución 2026–2030.

---

## Autor

**Juan Felipe Gutiérrez**
Estudiante de Ingeniería de Sistemas
Bogotá, Colombia

- GitHub: [@juanguti1231](https://github.com/juanguti1231)

---

## Licencia

Este proyecto se distribuye bajo licencia MIT. Ver [LICENSE](LICENSE) para
los términos completos.

Los datos originales pertenecen a sus respectivas entidades (SNR, DANE,
DNP) y se utilizan bajo las condiciones de uso de datos abiertos del
Estado colombiano.

---

## Cita sugerida

Si este trabajo es de utilidad para investigación o política pública, por
favor cítelo como:

> Gutiérrez, J. F. (2026). *Análisis de la Utilización del Sistema Notarial
> Colombiano: un enfoque departamental con datos del SNR y el DANE.*
> Repositorio GitHub. https://github.com/juanguti1231/acceso-notarial-colombia