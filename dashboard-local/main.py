from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
from datetime import date
import random

app = FastAPI(title="Demo local – Muertes evitables")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

#Generamos datos sintéticos, los tableros no van a funcionar con datos reales por ahiora
random.seed(42)
np.random.seed(42)

DEPARTAMENTOS = [
    "Antioquia","Atlántico","Bogotá D.C.","Bolívar","Boyacá","Caldas","Caquetá","Cauca",
    "Cesar","Córdoba","Cundinamarca","Chocó","Huila","La Guajira","Magdalena","Meta",
    "Nariño","Norte de Santander","Quindío","Risaralda","Santander","Sucre","Tolima",
    "Valle del Cauca","Arauca","Casanare","Putumayo","San Andrés","Amazonas","Guainía",
    "Guaviare","Vaupés","Vichada"
]

def _hist_years(base_year: int, n_back: int = 7) -> List[int]:
    start = base_year - n_back
    return list(range(start, base_year + 1))

def generar_base(anio: int = 2024) -> pd.DataFrame:
    pobl = np.random.randint(100_000, 8_000_000, size=len(DEPARTAMENTOS)).astype(float)

    tasa_base = np.clip(np.random.normal(150, 15, size=len(DEPARTAMENTOS)), 110, 190)
    shock = np.random.normal(0, 4, size=len(DEPARTAMENTOS))
    tasa_proj = np.round(tasa_base + shock, 1)

    muertes_esp = np.round((pobl / 100_000) * tasa_base).astype(int)
    evitables = np.round(muertes_esp * np.random.uniform(0.03, 0.08, size=len(DEPARTAMENTOS))).astype(int)

    df = pd.DataFrame({
        "anio": anio,
        "departamento": DEPARTAMENTOS,
        "poblacion": pobl.astype(int),
        "tasa_base_x100k": tasa_base,
        "tasa_proyectada_x100k": tasa_proj,
        "muertes_esperadas_12m": muertes_esp,
        "muertes_evitables_estimadas": evitables
    })
    
    df["trend_slope"] = np.random.uniform(-2.0, -0.2, size=len(df))
    df["volatilidad"] = np.random.uniform(0.5, 1.8, size=len(df))
    df["comp_inversion"] = np.round(np.random.uniform(-2.0, -0.3, size=len(df)), 2)
    df["comp_cobertura"] = np.round(np.random.uniform(-2.5, -0.5, size=len(df)), 2)
    df["comp_calidad"] = np.round(np.random.uniform(0.3, 1.4, size=len(df)), 2)
    df["comp_otros"] = np.round(np.random.uniform(0.1, 0.9, size=len(df)), 2)
    return df

_CACHE: Dict[int, pd.DataFrame] = {}

def get_df(year: int) -> pd.DataFrame:
    if year not in _CACHE:
        _CACHE[year] = generar_base(year)
    return _CACHE[year].copy()

# ----------------- Página 1: INICIO -----------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "default_year": date.today().year})

@app.get("/api/metrics")
def api_metrics(year: int):
    df = get_df(year)
    k1 = int(df["muertes_esperadas_12m"].sum())
    k2 = int(df["muertes_evitables_estimadas"].sum())
    tasa_nac = (df["tasa_proyectada_x100k"] * df["poblacion"]).sum() / df["poblacion"].sum()
    return {
        "kpi1_muertes_esperadas_12m": k1,
        "kpi2_muertes_evitables_estimadas": k2,
        "kpi3_tasa_proyectada_x100k": round(float(tasa_nac), 1)
    }

@app.get("/api/grid")
def api_grid(year: int):
    df = get_df(year)[["departamento","tasa_base_x100k","tasa_proyectada_x100k","muertes_evitables_estimadas"]].copy()
    n = len(df); rows = 6; cols = int(np.ceil(n / rows))
    df = df.sort_values("tasa_proyectada_x100k", ascending=False).reset_index(drop=True)
    df["row"] = df.index // cols
    df["col"] = df.index % cols
    heat = df[["departamento","tasa_proyectada_x100k","row","col"]].to_dict(orient="records")
    table_records = df[["departamento","tasa_base_x100k","tasa_proyectada_x100k","muertes_evitables_estimadas"]].round(1).to_dict(orient="records")
    return {"heat": heat, "table": table_records, "rows": rows, "cols": cols}

@app.get("/api/top10")
def api_top10(year: int):
    df = get_df(year).sort_values("muertes_evitables_estimadas", ascending=False).head(10)
    return {"departamentos": df["departamento"].tolist(),
            "valores": df["muertes_evitables_estimadas"].astype(int).tolist()}

# ----------------- Página 2: SIMULACIÓN -----------------
BETA_INV_EVI = 0.6   # inversión reduce evitables
BETA_COV_EVI = 0.4   # cobertura reduce evitables
BETA_COV_TASA = 0.3  # cobertura reduce tasa

def aplicar_escenario(df: pd.DataFrame, inv_pct: int, cov_pct: int) -> Tuple[pd.DataFrame, dict]:
    df = df.copy()
    factor_evi = 1 - (inv_pct/100)*BETA_INV_EVI - (cov_pct/100)*BETA_COV_EVI
    df["muertes_evitables_estimadas_sim"] = np.maximum(0, np.round(df["muertes_evitables_estimadas"] * factor_evi).astype(int))
    factor_tasa = 1 - (cov_pct/100)*BETA_COV_TASA
    df["tasa_proyectada_x100k_sim"] = np.round(df["tasa_proyectada_x100k"] * factor_tasa, 1)
    kpi1 = int(df["muertes_esperadas_12m"].sum())
    kpi2 = int(df["muertes_evitables_estimadas_sim"].sum())
    tasa_nac = (df["tasa_proyectada_x100k_sim"] * df["poblacion"]).sum() / df["poblacion"].sum()
    resumen = {
        "escenario": {"inversion_pct": inv_pct, "cobertura_pct": cov_pct},
        "kpi1_muertes_esperadas_12m": kpi1,
        "kpi2_muertes_evitables_estimadas": kpi2,
        "kpi3_tasa_proyectada_x100k": round(float(tasa_nac), 1)
    }
    return df, resumen

@app.get("/simulacion", response_class=HTMLResponse)
async def simulacion_page(request: Request):
    return templates.TemplateResponse("simulacion.html", {"request": request, "default_year": date.today().year})

@app.get("/api/sim")
def api_sim(year: int, inv: int = 0, cov: int = 0):
    df = get_df(year)
    df2, resumen = aplicar_escenario(df, inv, cov)
    top = df2.sort_values("muertes_evitables_estimadas_sim", ascending=False).head(10)
    payload_top = {
        "departamentos": top["departamento"].tolist(),
        "valores": top["muertes_evitables_estimadas_sim"].astype(int).tolist()
    }
    table_records = df2[[
        "departamento","tasa_proyectada_x100k","tasa_proyectada_x100k_sim",
        "muertes_evitables_estimadas","muertes_evitables_estimadas_sim"
    ]].round(1).to_dict(orient="records")
    return {"resumen": resumen, "top10": payload_top, "tabla": table_records}

# ----------------- Página 3: DETALLE -----------------
@app.get("/detalle", response_class=HTMLResponse)
async def detalle_page(request: Request):
    return templates.TemplateResponse("detalle.html", {"request": request, "default_year": date.today().year})

@app.get("/api/deptos")
def api_deptos():
    return {"deptos": DEPARTAMENTOS}

def _serie_historica(row: pd.Series, base_year: int) -> Tuple[List[int], List[float]]:
    years = _hist_years(base_year, n_back=np.random.randint(5, 8))
    last_val = row["tasa_base_x100k"]
    series = []
    for i, _ in enumerate(reversed(years)):
        t = last_val - (len(years) - 1 - i) * row["trend_slope"]
        noise = np.random.normal(0, row["volatilidad"])
        series.append(max(100.0, float(np.round(t + noise, 1))))
    return years, series

@app.get("/api/detalle")
def api_detalle(year: int, dpto: str):
    df = get_df(year)
    if dpto not in set(df["departamento"]):
        return {"error": "Departamento no encontrado."}
    row = df.loc[df["departamento"] == dpto].iloc[0]

    kpi = {
        "tasa_base_x100k": round(float(row["tasa_base_x100k"]), 1),
        "tasa_proyectada_x100k": round(float(row["tasa_proyectada_x100k"]), 1),
        "muertes_evitables": int(row["muertes_evitables_estimadas"])
    }

    years, hist = _serie_historica(row, year)
    proj_year = years[-1] + 1
    proj_val = float(np.round(row["tasa_proyectada_x100k"], 1))
    band = float(np.round(max(0.8, 1.5 * row["volatilidad"]), 1))

    serie = {
        "years": years + [proj_year],
        "hist": hist + [None],
        "proj": [None]*len(hist) + [proj_val],
        "band_lo": [None]*len(hist) + [max(80.0, proj_val - band)],
        "band_hi": [None]*len(hist) + [proj_val + band]
    }

    contrib = {
        "labels": ["inversión","cobertura","calidad","otros"],
        "values": [float(row["comp_inversion"]),
                   float(row["comp_cobertura"]),
                   float(row["comp_calidad"]),
                   float(row["comp_otros"])]
    }
    return {"kpi": kpi, "serie": serie, "contrib": contrib}

