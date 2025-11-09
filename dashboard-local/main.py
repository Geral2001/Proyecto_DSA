from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import numpy as np
import pandas as pd
from datetime import date
import random

### primer layer de API

app = FastAPI(title="Muertes evitables – demo local")

# templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Vamos a trabajar con datos sintéticos, sin aún usar predicciones de los modelos que se están probando
# tablero sujeto a cambios en función de los resultados y test en Mlflow
random.seed(42)
np.random.seed(42)

DEPARTAMENTOS = [
    "Antioquia","Atlántico","Bogotá D.C.","Bolívar","Boyacá","Caldas","Caquetá","Cauca",
    "Cesar","Córdoba","Cundinamarca","Chocó","Huila","La Guajira","Magdalena","Meta",
    "Nariño","Norte de Santander","Quindío","Risaralda","Santander","Sucre","Tolima",
    "Valle del Cauca","Arauca","Casanare","Putumayo","San Andrés","Amazonas","Guainía",
    "Guaviare","Vaupés","Vichada"
]

def generar_base(anio: int = 2024):
    
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
    return df


_CACHE = {}

def get_df(anio: int):
    if anio not in _CACHE:
        _CACHE[anio] = generar_base(anio)
    return _CACHE[anio].copy()

# ---------- Rutas ----------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Página con los gráficos
    return templates.TemplateResponse("index.html", {"request": request, "default_year": date.today().year})

class YearParam(BaseModel):
    year: int

@app.get("/api/metrics")
def api_metrics(year: int):
    df = get_df(year)
    kpi1 = int(df["muertes_esperadas_12m"].sum())
    kpi2 = int(df["muertes_evitables_estimadas"].sum())
    # tasa proyectada “nacional” (promedio ponderado por población)
    tasa_nac = (df["tasa_proyectada_x100k"] * df["poblacion"]).sum() / df["poblacion"].sum()
    return {
        "kpi1_muertes_esperadas_12m": kpi1,
        "kpi2_muertes_evitables_estimadas": kpi2,
        "kpi3_tasa_proyectada_x100k": round(float(tasa_nac), 1)
    }

@app.get("/api/grid")
def api_grid(year: int):
    """Devuelve datos para el mapa tipo mosaico (heatmap) y una mini tabla de detalle."""
    df = get_df(year)[["departamento","tasa_base_x100k","tasa_proyectada_x100k","muertes_evitables_estimadas"]].copy()

    
    n = len(df)
    rows = 6
    cols = int(np.ceil(n / rows))
    df = df.sort_values("tasa_proyectada_x100k", ascending=False).reset_index(drop=True)
    df["row"] = df.index // cols
    df["col"] = df.index % cols

    heat = df[["departamento","tasa_proyectada_x100k","row","col"]].to_dict(orient="records")
    # mini tabla para el panel central
    table_records = df[["departamento","tasa_base_x100k","tasa_proyectada_x100k","muertes_evitables_estimadas"]].round(1).to_dict(orient="records")

    return {"heat": heat, "table": table_records, "rows": rows, "cols": cols}

@app.get("/api/top10")
def api_top10(year: int):
    df = get_df(year).sort_values("muertes_evitables_estimadas", ascending=False).head(10)
    return {
        "departamentos": df["departamento"].tolist(),
        "valores": df["muertes_evitables_estimadas"].astype(int).tolist()
    }
