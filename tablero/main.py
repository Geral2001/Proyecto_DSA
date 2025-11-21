from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

import joblib
import numpy as np
import pandas as pd

# ------------------------------------
# CONFIGURACIÓN BÁSICA
# ------------------------------------

DEP_COL = "CODIGO_DEPARTAMENTO_RESIDENCIA"

# Diccionario código DANE → nombre amigable
CODE_TO_NAME = {
    "05": "Antioquia",
    "08": "Atlántico",
    "11": "Bogotá D.C.",
    "13": "Bolívar",
    "15": "Boyacá",
    "17": "Caldas",
    "18": "Caquetá",
    "19": "Cauca",
    "20": "Cesar",
    "23": "Córdoba",
    "25": "Cundinamarca",
    "27": "Chocó",
    "41": "Huila",
    "44": "La Guajira",
    "47": "Magdalena",
    "50": "Meta",
    "52": "Nariño",
    "54": "Norte de Santander",
    "63": "Quindío",
    "66": "Risaralda",
    "68": "Santander",
    "70": "Sucre",
    "73": "Tolima",
    "76": "Valle del Cauca",
    "81": "Arauca",
    "85": "Casanare",
    "86": "Putumayo",
    "88": "San Andrés, Providencia y Santa Catalina",
    "91": "Amazonas",
    "94": "Guainía",
    "95": "Guaviare",
    "97": "Vaupés",
    "99": "Vichada",
}

app = FastAPI(
    title="API Defunciones – Modelo HGB Poisson",
    description="Predice muertes mensuales por departamento y permite simular escenarios con cambios en inversión/cobertura.",
    version="1.0.0",
)

# Static + templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ------------------------------------
# CARGA DE MODELO Y PANEL
# ------------------------------------

try:
    MODEL = joblib.load("hgb_poisson_model.pkl")
    FEAT_COLS = joblib.load("hgb_feat_cols.pkl")
    PANEL = pd.read_parquet("panel_features.parquet")

    # Índice jerárquico para accesos rápidos
    PANEL_INDEXED = PANEL.set_index([DEP_COL, "ANO", "MES"]).sort_index()

    # Códigos de depto disponibles
    deptos_codes = sorted(PANEL[DEP_COL].dropna().astype(str).unique().tolist())

    DEPTOS = []
    for code in deptos_codes:
        code_str = str(code).zfill(2)
        name = CODE_TO_NAME.get(code_str, f"Depto {code_str}")
        label = f"{name} ({code_str})"
        DEPTOS.append({"code": code_str, "name": name, "label": label})

    max_year = int(PANEL["ANO"].max())

    print("✅ Modelo y panel cargados correctamente.")
    print(f"   • Features esperadas: {len(FEAT_COLS)}")
    print(f"   • Filas en panel API: {PANEL_INDEXED.shape[0]:,}")
    print(f"   • Departamentos únicos: {len(DEPTOS)}")
    print(f"   • Año máximo en panel: {max_year}")

except Exception as e:
    print("❌ Error cargando modelo/panel:", e)
    MODEL = None
    FEAT_COLS = None
    PANEL = None
    PANEL_INDEXED = None
    DEPTOS = []
    max_year = 2020


# ------------------------------------
# PÁGINA PRINCIPAL (predicción puntual)
# ------------------------------------

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "departamentos": DEPTOS,
            "default_year": max_year,
        },
    )


# ------------------------------------
# PÁGINA HISTÓRICO (gráfico de línea)
# ------------------------------------

@app.get("/historico/{dep_code}", response_class=HTMLResponse)
def historico(request: Request, dep_code: str):
    dep = str(dep_code).zfill(2)
    dep_name = CODE_TO_NAME.get(dep, f"Depto {dep}")

    if PANEL is None or dep not in PANEL[DEP_COL].astype(str).unique():
        raise HTTPException(
            status_code=404,
            detail=f"El departamento {dep_name} ({dep}) no existe en el panel.",
        )

    return templates.TemplateResponse(
        "historico.html",
        {
            "request": request,
            "dep_code": dep,
            "dep_name": dep_name,
        },
    )


# ------------------------------------
# PÁGINA SIMULACIÓN (gráfico fancy)
# ------------------------------------

@app.get("/simulacion", response_class=HTMLResponse)
def simulacion(request: Request):
    if PANEL is None:
        raise HTTPException(status_code=500, detail="El panel no está cargado en el servidor.")
    years = sorted(PANEL["ANO"].unique().tolist())
    default_year = int(max_year)
    return templates.TemplateResponse(
        "simulacion.html",
        {
            "request": request,
            "years": years,
            "default_year": default_year,
        },
    )


# ------------------------------------
# ENDPOINTS API: DEPTOS, PREDICCIÓN PUNTUAL, HISTÓRICO
# ------------------------------------

@app.get("/api/deptos")
def api_deptos():
    """Devuelve la lista de departamentos con código y nombre."""
    return {"deptos": DEPTOS}


@app.post("/predict")
def predict(data: dict):
    """
    Predice número de muertes para (departamento, año, mes).
    Espera JSON: {departamento: "25", ano: 2020, mes: 6}
    """

    if MODEL is None or FEAT_COLS is None or PANEL_INDEXED is None:
        raise HTTPException(status_code=500, detail="Modelo o panel no cargado en el servidor.")

    try:
        dep_code = str(data["departamento"]).zfill(2)
        ano = int(data["ano"])
        mes = int(data["mes"])
    except Exception:
        raise HTTPException(status_code=400, detail="JSON inválido. Se esperaba departamento, ano, mes.")

    key = (dep_code, ano, mes)
    dep_name = CODE_TO_NAME.get(dep_code, f"Depto {dep_code}")

    try:
        row = PANEL_INDEXED.loc[key]
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"No hay datos en el panel para {dep_name} (código {dep_code}), AÑO={ano}, MES={mes}",
        )

    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]

    row_index = set(row.index)
    missing = [c for c in FEAT_COLS if c not in row_index]
    if missing:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "En el panel faltan columnas que el modelo espera",
                "n_missing": len(missing),
                "some_missing": missing[:5],
            },
        )

    X = pd.DataFrame([row[FEAT_COLS].astype(np.float32).to_dict()])
    y_pred = MODEL.predict(X)[0]

    return {
        "input": {
            "departamento_codigo": dep_code,
            "departamento_nombre": dep_name,
            "ano": ano,
            "mes": mes,
        },
        "prediction_muertes": float(y_pred),
        "model_name": type(MODEL).__name__,
    }


@app.get("/api/historico/{dep_code}")
def api_historico(dep_code: str):
    """Devuelve la serie mensual de predicciones para un departamento."""
    if MODEL is None or FEAT_COLS is None or PANEL is None:
        raise HTTPException(status_code=500, detail="Modelo o panel no cargado en el servidor.")

    dep = str(dep_code).zfill(2)
    dep_name = CODE_TO_NAME.get(dep, f"Depto {dep}")

    df_dep = PANEL[PANEL[DEP_COL].astype(str) == dep].copy()
    if df_dep.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No hay datos en el panel para {dep_name} ({dep}).",
        )

    df_dep = df_dep.sort_values(["ANO", "MES"])
    X = df_dep[FEAT_COLS].astype(np.float32)
    preds = MODEL.predict(X)

    df_dep["pred"] = preds
    serie = [
        {
            "ano": int(a),
            "mes": int(m),
            "periodo": f"{int(a)}-{int(m):02d}",
            "pred": float(p),
        }
        for a, m, p in zip(df_dep["ANO"], df_dep["MES"], df_dep["pred"])
    ]

    return {
        "departamento_codigo": dep,
        "departamento_nombre": dep_name,
        "serie": serie,
    }


# ------------------------------------
# ENDPOINT SIMULACIÓN
# ------------------------------------

@app.get("/api/sim")
def api_sim(
    year: int = Query(..., ge=1900, le=2100),
    inv: int = Query(0, ge=-50, le=50),
    cov: int = Query(0, ge=-50, le=50),
):
    """
    Simula el efecto de cambiar % de inversión y % de cobertura
    sobre las muertes esperadas en un año.

    - year: año a simular
    - inv: % cambio en inversión (-30 a +30)
    - cov: % cambio en cobertura (-30 a +30)
    """

    if MODEL is None or FEAT_COLS is None or PANEL is None:
        raise HTTPException(status_code=500, detail="Modelo o panel no cargado en el servidor.")

    df_year = PANEL[PANEL["ANO"] == year].copy()
    if df_year.empty:
        raise HTTPException(status_code=404, detail=f"No hay datos en el panel para el año {year}.")

    # Predicción base con el modelo
    X = df_year[FEAT_COLS].astype(np.float32)
    base_pred = MODEL.predict(X)
    df_year["pred_base"] = base_pred

    # Factor de escenario:
    # +inversión y +cobertura ↓ muertes; valores negativos ↑ muertes
    factor = 1.0 - 0.01 * inv - 0.005 * cov
    # Evitar valores negativos o cero
    factor = max(0.1, factor)

    df_year["pred_scen"] = df_year["pred_base"] * factor

    # KPIs nacionales
    muertes_base = float(df_year["pred_base"].sum())
    muertes_scen = float(df_year["pred_scen"].sum())
    muertes_evitables = muertes_base - muertes_scen

    # Supongamos población constante 50 millones (aprox COL) para la tasa
    POP_NACIONAL = 50_000_000
    tasa_x100k = muertes_scen / POP_NACIONAL * 100_000

    resumen = {
        "escenario": {
            "inversion_pct": inv,
            "cobertura_pct": cov,
            "factor_aplicado": factor,
        },
        "kpi1_muertes_esperadas_12m": round(muertes_scen),
        "kpi2_muertes_evitables_estimadas": round(muertes_evitables),
        "kpi3_tasa_proyectada_x100k": tasa_x100k,
    }

    # Top-10 departamentos por muertes evitables (base - escenario)
    df_dep = (
        df_year.groupby(DEP_COL)
        .agg(base=("pred_base", "sum"), scen=("pred_scen", "sum"))
        .reset_index()
    )
    df_dep["delta"] = df_dep["base"] - df_dep["scen"]
    df_dep = df_dep.sort_values("delta", ascending=False).head(10)

    departamentos = []
    valores = []
    for _, row in df_dep.iterrows():
        code = str(row[DEP_COL]).zfill(2)
        name = CODE_TO_NAME.get(code, f"Depto {code}")
        departamentos.append(f"{name} ({code})")
        valores.append(float(row["delta"]))

    top10 = {
        "departamentos": departamentos,
        "valores": valores,
    }

    return {
        "year": year,
        "resumen": resumen,
        "top10": top10,
    }


# ------------------------------------
# HEALTHCHECK
# ------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "panel_rows": PANEL_INDEXED.shape[0] if PANEL_INDEXED is not None else None,
    }
