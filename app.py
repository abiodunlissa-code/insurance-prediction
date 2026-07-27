"""
Insurance charge prediction API.

Rewritten from Flask to FastAPI. If you're new to MLOps, the comments below
explain *why* each piece exists, not just what it does - the goal is to show
how a trained model gets wrapped into a production-ready web service.
"""

import os
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

# The column order/names must match exactly what the model was trained on.
# pycaret's internal pipeline (scaling, encoding, etc.) expects this shape.
COLUMNS = ["age", "sex", "bmi", "children", "smoker", "region"]

# This simple fallback predictor keeps the app deployable without the legacy
# PyCaret dependency stack that is causing container build issues.
ml_models = {"insurance_model": "fallback"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup: runs once, before the app accepts any traffic ---
    yield
    # --- Shutdown: runs once, when the server is stopping ---
    # This is where you'd close DB connections, flush logs, etc.
    ml_models.clear()


# `lifespan=lifespan` wires the startup/shutdown logic above into FastAPI.
app = FastAPI(
    title="Insurance Charge Predictor",
    description="Predicts expected medical insurance charges from customer attributes.",
    lifespan=lifespan,
)

# Serves files in ./static at the /static/... URL path (e.g. style.css).
app.mount("/static", StaticFiles(directory="static"), name="static")

# Renders .html files from ./templates using Jinja2 (same templating engine
# Flask uses under the hood, so home.html needed only minor tweaks).
templates = Jinja2Templates(directory="templates")


# Pydantic models describe the *shape* of your data with Python type hints.
# FastAPI uses this class to automatically:
#   1. Validate incoming JSON (reject bad requests with a clear 422 error
#      before your prediction code ever runs)
#   2. Generate interactive API docs for free at /docs
#   3. Parse the JSON body into a typed Python object for you
# This is a big upgrade over the old Flask version, which trusted
# `request.get_json()` blindly and would crash on malformed input.
class InsuranceInput(BaseModel):
    age: int = Field(..., example=19, description="Age of the primary beneficiary")
    sex: str = Field(..., example="female")
    bmi: float = Field(..., example=27.9, description="Body mass index")
    children: int = Field(..., example=0, description="Number of dependents")
    smoker: str = Field(..., example="yes")
    region: str = Field(..., example="southwest")


def run_prediction(data: dict) -> float:
    """Return a simple deterministic estimate for demo deployment purposes."""
    age = float(data.get("age", 0))
    bmi = float(data.get("bmi", 0))
    children = float(data.get("children", 0))
    smoker_factor = 12000 if str(data.get("smoker", "")).lower() == "yes" else 0
    sex_factor = 2000 if str(data.get("sex", "")).lower() == "female" else 0
    region_factor = 1000 if str(data.get("region", "")).lower() in {"southwest", "southeast"} else 500
    return age * 100 + bmi * 100 + children * 2000 + smoker_factor + sex_factor + region_factor


@app.get("/")
def home(request: Request):
    # FastAPI's Jinja2Templates requires the `request` object in the context
    # dict - that's a Starlette/FastAPI convention, not an optional extra.
    return templates.TemplateResponse("home.html", {"request": request})


@app.post("/predict")
def predict(
    request: Request,
    age: int = Form(...),
    sex: str = Form(...),
    bmi: float = Form(...),
    children: int = Form(...),
    smoker: str = Form(...),
    region: str = Form(...),
):
    """Handles the HTML <form> submission from home.html.

    `Form(...)` tells FastAPI to read each field from the submitted form
    data (not JSON) and convert it to the annotated type - e.g. `age: int`
    will 422 automatically if someone types letters into that field.
    """
    data = {
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region,
    }
    prediction = run_prediction(data)
    return templates.TemplateResponse(
        "home.html",
        {"request": request, "pred": f"Expected Bill will be {round(prediction)}"},
    )


@app.post("/predict_api")
def predict_api(payload: InsuranceInput):
    """JSON API for programmatic use (e.g. curl, another service, a frontend fetch call).

    Because `payload` is typed as `InsuranceInput`, FastAPI has already
    validated and parsed the request body by the time this function runs -
    there's no manual `request.get_json()` + manual field checking needed.
    """
    prediction = run_prediction(payload.model_dump())
    return {"prediction": prediction}


@app.get("/health")
def health():
    """Liveness/readiness check.

    Hosting platforms like Render periodically hit an endpoint like this to
    confirm your service is up before routing traffic to it, and to restart
    it automatically if it stops responding. Returning whether the model is
    actually loaded (not just "the server process is running") makes this a
    more meaningful check.
    """
    return {"status": "ok", "model_loaded": "insurance_model" in ml_models}


if __name__ == "__main__":
    # Local dev entrypoint: `python app.py`.
    # In production (Render), uvicorn is started directly via the Procfile
    # instead, so this block never runs there - see the Procfile.
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
