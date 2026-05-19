from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.automl import run_automl, save_model_path
from app.config import JOB_DIR, MODEL_DIR, REPORT_DIR, UPLOAD_DIR
from app.eda import profile_dataframe
from app.llm import understand_task
from app.reports import save_json_report, save_pdf_report


app = FastAPI(
    title="ABB EngineeredX 2.0",
    description="Industrial LLM-driven AutoML platform for predictive analytics, anomaly detection, and adaptive ML workflows.",
    version="2.0.0",
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse("static/index.html")


@app.get("/api/health")
def health() -> dict:
    return {"status": "online", "platform": "ABB EngineeredX 2.0"}


@app.post("/api/analyze")
async def analyze_dataset(
    file: UploadFile = File(...),
    instruction: str = Form("Automatically understand this industrial dataset and build the best ML workflow."),
) -> JSONResponse:
    job_id = uuid.uuid4().hex[:12]
    suffix = Path(file.filename or "dataset.csv").suffix.lower()
    if suffix not in {".csv", ".xlsx", ".xls"}:
        raise HTTPException(status_code=400, detail="Upload a CSV or Excel dataset.")

    upload_path = UPLOAD_DIR / f"{job_id}_{Path(file.filename or 'dataset').name}"
    upload_path.write_bytes(await file.read())

    try:
        df = _read_dataset(upload_path)
        if df.empty or df.shape[1] < 2:
            raise ValueError("Dataset must contain rows and at least two columns.")
        job = _run_job(job_id, df, upload_path, instruction, feedback="")
        return JSONResponse(job)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc


@app.post("/api/refine")
async def refine_job(job_id: str = Form(...), feedback: str = Form(...)) -> JSONResponse:
    job_path = JOB_DIR / f"{job_id}.json"
    if not job_path.exists():
        raise HTTPException(status_code=404, detail="Job not found.")
    previous = json.loads(job_path.read_text(encoding="utf-8"))
    upload_path = Path(previous["upload_path"])
    df = _read_dataset(upload_path)
    instruction = f"{previous.get('instruction', '')}. Refinement: {feedback}"
    job = _run_job(job_id, df, upload_path, instruction, feedback=feedback, previous=previous)
    return JSONResponse(job)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> JSONResponse:
    job_path = JOB_DIR / f"{job_id}.json"
    if not job_path.exists():
        raise HTTPException(status_code=404, detail="Job not found.")
    return JSONResponse(json.loads(job_path.read_text(encoding="utf-8")))


@app.get("/api/download/{job_id}/{artifact}")
def download_artifact(job_id: str, artifact: str) -> FileResponse:
    mapping = {
        "json": REPORT_DIR / f"{job_id}_report.json",
        "pdf": REPORT_DIR / f"{job_id}_report.pdf",
        "model": MODEL_DIR / f"{job_id}_best_model.pkl",
    }
    path = mapping.get(artifact)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="Artifact not found.")
    return FileResponse(path, filename=path.name)


@app.get("/api/sample-data")
def sample_data() -> FileResponse:
    path = UPLOAD_DIR / "sample_motor_telemetry.csv"
    if not path.exists():
        rng = np.random.default_rng(42)
        rows = 360
        vibration = rng.normal(2.4, 0.45, rows)
        temperature = rng.normal(72, 6, rows)
        current = rng.normal(38, 4, rows)
        load = rng.normal(0.68, 0.12, rows)
        fault = ((vibration > 3.05) & (temperature > 78) | (current > 45)).astype(int)
        data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=rows, freq="h"),
                "motor_vibration_mm_s": vibration.round(3),
                "bearing_temperature_c": temperature.round(2),
                "phase_current_a": current.round(2),
                "load_factor": load.clip(0.25, 1.05).round(3),
                "fault_status": fault,
            }
        )
        data.to_csv(path, index=False)
    return FileResponse(path, filename="sample_motor_telemetry.csv")


def _read_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def _run_job(
    job_id: str,
    df: pd.DataFrame,
    upload_path: Path,
    instruction: str,
    feedback: str = "",
    previous: dict | None = None,
) -> dict:
    eda = profile_dataframe(df)
    understanding = understand_task(instruction, df.columns.tolist(), {"rows": eda["rows"], "columns": eda["columns"]})
    if previous and previous.get("understanding"):
        understanding = {**previous["understanding"], **understanding}

    model_path = save_model_path(job_id, MODEL_DIR)
    automl = run_automl(df, understanding, feedback=feedback, model_path=model_path)
    insights = _industrial_insights(eda, understanding, automl)
    job = {
        "job_id": job_id,
        "instruction": instruction,
        "feedback": feedback,
        "upload_path": str(upload_path),
        "dataset": {"filename": upload_path.name, "rows": eda["rows"], "columns": eda["columns"]},
        "understanding": understanding,
        "eda": eda,
        "automl": automl,
        "insights": insights + automl.get("recommendations", []),
        "artifacts": {
            "json": f"/api/download/{job_id}/json",
            "pdf": f"/api/download/{job_id}/pdf",
            "model": f"/api/download/{job_id}/model",
        },
    }
    json_path = REPORT_DIR / f"{job_id}_report.json"
    pdf_path = REPORT_DIR / f"{job_id}_report.pdf"
    save_json_report(job, json_path)
    save_pdf_report(job, pdf_path)
    (JOB_DIR / f"{job_id}.json").write_text(json.dumps(job, indent=2, default=str), encoding="utf-8")
    return job


def _industrial_insights(eda: dict, understanding: dict, automl: dict) -> list[str]:
    insights = []
    if eda["missing_total"]:
        insights.append(f"Detected {eda['missing_total']} missing sensor values; automatic imputation was applied before training.")
    if eda["duplicate_rows"]:
        insights.append(f"Found {eda['duplicate_rows']} duplicate historian rows; review export settings for repeated samples.")
    if understanding.get("task_type") == "classification" and automl.get("best_metrics", {}).get("f1", 1) < 0.75:
        insights.append("Model confidence is moderate; collect more labeled fault examples before safety-critical deployment.")
    if understanding.get("task_type") == "regression" and automl.get("best_metrics", {}).get("r2", 1) < 0.5:
        insights.append("Forecast fit is limited; add operating mode, shift, weather, or production schedule features.")
    if automl.get("xai_summary"):
        insights.append(automl["xai_summary"])
    return insights
