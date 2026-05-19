from __future__ import annotations

import json
from pathlib import Path


def save_json_report(job: dict, path: Path) -> None:
    path.write_text(json.dumps(job, indent=2, default=str), encoding="utf-8")


def save_pdf_report(job: dict, path: Path) -> None:
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except Exception:
        path.write_text(_plain_report(job), encoding="utf-8")
        return

    c = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter
    y = height - 48
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, y, "AutoML_XAI AutoML and Explainable AI Report")
    y -= 28
    c.setFont("Helvetica", 10)

    for line in _plain_report(job).splitlines():
        if y < 52:
            c.showPage()
            c.setFont("Helvetica", 10)
            y = height - 48
        c.drawString(40, y, line[:115])
        y -= 14
    c.save()


def _plain_report(job: dict) -> str:
    understanding = job.get("understanding", {})
    metrics = job.get("automl", {}).get("best_metrics", {})
    insights = job.get("insights", [])
    leaderboard = job.get("automl", {}).get("leaderboard", [])
    lines = [
        f"Job ID: {job.get('job_id')}",
        f"Instruction: {job.get('instruction')}",
        f"Task Type: {understanding.get('task_type')}",
        f"Industrial Context: {understanding.get('industrial_context')}",
        f"Best Model: {job.get('automl', {}).get('best_model')}",
        f"Metrics: {metrics}",
        "",
        "Leaderboard:",
    ]
    lines += [f"- {row.get('model')}: {row.get('metrics')}" for row in leaderboard]
    lines += ["", "Industrial Recommendations:"]
    lines += [f"- {item}" for item in insights]
    return "\n".join(lines)
