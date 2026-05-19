import json
import os
import re
from typing import Any

import requests


TASK_KEYWORDS = {
    "anomaly_detection": ["anomaly", "abnormal", "unusual", "outlier", "fault", "vibration", "deviation"],
    "regression": ["forecast", "predict usage", "predict energy", "estimate", "rmse", "continuous", "temperature"],
    "classification": ["failure", "classify", "risk", "defect", "pass fail", "fault type", "shutdown"],
    "clustering": ["segment", "cluster", "group", "operating modes", "patterns"],
}

CONTEXT_KEYWORDS = {
    "predictive maintenance": ["failure", "maintenance", "motor", "bearing", "pump", "compressor", "vibration"],
    "energy forecasting": ["energy", "power", "load", "consumption", "kwh", "demand"],
    "SCADA analytics": ["scada", "historian", "tag", "plc", "telemetry"],
    "fault detection": ["fault", "alarm", "trip", "shutdown", "abnormal"],
}


def _extract_json(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def heuristic_understanding(instruction: str, columns: list[str]) -> dict[str, Any]:
    text = f"{instruction} {' '.join(columns)}".lower()
    scores = {
        task: sum(1 for word in words if word in text)
        for task, words in TASK_KEYWORDS.items()
    }
    task_type = max(scores, key=scores.get) if max(scores.values() or [0]) else "regression"
    if (
        ("predict" in text and any(token in text for token in ["failure", "fault", "status", "risk"]))
        or any(token in text for token in ["predict failure", "predict fault", "failure risk", "fault status", "classify", "defect"])
    ):
        task_type = "classification"
    elif any(token in text for token in ["detect unusual", "detect abnormal", "find abnormal", "anomaly", "outlier"]):
        task_type = "anomaly_detection"
    elif any(token in text for token in ["forecast", "predict energy", "predict load", "estimate"]):
        task_type = "regression"

    context_scores = {
        ctx: sum(1 for word in words if word in text)
        for ctx, words in CONTEXT_KEYWORDS.items()
    }
    context = max(context_scores, key=context_scores.get) if max(context_scores.values() or [0]) else "industrial predictive analytics"

    target_candidates = [
        col for col in columns
        if col.lower() in {"target", "label", "failure", "fault", "status", "energy", "power", "load"}
        or any(token in col.lower() for token in ["target", "label", "failure", "fault", "status", "energy", "power", "load"])
    ]

    return {
        "task_type": task_type,
        "industrial_context": context,
        "target_column": target_candidates[0] if target_candidates and task_type not in {"anomaly_detection", "clustering"} else None,
        "confidence": 0.72 if max(scores.values() or [0]) else 0.45,
        "reasoning": "Task inferred from industrial keywords, available columns, and common AutoML conventions.",
        "recommended_priority": "balanced",
    }


def understand_task(instruction: str, columns: list[str], sample_profile: dict[str, Any]) -> dict[str, Any]:
    provider = os.getenv("ENGINEEREDX_LLM_PROVIDER", "heuristic").lower()
    prompt = (
        "You are an industrial AutoML planner. Return only JSON with keys: "
        "task_type(regression|classification|anomaly_detection|clustering), industrial_context, "
        "target_column, confidence, reasoning, recommended_priority. "
        f"Instruction: {instruction}. Columns: {columns}. Profile: {sample_profile}"
    )

    try:
        if provider == "groq" and os.getenv("GROQ_API_KEY"):
            payload = {
                "model": os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile"),
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
            }
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"},
                json=payload,
                timeout=20,
            )
            response.raise_for_status()
            parsed = _extract_json(response.json()["choices"][0]["message"]["content"])
            if parsed:
                return {**heuristic_understanding(instruction, columns), **parsed, "llm_provider": "groq"}

        if provider == "ollama":
            response = requests.post(
                os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate"),
                json={"model": os.getenv("OLLAMA_MODEL", "llama3.1"), "prompt": prompt, "stream": False},
                timeout=30,
            )
            response.raise_for_status()
            parsed = _extract_json(response.json().get("response", ""))
            if parsed:
                return {**heuristic_understanding(instruction, columns), **parsed, "llm_provider": "ollama"}
    except Exception as exc:
        fallback = heuristic_understanding(instruction, columns)
        fallback["llm_warning"] = f"LLM unavailable, used deterministic planner: {exc}"
        return fallback

    fallback = heuristic_understanding(instruction, columns)
    fallback["llm_provider"] = "heuristic"
    return fallback
