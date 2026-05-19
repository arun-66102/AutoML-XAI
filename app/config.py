from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = BASE_DIR / "storage"
UPLOAD_DIR = STORAGE_DIR / "uploads"
MODEL_DIR = STORAGE_DIR / "models"
REPORT_DIR = STORAGE_DIR / "reports"
JOB_DIR = STORAGE_DIR / "jobs"

for folder in (UPLOAD_DIR, MODEL_DIR, REPORT_DIR, JOB_DIR):
    folder.mkdir(parents=True, exist_ok=True)

