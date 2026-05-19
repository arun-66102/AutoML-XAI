# AutoML_XAI

AutoML_XAI is a hackathon project built for the **ABB EngineeredX** problem statement. The project demonstrates an industrial AI platform that can automatically understand uploaded industrial datasets, decide the right machine learning approach, train suitable models, explain the results, and generate practical outputs without requiring the user to have machine learning expertise.

The goal is to make machine learning more accessible for industrial teams working with sensor data, SCADA exports, energy data, equipment logs, and predictive maintenance datasets.

## Hackathon Context

**Hackathon:** ABB EngineeredX  
**Project Name:** AutoML_XAI  
**Theme:** Industrial AI, AutoML, Explainable AI, Predictive Analytics  

The problem statement focuses on building an intelligent system that can:

> Automatically understand industrial datasets and intelligently decide which machine learning approach should be used without requiring ML expertise from the user.

AutoML_XAI addresses this by combining:

- Natural-language task understanding
- Automated exploratory data analysis
- Intelligent preprocessing
- AutoML model selection
- Explainable AI insights
- Industrial report generation
- Feedback-driven workflow refinement

## Problem Statement

Industrial organizations collect large volumes of operational data from machines, sensors, controllers, meters, and SCADA systems. However, converting this data into useful machine learning models usually requires expertise in:

- Data cleaning
- Feature engineering
- Model selection
- Hyperparameter tuning
- Evaluation metrics
- Explainability
- Deployment preparation

Many plant engineers, maintenance teams, and operations users understand the equipment and process behavior very well, but they may not know which ML algorithm to choose or how to build a complete ML pipeline.

AutoML_XAI solves this gap by acting as an intelligent ML assistant for industrial datasets.

## Proposed Solution

AutoML_XAI allows a user to upload a CSV or Excel dataset and provide a simple instruction such as:

```text
Predict transformer failure.
Detect unusual vibration patterns.
Forecast energy usage.
Find abnormal machine behavior.
```

The system then automatically:

1. Understands the user instruction.
2. Detects the industrial use case.
3. Identifies the ML task type.
4. Performs EDA.
5. Builds a preprocessing pipeline.
6. Trains multiple candidate models.
7. Compares model performance.
8. Selects the best model.
9. Explains important features.
10. Generates reports, recommendations, and a trained model artifact.

## Industry Relevance

AutoML_XAI is relevant to industries where large amounts of operational data are generated but ML expertise is not always available at the plant or asset level.

### Predictive Maintenance

Industrial assets such as motors, transformers, pumps, drives, bearings, and compressors often generate sensor signals such as vibration, temperature, current, pressure, and load. AutoML_XAI can help identify patterns linked to future failure risk and generate maintenance-focused recommendations.

### Energy Forecasting

Factories and utilities need to forecast power demand, energy consumption, and load behavior. AutoML_XAI can automatically treat these problems as regression or time-series-style forecasting tasks and evaluate suitable models.

### Fault Detection

Industrial systems often show early abnormal behavior before alarms are triggered. AutoML_XAI supports anomaly detection workflows using methods such as Isolation Forest and DBSCAN to detect unusual operating conditions.

### SCADA and Historian Analytics

SCADA and historian systems store process tags over time. AutoML_XAI can process exported historian datasets, detect missing values, extract datetime features, analyze correlations, and build models from tag-level data.

### Operations Decision Support

The project does not only produce predictions. It also explains which features influenced the result, helping engineers understand whether signals such as vibration, temperature, current, or load are driving the prediction.

### Reducing Dependency on ML Experts

In many industrial settings, data scientists are not available for every plant-level analytics requirement. AutoML_XAI enables domain experts to experiment with ML workflows using natural language and automated model selection.

## Core Features

### Dataset Upload

Supports CSV and Excel uploads for industrial datasets such as:

- Sensor logs
- SCADA historian exports
- Energy monitoring datasets
- Equipment telemetry
- Time-series machine data
- Fault and maintenance records

### Natural-Language Understanding

The platform interprets user instructions and detects:

- Regression
- Classification
- Anomaly detection
- Clustering
- Industrial context
- Candidate target column

It supports Groq API and Ollama integration. If no LLM is configured, it uses deterministic fallback logic so the project still works locally.

### Automated EDA

The system automatically analyzes:

- Dataset shape
- Numeric columns
- Categorical columns
- Missing values
- Duplicate rows
- Datetime columns
- Class imbalance candidates
- Correlations
- Summary statistics

### Intelligent Preprocessing

The preprocessing pipeline handles:

- Missing-value imputation
- Categorical encoding
- Numeric scaling
- Datetime feature extraction
- Constant-column removal
- SMOTE balancing for imbalanced classification when suitable

### AutoML Model Selection

Based on the detected task type, the platform trains and compares models.

Regression:

- Linear Regression
- SVR
- Random Forest Regressor
- XGBoost Regressor

Classification:

- Logistic Regression
- SVM
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost Classifier

Anomaly Detection:

- Isolation Forest
- DBSCAN

Clustering:

- KMeans

### Evaluation Dashboard

The dashboard displays metrics such as:

- Accuracy
- F1-score
- ROC-AUC
- RMSE
- MAE
- R2
- Silhouette score
- Anomaly rate
- Model leaderboard

### Explainable AI

AutoML_XAI provides feature-importance-based explanations and industrial recommendations. The goal is to make predictions understandable to engineers and operations teams.

Example insight:

```text
Motor failure risk is influenced mainly by vibration, bearing temperature, and phase current.
```

### Output Generation

Each run can generate:

- JSON report
- PDF report
- Trained model `.pkl`
- Prediction preview
- Industrial recommendations

### Feedback Loop

Users can refine the ML workflow using prompts such as:

```text
Use only tree-based models.
Reduce false negatives.
Improve anomaly detection sensitivity.
```

The system reruns the workflow using the feedback as an adaptive constraint.

## System Workflow

```text
Dataset Upload
  -> Natural-Language Understanding
  -> Automated EDA
  -> Intelligent Preprocessing
  -> AutoML Training
  -> Model Evaluation
  -> Explainable AI
  -> Industrial Report Generation
  -> Feedback Refinement
```

## Tech Stack

Frontend:

- HTML
- CSS
- JavaScript
- Chart.js

Backend:

- FastAPI
- Python

AI and ML:

- Ollama
- Groq API
- Scikit-learn
- XGBoost
- SHAP-ready explainability layer
- Pandas
- NumPy
- Optuna
- imbalanced-learn

Storage:

- Local filesystem storage for uploaded datasets, trained models, reports, and job metadata

Not used:

- Docker
- Kubernetes
- React / Next.js
- Heavy cloud infrastructure
- Complex enterprise deployment tools

## Project Structure

```text
AutoML_XAI
├── main.py
├── requirements.txt
├── README.md
├── app/
│   ├── __init__.py
│   ├── automl.py
│   ├── config.py
│   ├── eda.py
│   ├── llm.py
│   └── reports.py
├── static/
│   ├── index.html
│   ├── css/
│   │   └── styles.css
│   └── js/
│       └── app.js
└── storage/
    ├── uploads/
    ├── models/
    ├── reports/
    └── jobs/
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Opens the dashboard |
| `GET` | `/api/health` | Checks backend health |
| `POST` | `/api/analyze` | Uploads dataset and runs AutoML |
| `POST` | `/api/refine` | Applies feedback and reruns workflow |
| `GET` | `/api/jobs/{job_id}` | Retrieves saved job metadata |
| `GET` | `/api/download/{job_id}/json` | Downloads JSON report |
| `GET` | `/api/download/{job_id}/pdf` | Downloads PDF report |
| `GET` | `/api/download/{job_id}/model` | Downloads trained model |
| `GET` | `/api/sample-data` | Downloads sample motor telemetry CSV |

## How to Run Locally

Create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Start the FastAPI server:

```powershell
py -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Open the dashboard:

```text
http://127.0.0.1:8000
```

## LLM Configuration

The project works without an LLM by using fallback heuristics.

To use Groq:

```powershell
$env:GROQ_API_KEY="your-groq-api-key"
$env:ENGINEEREDX_LLM_PROVIDER="groq"
$env:GROQ_MODEL="llama-3.1-70b-versatile"
```

To use Ollama:

```powershell
ollama pull llama3.1
$env:ENGINEEREDX_LLM_PROVIDER="ollama"
$env:OLLAMA_MODEL="llama3.1"
```

## Demo Instructions

1. Start the app.
2. Open `http://127.0.0.1:8000`.
3. Download the sample dataset from `/api/sample-data`.
4. Upload the dataset.
5. Enter a prompt such as:

```text
Predict motor failure and explain vibration risk.
```

6. Review the detected task type, EDA, leaderboard, best model, metrics, feature importance, and recommendations.
7. Apply feedback such as:

```text
Use only tree-based models and reduce false negatives.
```

8. Download the generated report and trained model.

## Expected Output

After a successful run, AutoML_XAI generates:

- Best model name
- Evaluation metrics
- Model leaderboard
- Prediction preview
- Feature importance
- Industrial insights
- JSON report
- PDF report
- Trained `.pkl` model

## Why This Project Matters

AutoML_XAI is designed to bridge the gap between industrial domain knowledge and machine learning implementation. Instead of expecting every engineer to understand model selection, tuning, and explainability, the system automates the ML workflow and presents the output in a way that is useful for industrial decision-making.

This makes the project suitable for:

- ABB-style industrial AI hackathons
- Smart manufacturing demos
- Predictive maintenance prototypes
- Energy analytics proof-of-concepts
- SCADA data analytics experiments
- Academic AutoML/XAI demonstrations

## Final Outcome

AutoML_XAI demonstrates a complete industrial AutoML and Explainable AI workflow for the ABB EngineeredX hackathon problem statement. It shows how an intelligent assistant can understand industrial data problems, select machine learning approaches automatically, train models, explain results, and generate practical outputs for real-world industrial analytics.
