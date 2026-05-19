let currentJobId = null;
let charts = {};

const form = document.getElementById("uploadForm");
const feedbackForm = document.getElementById("feedbackForm");
const statusText = document.getElementById("status");
const datasetInput = document.getElementById("dataset");
const fileLabel = document.getElementById("fileLabel");
const fileDrop = document.getElementById("fileDrop");

datasetInput.addEventListener("change", () => {
  updateSelectedFile();
});

["dragenter", "dragover"].forEach((name) => {
  fileDrop.addEventListener(name, (event) => {
    event.preventDefault();
    fileDrop.classList.add("dragging");
  });
});

["dragleave", "drop"].forEach((name) => {
  fileDrop.addEventListener(name, (event) => {
    event.preventDefault();
    fileDrop.classList.remove("dragging");
  });
});

fileDrop.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (!file) return;
  const transfer = new DataTransfer();
  transfer.items.add(file);
  datasetInput.files = transfer.files;
  updateSelectedFile();
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = document.getElementById("dataset").files[0];
  const instruction = document.getElementById("instruction").value.trim();
  if (!file) return;
  setStatus("Uploading dataset and starting AutoML run...");
  markPipeline(1);
  const data = new FormData();
  data.append("file", file);
  data.append("instruction", instruction);
  await submitRun("/api/analyze", data);
});

feedbackForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!currentJobId) {
    setStatus("Run an analysis before applying feedback.");
    return;
  }
  const feedback = document.getElementById("feedback").value.trim();
  if (!feedback) return;
  setStatus("Applying feedback and rebuilding the workflow...");
  const data = new FormData();
  data.append("job_id", currentJobId);
  data.append("feedback", feedback);
  await submitRun("/api/refine", data);
});

async function submitRun(url, data) {
  try {
    const response = await fetch(url, { method: "POST", body: data });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Analysis failed");
    currentJobId = payload.job_id;
    renderJob(payload);
    markPipeline(7);
    setStatus(`Completed job ${payload.job_id}.`);
  } catch (error) {
    setStatus(error.message);
  }
}

function renderJob(job) {
  const understanding = job.understanding || {};
  const eda = job.eda || {};
  const automl = job.automl || {};
  document.getElementById("taskType").textContent = label(understanding.task_type);
  document.getElementById("industrialContext").textContent = understanding.industrial_context || "-";
  document.getElementById("targetColumn").textContent = understanding.target_column || "Not required";
  document.getElementById("rows").textContent = eda.rows ?? "-";
  document.getElementById("cols").textContent = eda.columns ?? "-";
  document.getElementById("missing").textContent = eda.missing_total ?? "-";
  document.getElementById("dupes").textContent = eda.duplicate_rows ?? "-";
  document.getElementById("bestModel").textContent = automl.best_model || "-";
  document.getElementById("bestMetrics").textContent = JSON.stringify(automl.best_metrics || {}, null, 2);
  document.getElementById("xaiSummary").textContent = automl.xai_summary || "No explanation generated.";

  renderLeaderboard(automl.leaderboard || []);
  renderPredictions(automl.prediction_preview || []);
  renderInsights(job.insights || []);
  renderCharts(eda, automl);
  bindDownloads(job.artifacts || {});
}

function renderLeaderboard(rows) {
  const body = document.getElementById("leaderboard");
  body.innerHTML = "";
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${row.model}</td><td><code>${JSON.stringify(row.metrics)}</code></td>`;
    body.appendChild(tr);
  });
}

function renderPredictions(rows) {
  const box = document.getElementById("predictions");
  box.innerHTML = "";
  rows.slice(0, 12).forEach((row) => {
    const item = document.createElement("span");
    const value = row.prediction ?? row.status ?? row.cluster ?? "-";
    item.innerHTML = `<small>Row ${row.row}</small><strong>${value}</strong>`;
    box.appendChild(item);
  });
}

function renderInsights(items) {
  const list = document.getElementById("insights");
  list.innerHTML = "";
  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    list.appendChild(li);
  });
}

function renderCharts(eda, automl) {
  drawChart("typesChart", {
    type: "doughnut",
    data: {
      labels: ["Numeric", "Categorical"],
      datasets: [{ data: [(eda.numeric_columns || []).length, (eda.categorical_columns || []).length], backgroundColor: ["#a8ff1a", "#ffe44d"] }],
    },
    options: baseOptions(),
  });

  const missing = eda.missing_values || {};
  drawChart("missingChart", {
    type: "bar",
    data: {
      labels: Object.keys(missing).slice(0, 8),
      datasets: [{ label: "Missing", data: Object.values(missing).slice(0, 8), backgroundColor: "#ffe44d" }],
    },
    options: baseOptions(),
  });

  const importance = automl.feature_importance || [];
  drawChart("importanceChart", {
    type: "bar",
    data: {
      labels: importance.map((x) => x.feature),
      datasets: [{ label: "Importance", data: importance.map((x) => x.importance), backgroundColor: "#a8ff1a" }],
    },
    options: { ...baseOptions(), indexAxis: "y" },
  });
}

function drawChart(id, config) {
  if (!window.Chart) {
    return;
  }
  if (charts[id]) charts[id].destroy();
  charts[id] = new Chart(document.getElementById(id), config);
}

function baseOptions() {
  return {
    responsive: true,
    plugins: { legend: { labels: { color: "#f7ffe9" } } },
    scales: {
      x: { ticks: { color: "#b9c9a0" }, grid: { color: "#395222" } },
      y: { ticks: { color: "#b9c9a0" }, grid: { color: "#395222" } },
    },
  };
}

function bindDownloads(artifacts) {
  [["jsonReport", artifacts.json], ["pdfReport", artifacts.pdf], ["modelFile", artifacts.model]].forEach(([id, href]) => {
    const link = document.getElementById(id);
    if (href) {
      link.href = href;
      link.classList.remove("disabled");
    }
  });
}

function markPipeline(count) {
  document.querySelectorAll("#pipeline span").forEach((step, index) => {
    step.classList.toggle("done", index < count);
  });
}

function setStatus(text) {
  statusText.textContent = text;
}

function updateSelectedFile() {
  const file = datasetInput.files[0];
  fileLabel.textContent = file ? file.name : "Choose or drop a dataset";
  setStatus(file ? `Selected ${file.name}. Ready to run.` : "Waiting for dataset.");
}

function label(value) {
  return (value || "-").replaceAll("_", " ").replace(/\b\w/g, (char) => char.toUpperCase());
}
