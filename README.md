# MLflow Practice

A simple MLflow practice setup with scikit-learn classification.

## Setup

1. Install dependencies with uv:
```bash
uv sync
```

2. Start MLflow server:
```bash
docker-compose up -d
```

3. Run the experiment:
```bash
uv run python main.py
```

4. Open MLflow UI:
```
http://localhost:5000
```

## What to Explore in MLflow UI

- **Experiments tab**: See the "Iris Classification Practice" experiment
- **Runs**: Compare 3 different Random Forest configurations
- **Parameters**: Click runs to see n_estimators, max_depth values
- **Metrics**: Compare accuracy scores across runs
- **Artifacts**: Download the trained models and dataset info
- **Model Registry**: Register best performing model
- **Charts**: View parameter vs metric relationships