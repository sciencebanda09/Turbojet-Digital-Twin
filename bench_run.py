import json
import joblib
from src.deployment.benchmark import benchmark_latency
from src.dataset.loader import load_dataset

model = joblib.load("models/best_model.joblib")

df = load_dataset("data/turbojet_complete_dataset.csv", require_targets=False)
sample = df[model.feature_names].iloc[[0]]

result = benchmark_latency(lambda x: model.predict(x), sample, runs=200)
print(json.dumps(result, indent=2))
