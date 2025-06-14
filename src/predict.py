import joblib, pandas as pd, pathlib, numpy as np

MODEL_PATH = pathlib.Path(__file__).parent / "model_xgb.pkl"

def load_model(path: pathlib.Path = MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError("Run train.py first to create model.")
    return joblib.load(path)

def predict_one(record: dict) -> int:
    """
    Return 0 (legit) or 1 (fraud) for single transaction.
    """
    model = load_model()
    df = pd.DataFrame([record])
    # ➜ XGBoost / scikit-learn ≥1.4: передаём numpy, чтобы не проверялись имена
    proba = model.predict_proba(df.to_numpy())[0, 1]
    return int(proba >= 0.5)

if __name__ == "__main__":
    sample = {f"V{i}": 0 for i in range(1, 29)} | {"Time": 0, "Amount": 123}
    print("Prediction:", predict_one(sample))
