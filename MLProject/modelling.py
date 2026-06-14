import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Set konfigurasi agar MLflow bisa menyimpan di lokal server GitHub
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

def train():
    print("Memulai proses training...")
    
    # 2. Load Data
    try:
        X_train = pd.read_csv('namadataset_preprocessing/X_train.csv')
        X_test = pd.read_csv('namadataset_preprocessing/X_test.csv')
        y_train = pd.read_csv('namadataset_preprocessing/y_train.csv')
        y_test = pd.read_csv('namadataset_preprocessing/y_test.csv')
        print("Data berhasil dimuat.")
    except Exception as e:
        print(f"Error saat load data: {e}")
        return

    # 3. Setup MLflow
    mlflow.set_tracking_uri("file:///mlruns")
    mlflow.set_experiment("Loan_Prediction_CI")

    # 4. Training dengan Autolog
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="CI_Run"):
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train.values.ravel())
        
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Akurasi Model: {acc}")

if __name__ == "__main__":
    train()
