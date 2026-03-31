import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import plotly.express as px
import numpy as np
import os
import joblib

MODEL_PATH = "runtime_model.joblib"

def train_model():
    if not os.path.exists("benchmark_results.csv"):
        print("No benchmark data found to train ML model. Generating mock dataset...")

        # generate mock dataset for CI
        df = pd.DataFrame(
            {
                "grid_size": np.random.randint(100, 2000, 100),
                "water_content": np.random.uniform(0.01, 0.4, 100),
                "runtime_s": np.random.uniform(10, 300, 100),
            }
        )
        df["runtime_s"] += df["grid_size"] * 0.1 + df["water_content"] * 50
    else:
        df = pd.read_csv("benchmark_results.csv")

    X = df[["grid_size", "water_content"]]
    y = df["runtime_s"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model for future use
    joblib.dump(model, MODEL_PATH)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Model trained and saved to {MODEL_PATH}. MSE: {mse:.2f}")

    # Prediction error plot
    plot_df = pd.DataFrame(
        {"Actual Runtime (s)": y_test, "Predicted Runtime (s)": predictions}
    )
    fig = px.scatter(
        plot_df,
        x="Actual Runtime (s)",
        y="Predicted Runtime (s)",
        title="ML Runtime Prediction Error",
    )
    fig.add_shape(
        type="line",
        x0=y.min(),
        y0=y.min(),
        x1=y.max(),
        y1=y.max(),
        line=dict(color="red", dash="dash"),
    )

    fig.write_html("ml_prediction_error.html")
    print("Saved prediction error plot to ml_prediction_error.html")

def predict_runtime(grid_size, water_content):
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Training...")
        train_model()
    
    model = joblib.load(MODEL_PATH)
    prediction = model.predict([[grid_size, water_content]])
    return prediction[0]

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        train_model()
    else:
        print("Using saved model for prediction...")
        # Example prediction
        res = predict_runtime(500, 0.2)
        print(f"Predicted runtime for 500 nodes, 0.2 water content: {res:.2f}s")
