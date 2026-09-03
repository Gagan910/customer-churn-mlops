import joblib
import pandas as pd
import shap
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.pkl"

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)


def explain_prediction(customer_data):
    data = pd.DataFrame([customer_data])

    processed_data = preprocessor.transform(data)

    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(processed_data)[0]

    explanations = []

    for original_feature in data.columns:
        matching_indices = [
            i
            for i, name in enumerate(feature_names)
            if name.startswith(f"cat__{original_feature}_")
            or name == f"num__{original_feature}"
        ]

        if not matching_indices:
            continue

        total_shap = sum(shap_values[i] for i in matching_indices)

        explanations.append(
            {
                "feature": original_feature,
                "value": customer_data[original_feature],
                "shap_value": float(total_shap),
            }
        )

    explanations.sort(
        key=lambda x: abs(x["shap_value"]),
        reverse=True
    )

    return {
        "prediction": explanations[:10]
    }