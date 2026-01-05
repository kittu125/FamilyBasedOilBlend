import gradio as gr
import joblib
import pandas as pd

# Load artifacts
model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")
y_encoder = joblib.load("target_encoder.pkl")
FEATURE_COLUMNS = joblib.load("feature_columns.pkl")

def normalize(x):
    return str(x).lower().strip()

def predict_oil_blend(
    family_size,
    age_mix,
    cardio_history,
    cooking_temp,
    cooking_style,
    usage
):
    input_data = {
        "FamilySize": family_size,
        "AgeMix": age_mix,
        "CardioHistory": cardio_history,
        "CookingTemp": cooking_temp,
        "CookingStyle": cooking_style,
        "Usage": usage
    }

    row = {}
    for col in FEATURE_COLUMNS:
        value = normalize(input_data[col])
        row[col] = encoders[col].transform([value])[0]

    X_new = pd.DataFrame([row])

    pred_encoded = model.predict(X_new)[0]
    pred_label = y_encoder.inverse_transform([pred_encoded])[0]

    return f"🫒 Recommended Oil Blend: **{pred_label.title()}**"

iface = gr.Interface(
    fn=predict_oil_blend,
    inputs=[
        gr.Dropdown(encoders["FamilySize"].classes_.tolist(), label="Family Size"),
        gr.Dropdown(encoders["AgeMix"].classes_.tolist(), label="Age Mix"),
        gr.Dropdown(encoders["CardioHistory"].classes_.tolist(), label="Cardio History"),
        gr.Dropdown(encoders["CookingTemp"].classes_.tolist(), label="Cooking Temp"),
        gr.Dropdown(encoders["CookingStyle"].classes_.tolist(), label="Cooking Style"),
        gr.Dropdown(encoders["Usage"].classes_.tolist(), label="Usage"),
    ],
    outputs=gr.Markdown(),
    title="🛢️ Oil Clinic – Family Based Recommendation"
)

iface.launch()
