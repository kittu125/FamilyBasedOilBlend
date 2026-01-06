import gradio as gr
import pandas as pd
import joblib
import numpy as np

# ----------------------------
# Load trained artifacts
# ----------------------------
model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")
y_encoder = joblib.load("target_encoder.pkl")
FEATURE_COLUMNS = joblib.load("feature_columns.pkl")

# ----------------------------
# Normalize encoder classes
# (must stay NumPy array!)
# ----------------------------
for col, le in encoders.items():
    le.classes_ = np.array(
        [str(c).lower().strip() for c in le.classes_]
    )

# ----------------------------
# Helper: normalize text
# ----------------------------
def normalize(x):
    return str(x).lower().strip()

# ----------------------------
# Prediction function
# ----------------------------
def predict_oil_blend(
    family_size,
    age_mix,
    cardio_history,
    cooking_temp,
    cooking_style,
    usage
):
            # ----------------------------
        # SAFETY RULE OVERRIDE (CRITICAL)
        # ----------------------------
    if cardio_history == "strong":
        return (
                "🫒 **Recommended Oil Blend:** Heart-Safe Blend "
                "(Doctor-Preferred)\n\n"
                "⚠️ This recommendation prioritizes cardiac safety. "
                "Please consult a healthcare professional for "
                "personalized advice."
            )

        # ----------------------------

    try:
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

            if value in encoders[col].classes_:
                row[col] = encoders[col].transform([value])[0]
            else:
                # safety fallback (should rarely happen now)
                row[col] = encoders[col].transform(
                    [encoders[col].classes_[0]]
                )[0]

        X_new = pd.DataFrame([row])

        pred_encoded = model.predict(X_new)[0]
        pred_label = y_encoder.inverse_transform([pred_encoded])[0]

        return f"🫒 **Recommended Oil Blend:** {pred_label.title()}"

    except Exception as e:
        return f"❌ Error: {str(e)}"

# ----------------------------
# Gradio UI
# ----------------------------
iface = gr.Interface(
    fn=predict_oil_blend,
    inputs=[
        gr.Dropdown(
            choices=encoders["FamilySize"].classes_.tolist(),
            label="Family Size"
        ),
        gr.Dropdown(
            choices=encoders["AgeMix"].classes_.tolist(),
            label="Age Mix"
        ),
        gr.Dropdown(
            choices=encoders["CardioHistory"].classes_.tolist(),
            label="Cardio History"
        ),
        gr.Dropdown(
            choices=encoders["CookingTemp"].classes_.tolist(),
            label="Cooking Temp"
        ),
        gr.Dropdown(
            choices=encoders["CookingStyle"].classes_.tolist(),
            label="Cooking Style"
        ),
        gr.Dropdown(
            choices=encoders["Usage"].classes_.tolist(),
            label="Usage"
        ),
    ],
    outputs=gr.Markdown(label="Oil Recommendation"),
    title="🛢️ Oil Clinic – Family Based Recommendation",
    description="This recommendation is for general cooking guidance only and not a medical substitute"
)

iface.launch()
