import os
import joblib
import pandas as pd
import gradio as gr

# =====================================================
# LOAD TRAINED ARTIFACTS (TOP LEVEL — ONCE)
# =====================================================

def safe_load(path, name):
    if not os.path.exists(path):
        raise RuntimeError(f"❌ Missing required file: {name} ({path})")
    return joblib.load(path)

model = safe_load("model.pkl", "Model")
X_encoders = safe_load("encoders.pkl", "Encoders")
y_encoder = safe_load("target_encoder.pkl", "Target Encoder")
FEATURE_COLUMNS = safe_load("feature_columns.pkl", "Feature Columns")

# =====================================================
# VALIDATION & ENCODING HELPERS (NO SILENT FAILURES)
# =====================================================

def validate_input(value, allowed, field_name):
    if value not in allowed:
        raise ValueError(
            f"❌ Invalid value for '{field_name}': '{value}'. "
            f"Allowed values: {list(allowed)}"
        )

def encode_feature(value, encoder, field_name):
    validate_input(value, encoder.classes_, field_name)
    return encoder.transform([value])[0]

# =====================================================
# PREDICTION FUNCTION (LOGIC ONLY)
# =====================================================

def predict_oil_blend_ui(
    family_size,
    age_mix,
    cardio_history,
    cooking_temp,
    cooking_style,
    usage
):
    try:
        # -------------------------------------------------
        # CRITICAL SAFETY OVERRIDE (EXPLICIT BUSINESS RULE)
        # -------------------------------------------------
        if cardio_history == "strong":
            return (
                "🫒 Recommended Oil Blend: Heart-Safe Blend\n\n"
                "⚠️ Cardiac history detected. "
                "This recommendation prioritizes heart safety."
            )

        # -------------------------------------------------
        # STRICT ENCODING (NO FALLBACKS)
        # -------------------------------------------------
        X = pd.DataFrame([{
            "FamilySize": encode_feature(
                family_size, X_encoders["FamilySize"], "FamilySize"
            ),
            "AgeMix": encode_feature(
                age_mix, X_encoders["AgeMix"], "AgeMix"
            ),
            "CardioHistory": encode_feature(
                cardio_history, X_encoders["CardioHistory"], "CardioHistory"
            ),
            "CookingTemp": encode_feature(
                cooking_temp, X_encoders["CookingTemp"], "CookingTemp"
            ),
            "CookingStyle": encode_feature(
                cooking_style, X_encoders["CookingStyle"], "CookingStyle"
            ),
            "Usage": encode_feature(
                usage, X_encoders["Usage"], "Usage"
            ),
        }])[FEATURE_COLUMNS]

        pred = model.predict(X)[0]

        # -------------------------------------------------
        # SAFE DECODE (NO classes_[0])
        # -------------------------------------------------
        validate_input(
            pred, range(len(y_encoder.classes_)), "Model Output"
        )

        return f"🫒 Recommended Oil Blend: {y_encoder.inverse_transform([pred])[0]}"

    except ValueError as e:
        # User-visible validation error
        return str(e)

    except Exception as e:
        # User-visible system error
        return f"🚨 System error: {str(e)}"

# =====================================================
# GRADIO UI
# =====================================================

with gr.Blocks() as demo:
    gr.Markdown("## 🫒 Oil Blend Recommendation System")

    family_size = gr.Dropdown(
        choices=list(X_encoders["FamilySize"].classes_),
        label="Family Size"
    )
    age_mix = gr.Dropdown(
        choices=list(X_encoders["AgeMix"].classes_),
        label="Age Mix"
    )
    cardio_history = gr.Dropdown(
        choices=list(X_encoders["CardioHistory"].classes_),
        label="Cardio History"
    )
    cooking_temp = gr.Dropdown(
        choices=list(X_encoders["CookingTemp"].classes_),
        label="Cooking Temperature"
    )
    cooking_style = gr.Dropdown(
        choices=list(X_encoders["CookingStyle"].classes_),
        label="Cooking Style"
    )
    usage = gr.Dropdown(
        choices=list(X_encoders["Usage"].classes_),
        label="Usage"
    )

    output = gr.Textbox(
        label="Recommended Oil Blend",
        lines=4
    )

    gr.Button("Recommend").click(
        fn=predict_oil_blend_ui,
        inputs=[
            family_size,
            age_mix,
            cardio_history,
            cooking_temp,
            cooking_style,
            usage
        ],
        outputs=output
    )

# =====================================================
# LAUNCH (MANDATORY FOR HF SPACES)
# =====================================================

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True  # ensures errors are visible to user
    )
