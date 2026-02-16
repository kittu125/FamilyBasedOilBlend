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
    BMI,
    gut_wellness,
    usage
):
    try:
        # -------------------------------------------------
        # CRITICAL SAFETY OVERRIDE
        # -------------------------------------------------
        if cardio_history == "Strong":
            return (
                "🫒 Recommended Oil Blend: Heart-Safe Blend\n\n"
                "🔍 Why this recommendation?\n"
                "• Strong cardiac history detected\n"
                "• Safety-first medical rule overrides ML prediction\n\n"
                "⚠️ This recommendation prioritizes heart health."
            )

        # -------------------------------------------------
        # STRICT ENCODING
        # -------------------------------------------------
        X = pd.DataFrame([{
            "FamilySize": encode_feature(family_size, X_encoders["FamilySize"], "FamilySize"),
            "AgeMix": encode_feature(age_mix, X_encoders["AgeMix"], "AgeMix"),
            "CardioHistory": encode_feature(cardio_history, X_encoders["CardioHistory"], "CardioHistory"),
            "CookingTemp": encode_feature(cooking_temp, X_encoders["CookingTemp"], "CookingTemp"),
            "BMI": encode_feature(BMI, X_encoders["BMI"], "BMI"),
            "GutWellness": encode_feature(gut_wellness, X_encoders["GutWellness"], "GutWellness"),
            "Usage": encode_feature(usage, X_encoders["Usage"], "Usage"),
        }])[FEATURE_COLUMNS]

        # -------------------------------------------------
        # MODEL PREDICTION
        # -------------------------------------------------
        pred = model.predict(X)[0]

        validate_input(
            pred, range(len(y_encoder.classes_)), "Model Output"
        )

        # -------------------------------------------------
        # EXPLAINABILITY
        # -------------------------------------------------
        inputs = {
            "FamilySize": family_size,
            "AgeMix": age_mix,
            "CardioHistory": cardio_history,
            "CookingTemp": cooking_temp,
            "BMI": BMI,
            "GutWellness": gut_wellness,
            "Usage": usage
        }

        explanations = build_explanation(inputs)
        explanation_text = "\n".join(
            [f"• {reason}" for reason in explanations]
        )

        return (
            f"🫒 Recommended Oil Blend: "
            f"{y_encoder.inverse_transform([pred])[0]}\n\n"
            f"🔍 Why this recommendation?\n"
            f"{explanation_text}"
        )

    except ValueError as e:
        return str(e)

    except Exception as e:
        return f"🚨 System error: {str(e)}"

# =====================================================
# EXPLAINABILITY HELPER (RULE-BASED, HUMAN READABLE)
# =====================================================

def build_explanation(inputs):
    reasons = []

    # Heat-based reasoning
    if inputs["CookingTemp"] == "High":
        reasons.append(
            "High-temperature cooking → oils with higher smoke point preferred"
        )
    elif inputs["CookingTemp"] == "Medium":
        reasons.append(
            "Medium-temperature cooking → balanced heat-stable oils selected"
        )

    # BMI reasoning
    if inputs["BMI"] == "High":
        reasons.append(
            "Stimulate the Production of Nitrix Oxide for shutting down SCoR2"
        )
    elif inputs["BMI"] == "Medium":
        reasons.append(
            "Continue active Lifestyle like excerice and good sleep"
        )
    # Gut_Wellness reasoning
    if inputs["GutWellness"] == "Good":
        reasons.append(
            "Looks Having the needed food and requirements for maintanence of Gut"
        )
    elif inputs["GutWellness"] == "Need Support":
        reasons.append(
            "Need to Provide Gocomole Oil helps in stimulation & support for Bowl Movement& also stimulate required Gut Bacteria "
        )    

    # Usage reasoning
    if inputs["Usage"] == "High":
        reasons.append(
            "High daily usage → balanced fatty-acid profile recommended"
        )
    elif inputs["Usage"] == "Moderate":
        reasons.append(
            "Moderate daily usage → everyday family-friendly oil preferred"
        )

    # Age-based reasoning
    if inputs["AgeMix"] in ["Adult/Elder", "multi-age"]:
        reasons.append(
            "Presence of elders → heart-friendly fat composition prioritized"
        )
    elif inputs["AgeMix"] == "Adult":
        reasons.append(
            "Adult family profile → long-term heart maintenance considered"
        )

    # Final safety net (rare now)
    if not reasons:
        reasons.append(
            "Overall family cooking pattern best aligns with this oil blend"
        )

    return reasons[:2]



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
    BMI = gr.Dropdown(
        choices=list(X_encoders["BMI"].classes_),
        label="BMI"
    )
    GutWellness = gr.Dropdown(
        choices=list(X_encoders["GutWellness"].classes_),
        label="Gut Wellness"
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
            BMI,
            gut_wellness,
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