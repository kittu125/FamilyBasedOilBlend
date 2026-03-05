```python
import os
import glob
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
# LOAD RAG DOCUMENTS
# =====================================================

def load_rag_docs():
    docs = {}
    for file in glob.glob("rag_docs/*.txt"):
        key = os.path.basename(file).replace(".txt", "").lower()
        with open(file, "r", encoding="utf-8") as f:
            docs[key] = f.read()
    return docs

RAG_DOCS = load_rag_docs()

def get_rag_explanation(blend_name):
    key = blend_name.lower().replace(" ", "_")

    if key in RAG_DOCS:
        return RAG_DOCS[key]

    return "General cooking oil blend suitable for everyday cooking."

# =====================================================
# VALIDATION & ENCODING HELPERS
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
# EXPLAINABILITY HELPER (RULE-BASED)
# =====================================================

def build_explanation(inputs):
    reasons = []

    if inputs["CookingTemp"] == "High":
        reasons.append(
            "High-temperature cooking → oils with higher smoke points preferred"
        )
    elif inputs["CookingTemp"] == "Medium":
        reasons.append(
            "Medium-temperature cooking → balanced heat-stable oils selected"
        )

    if inputs["BMI"] == "High":
        reasons.append(
            "Higher BMI → balanced fatty acid oils recommended for everyday cooking"
        )
    elif inputs["BMI"] == "Medium":
        reasons.append(
            "Moderate BMI → maintaining balanced dietary fat intake is beneficial"
        )

    if inputs["GutWellness"] == "Good":
        reasons.append(
            "Healthy digestion → maintaining balanced cooking oils supports gut wellness"
        )
    elif inputs["GutWellness"] == "Need Support":
        reasons.append(
            "Digestive support needed → oils with natural bioactive compounds may help gut balance"
        )

    if inputs["Usage"] == "High":
        reasons.append(
            "Frequent oil usage → balanced fatty-acid profile recommended"
        )
    elif inputs["Usage"] == "Moderate":
        reasons.append(
            "Moderate oil usage → everyday family-friendly oil preferred"
        )

    if inputs["AgeMix"] in ["Adult/Elder", "multi-age"]:
        reasons.append(
            "Presence of elders → heart-conscious oil composition prioritized"
        )
    elif inputs["AgeMix"] == "Adult":
        reasons.append(
            "Adult family profile → long-term cooking stability considered"
        )

    if not reasons:
        reasons.append(
            "Overall family cooking pattern best aligns with this oil blend"
        )

    return reasons[:2]

# =====================================================
# PREDICTION FUNCTION
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

        # --------------------------------------------
        # SAFETY OVERRIDE
        # --------------------------------------------
        if cardio_history == "Strong":
            return (
                "🫒 Recommended Oil Blend: Heart-Safe Blend\n\n"
                "⚠️ Strong cardiac history detected.\n"
                "Safety-first medical rule overrides ML prediction."
            )

        # --------------------------------------------
        # ENCODE INPUT FEATURES
        # --------------------------------------------
        X = pd.DataFrame([{
            "FamilySize": encode_feature(family_size, X_encoders["FamilySize"], "FamilySize"),
            "AgeMix": encode_feature(age_mix, X_encoders["AgeMix"], "AgeMix"),
            "CardioHistory": encode_feature(cardio_history, X_encoders["CardioHistory"], "CardioHistory"),
            "CookingTemp": encode_feature(cooking_temp, X_encoders["CookingTemp"], "CookingTemp"),
            "BMI": encode_feature(BMI, X_encoders["BMI"], "BMI"),
            "GutWellness": encode_feature(gut_wellness, X_encoders["GutWellness"], "GutWellness"),
            "Usage": encode_feature(usage, X_encoders["Usage"], "Usage"),
        }])[FEATURE_COLUMNS]

        # --------------------------------------------
        # MODEL PREDICTION
        # --------------------------------------------
        pred = model.predict(X)[0]

        probs = model.predict_proba(X)[0]
        confidence = round(max(probs) * 100, 1)

        blend_name = y_encoder.inverse_transform([pred])[0]

        # --------------------------------------------
        # RULE-BASED EXPLANATION
        # --------------------------------------------
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
        explanation_text = "\n".join([f"• {r}" for r in explanations])

        # --------------------------------------------
        # RAG KNOWLEDGE
        # --------------------------------------------
        rag_text = get_rag_explanation(blend_name)

        # --------------------------------------------
        # FINAL RESPONSE
        # --------------------------------------------
        return (
            f"🫒 Recommended Oil Blend: {blend_name}\n\n"
            f"📊 Confidence: {confidence}%\n\n"
            f"🔍 Why this recommendation?\n"
            f"{explanation_text}\n\n"
            f"{rag_text}"
        )

    except ValueError as e:
        return str(e)

    except Exception as e:
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

    BMI = gr.Dropdown(
        choices=list(X_encoders["BMI"].classes_),
        label="BMI"
    )

    gut_wellness = gr.Dropdown(
        choices=list(X_encoders["GutWellness"].classes_),
        label="Gut Wellness"
    )

    usage = gr.Dropdown(
        choices=list(X_encoders["Usage"].classes_),
        label="Usage"
    )

    output = gr.Textbox(
        label="Recommended Oil Blend",
        lines=12
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
# LAUNCH FOR HUGGING FACE SPACES
# =====================================================

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
