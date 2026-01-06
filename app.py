def predict_oil_blend_ui(
    family_size,
    age_mix,
    cardio_history,
    cooking_temp,
    cooking_style,
    usage
):
     ----------------------------
# Load trained artifacts
# ----------------------------
model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")
y_encoder = joblib.load("target_encoder.pkl")
FEATURE_COLUMNS = joblib.load("feature_columns.pkl")

# ----------------------------
    try:
        # ----------------------------
        # CRITICAL SAFETY OVERRIDE
        # ----------------------------
        if cardio_history == "strong":
            return (
                "🫒 Recommended Oil Blend: Heart-Safe Blend\n\n"
                "⚠️ Cardiac history detected. "
                "This recommendation prioritizes heart safety."
            )

        # ----------------------------
        # STRICT ENCODING (NO FALLBACK)
        # ----------------------------
        X = pd.DataFrame([{
            "FamilySize": encode_feature(family_size, X_encoders["FamilySize"], "FamilySize"),
            "AgeMix": encode_feature(age_mix, X_encoders["AgeMix"], "AgeMix"),
            "CardioHistory": encode_feature(cardio_history, X_encoders["CardioHistory"], "CardioHistory"),
            "CookingTemp": encode_feature(cooking_temp, X_encoders["CookingTemp"], "CookingTemp"),
            "CookingStyle": encode_feature(cooking_style, X_encoders["CookingStyle"], "CookingStyle"),
            "Usage": encode_feature(usage, X_encoders["Usage"], "Usage")
        }])[FEATURE_COLUMNS]

        pred = model.predict(X)[0]

        # ----------------------------
        # SAFE DECODE
        # ----------------------------
        validate_input(pred, range(len(y_encoder.classes_)), "Model Output")
        return f"🫒 Recommended Oil Blend: {y_encoder.inverse_transform([pred])[0]}"

    except ValueError as e:
        # 👇 USER SEES EXACT ERROR
        return str(e)

    except Exception as e:
        # 👇 SYSTEM ERROR — STILL VISIBLE
        return f"🚨 System error: {str(e)}"
