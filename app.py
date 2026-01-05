import pandas as pd
df = pd.read_excel("C:/Users/pkitt/OneDrive/Desktop/Family Based Oil blend - Copy.xlsx")
df.isnull().sum()
df.head(10)
df.shape
print(df.isna().sum())
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# 1. Load data
df = pd.read_excel("C:/Users/pkitt/OneDrive/Desktop/Family Based Oil blend - Copy.xlsx")
df.columns = df.columns.str.strip()

# 2. Define features
FEATURE_COLUMNS = [
    "FamilySize",
    "AgeMix",
    "CardioHistory",
    "CookingTemp",
    "CookingStyle",
    "Usage"
]

# 3. Prepare X
X = df[FEATURE_COLUMNS].copy()

encoder_dict = {}   # ✅ CREATED HERE

for col in X.select_dtypes(include="object"):
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoder_dict[col] = le

# 4. Prepare y
y_raw = df["OilBlend"]
y_encoder = LabelEncoder()
y_encoded = y_encoder.fit_transform(y_raw)

# 5. Train model
model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X, y_encoded)

model.predict([[1,0,0,0,2,3]])
# Assertions (NOW VALID)
assert isinstance(encoder_dict, dict)
assert "FamilySize" in encoder_dict

# Save artifacts
joblib.dump(model, "model.pkl")
joblib.dump(encoder_dict, "encoders.pkl")
joblib.dump(y_encoder, "target_encoder.pkl")
joblib.dump(FEATURE_COLUMNS, "feature_columns.pkl")

print("SUCCESS")
print("Encoders:", encoder_dict.keys())

print(type(encoders))        # dict
print(encoders.keys())      # feature names

print(type(target_encoder)) # LabelEncoder
print(target_encoder.classes_)  # oil blend names

# RF class index → Blend name
OIL_MAP = {
    0: "Adult Family Blend",
    1: "Family Balanced",
    2: "Family Heart Care",
    3: "High-Heat Family"
}

# Blend → Customer-friendly differentiator
OIL_REASON_MAP = {
    "Adult Family Blend": "Supports an active lifestyle,Ideal for everyday home cooking",
    "Family Balanced": "Suitable for daily Indian cooking",
    "Family Heart Care": "Designed for heart-conscious families,Use “heart-conscious”, NOT “heart-protective”",
    "High-Heat Family": "High smoke point blend,Stable for Indian frying,Suitable for repeated heating"
}

def explain_choice(recommended_blend):
    reason = OIL_REASON_MAP.get(recommended_blend, "Balanced nutrition profile")

    return (
        f"### 🫒 Recommended Blend: **{recommended_blend}**\n\n"
        f"**Why this blend?**  \n"
        f"{reason}"
    )

df = pd.DataFrame([{
    "FamilySize": 0,
    "AgeMix": 1,
    "CardioHistory": 1,
    "CookingTemp": 0,
    "CookingStyle": 0,
    "Usage": 0,
}])

print(model.predict(df))


def predict_oil(
    FamilySize, AgeMix, CardioHistory, CookingTemp, CookingStyle, Usage
):
    input_data = {
        "FamilySize": FamilySize,
        "AgeMix": AgeMix,
        "CardioHistory": CardioHistory,
        "CookingTemp": CookingTemp,
        "CookingStyle": CookingStyle,
        "Usage": Usage
    }

    X = preprocess(input_data)
    prediction = model.predict(X)[0]

    return prediction

# load artifacts
import joblib
import pandas as pd

model = joblib.load("model.pkl")
X_encoders = joblib.load("encoders.pkl")
y_encoder = joblib.load("target_encoder.pkl")
FEATURE_COLUMNS = joblib.load("feature_columns.pkl")

def predict_oil_blend(input_data: dict):
    """
    input_data example:
    {
        "Family Size": "medium",
        "Age Mix": "adult",
        "Cardio History": "mild",
        "Cooking Temp": "high",
        "Cooking Style": "mixed",
        "Usage": "moderate"
    }
    """

    # Encode X
    row = {}
    for col in FEATURE_COLUMNS:
        value = input_data[col]
        row[col] = X_encoders[col].transform([value])[0]

    X_new = pd.DataFrame([row])[FEATURE_COLUMNS]

    # Predict encoded label
    pred_encoded = model.predict(X_new)[0]

    # Decode to original OilBlend
    pred_label = y_encoder.inverse_transform([pred_encoded])[0]

    return pred_label

test_input = {
    "FamilySize": "Medium",
    "AgeMix": "Adult",
    "CardioHistory": "Strong",
    "CookingTemp": "High",
    "CookingStyle": "Mixed",
    "Usage": "Moderate"
}

result = predict_oil_blend(test_input)
print("Recommended Oil Blend:", result)

import gradio as gr
import joblib
import pandas as pd

# Load artifacts
model = joblib.load("model.pkl")
X_encoders = joblib.load("encoders.pkl")
y_encoder = joblib.load("target_encoder.pkl")
FEATURE_COLUMNS = joblib.load("feature_columns.pkl")

# ✅ DEFINE FUNCTION FIRST
def predict_oil_blend_ui(
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
        value = input_data[col]
        row[col] = X_encoders[col].transform([value])[0]

    X_new = pd.DataFrame([row])[FEATURE_COLUMNS]

    pred_encoded = model.predict(X_new)[0]
    pred_label = y_encoder.inverse_transform([pred_encoded])[0]

    return f"✅ Recommended Oil Blend: {pred_label}"

# ✅ USE FUNCTION AFTER DEFINITION
iface = gr.Interface(
    fn=predict_oil_blend_ui,
    inputs=[
        gr.Dropdown(X_encoders["FamilySize"].classes_.tolist(), label="Family Size"),
        gr.Dropdown(X_encoders["AgeMix"].classes_.tolist(), label="Age Mix"),
        gr.Dropdown(X_encoders["CardioHistory"].classes_.tolist(), label="Cardio History"),
        gr.Dropdown(X_encoders["CookingTemp"].classes_.tolist(), label="Cooking Temp"),
        gr.Dropdown(X_encoders["CookingStyle"].classes_.tolist(), label="Cooking Style"),
        gr.Dropdown(X_encoders["Usage"].classes_.tolist(), label="Usage"),
    ],
    outputs=gr.Textbox(label="Oil Recommendation"),
    title="🛢️ Oil Clinic – Family Based Recommendation"
)

iface.launch()

