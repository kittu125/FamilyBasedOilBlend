# schema.py
from dataclasses import dataclass

@dataclass
class InputSchema:
    FamilySize: str
    AgeMix: str
    CardioHistory: str
    CookingTemp: str
    BMI: str
    Usage: str


def validate_input(data: dict, encoders: dict):
    """
    Strict validation:
    - All fields present
    - Values must exist in encoder classes
    """

    required_fields = InputSchema.__annotations__.keys()

    # Check missing fields
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing input field: {field}")

    # Check unseen categories
    for col, value in data.items():
        if col not in encoders:
            raise ValueError(f"No encoder found for column: {col}")

        if value not in encoders[col].classes_:
            raise ValueError(
                f"Invalid value '{value}' for '{col}'. "
                f"Allowed: {list(encoders[col].classes_)}"
            )

    return True
