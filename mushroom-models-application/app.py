"""
Tkinter UI for the Mushroom Classification project.

This version is designed to interface with the modular project models saved in:

    outputs/models/random_forest_top7_features.joblib
    outputs/models/decision_tree_top7_features.joblib

Those saved models are full sklearn Pipelines, meaning they already include:

    preprocessing + one-hot encoding + trained classifier

Therefore, this UI should pass raw mushroom category codes like "n", "b", "w", etc.
directly to the model. It should NOT use LabelEncoder or encoders.pkl.

To switch models, change MODEL_NAME below.
"""

from pathlib import Path
import tkinter as tk
from tkinter import ttk

import joblib
import pandas as pd


# -----------------------------------------------------------------------------
# Model selection
# -----------------------------------------------------------------------------
# Choose one of these model names:
#
#   "random_forest_top7_features"  -> recommended for the app because it is an ensemble model
#   "decision_tree_top7_features"  -> recommended for explainability and presentation diagrams
#
# To switch models, change only this line:
MODEL_NAME = "random_forest_top7_features"
# MODEL_NAME = "decision_tree_top7_features"


# -----------------------------------------------------------------------------
# Project path handling
# -----------------------------------------------------------------------------
def find_project_root(start_path: Path) -> Path:
    """
    Find the project root by walking upward until an outputs/models folder exists.

    This lets the UI work whether ui.py is located in the project root or in a
    subfolder such as app/ui.py.
    """
    start_path = start_path.resolve()

    for candidate in [start_path.parent, *start_path.parents]:
        if (candidate / "outputs" / "models").exists():
            return candidate

    # Fallback: assume the UI file is in the project root.
    return start_path.parent


PROJECT_ROOT = find_project_root(Path(__file__))
MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / f"{MODEL_NAME}.joblib"


# Load the selected sklearn Pipeline.
# This pipeline performs preprocessing automatically, so no manual encoding is needed.
try:
    model = joblib.load(MODEL_PATH)
    model_load_error = None
except Exception as exc:
    model = None
    model_load_error = exc


# -----------------------------------------------------------------------------
# Features expected by the Top 7 models
# -----------------------------------------------------------------------------
# These are the seven top-ranked features used by both:
#   decision_tree_top7_features
#   random_forest_top7_features
#
# The UI sends these exact column names to the saved sklearn Pipeline.
featureColumns = [
    "odor",
    "gill-size",
    "spore-print-color",
    "ring-type",
    "stalk-surface-above-ring",
    "stalk-surface-below-ring",
    "gill-color",
]


# -----------------------------------------------------------------------------
# Human-readable dropdown labels mapped to original dataset category codes
# -----------------------------------------------------------------------------
# The saved project models expect original mushroom category codes, not full words.
# Example: odor "none" maps to "n".
fieldMaps = {
    "odor": {
        "almond": "a",
        "anise": "l",
        "creosote": "c",
        "fishy": "y",
        "foul": "f",
        "musty": "m",
        "none": "n",
        "pungent": "p",
        "spicy": "s",
    },
    "gill-size": {
        "broad": "b",
        "narrow": "n",
    },
    "spore-print-color": {
        "black": "k",
        "brown": "n",
        "buff": "b",
        "chocolate": "h",
        "green": "r",
        "orange": "o",
        "purple": "u",
        "white": "w",
        "yellow": "y",
    },
    "ring-type": {
        "cobwebby": "c",
        "evanescent": "e",
        "flaring": "f",
        "large": "l",
        "none": "n",
        "pendant": "p",
        "sheathing": "s",
        "zone": "z",
    },
    "stalk-surface-above-ring": {
        "fibrous": "f",
        "scaly": "y",
        "silky": "k",
        "smooth": "s",
    },
    "stalk-surface-below-ring": {
        "fibrous": "f",
        "scaly": "y",
        "silky": "k",
        "smooth": "s",
    },
    "gill-color": {
        "black": "k",
        "brown": "n",
        "buff": "b",
        "chocolate": "h",
        "gray": "g",
        "green": "r",
        "orange": "o",
        "pink": "p",
        "purple": "u",
        "red": "e",
        "white": "w",
        "yellow": "y",
    },
}


prettyLabels = {
    "odor": "Odor",
    "gill-size": "Gill Size",
    "spore-print-color": "Spore Print Color",
    "ring-type": "Ring Type",
    "stalk-surface-above-ring": "Stalk Surface Above Ring",
    "stalk-surface-below-ring": "Stalk Surface Below Ring",
    "gill-color": "Gill Color",
}


# -----------------------------------------------------------------------------
# Tkinter window setup
# -----------------------------------------------------------------------------
window = tk.Tk()
window.title("Mushy Eats")
window.geometry("470x780")
window.minsize(470, 780)
window.configure(bg="#f2f2f2")

style = ttk.Style()
style.theme_use("clam")

style.configure("TFrame", background="#f2f2f2")
style.configure("Card.TFrame", background="white")
style.configure("TLabel", background="#f2f2f2", font=("Arial", 10))
style.configure("Title.TLabel", background="#f2f2f2", font=("Arial", 20, "bold"))
style.configure("TButton", font=("Arial", 11), padding=8)

mainFrame = ttk.Frame(window, padding=20)
mainFrame.pack(fill="both", expand=True)

titleLabel = ttk.Label(
    mainFrame,
    text="Pan Fry or Die?",
    style="Title.TLabel",
)
titleLabel.pack(pady=(0, 5))

subtitleLabel = ttk.Label(
    mainFrame,
    text="Choose mushroom traits below",
)
subtitleLabel.pack(pady=(0, 5))

modelLabel = ttk.Label(
    mainFrame,
    text=f"Selected model: {MODEL_NAME}",
)
modelLabel.pack(pady=(0, 15))

cardFrame = ttk.Frame(
    mainFrame,
    style="Card.TFrame",
    padding=20,
)
cardFrame.pack(fill="x")

inputBoxes = {}

for col in featureColumns:
    label = ttk.Label(
        cardFrame,
        text=prettyLabels[col],
        background="white",
    )
    label.pack(anchor="w", pady=(8, 2))

    values = list(fieldMaps[col].keys())

    comboBox = ttk.Combobox(
        cardFrame,
        values=values,
        state="readonly",
    )
    comboBox.pack(fill="x", pady=(0, 4))

    inputBoxes[col] = comboBox

resultLabel = tk.Label(
    mainFrame,
    text="Prediction will appear here",
    font=("Arial", 13, "bold"),
    bg="#f2f2f2",
)
resultLabel.pack(pady=(20, 5))

confidenceLabel = tk.Label(
    mainFrame,
    text="",
    font=("Arial", 11),
    bg="#f2f2f2",
)
confidenceLabel.pack()

warningLabel = tk.Label(
    mainFrame,
    text=(
        "Educational use only. Do not eat wild mushrooms based on this prediction. "
        "Consult a qualified expert for real-world mushroom identification."
    ),
    font=("Arial", 9),
    bg="#f2f2f2",
    fg="gray",
    wraplength=400,
)
warningLabel.pack(pady=(10, 0))


# -----------------------------------------------------------------------------
# Prediction logic
# -----------------------------------------------------------------------------
def predictMushroom():
    """
    Collect values from the UI, convert human-readable values to dataset codes,
    and pass a one-row DataFrame directly to the saved sklearn Pipeline.
    """
    if model is None:
        resultLabel.config(
            text=f"Model load error: {model_load_error}",
            fg="black",
        )
        confidenceLabel.config(text=f"Expected model at: {MODEL_PATH}")
        return

    inputRow = {}

    for col in featureColumns:
        selected_text = inputBoxes[col].get()

        if selected_text == "":
            resultLabel.config(
                text="Fill out all fields",
                fg="black",
            )
            confidenceLabel.config(text="")
            return

        # Convert readable dropdown value to original dataset code.
        # Example: "none" -> "n"
        inputRow[col] = fieldMaps[col][selected_text]

    try:
        # The saved model is a Pipeline, so it expects the original raw category
        # codes in a DataFrame with the correct column names.
        inputData = pd.DataFrame([inputRow])
        inputData = inputData[featureColumns]

        rawPrediction = model.predict(inputData)[0]

        # Handles both possible model outputs:
        # - old LabelEncoder model: 0 = edible, 1 = poisonous
        # - newer pipeline model: "e" = edible, "p" = poisonous
        predictionMap = {
            0: "e",
            1: "p",
            "0": "e",
            "1": "p",
            "e": "e",
            "p": "p",
        }

        classLabel = predictionMap.get(rawPrediction, rawPrediction)

        # Both RandomForestClassifier and DecisionTreeClassifier support predict_proba.
        # Since model is a Pipeline, model.predict_proba works if the final estimator supports it.
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(inputData)[0]
            confidence = max(probabilities) * 100

            confidenceLabel.config(
                text=f"Confidence: {confidence:.2f}%"
            )
        else:
            confidenceLabel.config(text="")

        if classLabel == "p":
            resultLabel.config(
                text="Likely Poisonous",
                fg="red",
            )
        elif classLabel == "e":
            resultLabel.config(
                text="Likely Edible",
                fg="green",
            )
        else:
            # Fallback in case a future model returns a different label format.
            resultLabel.config(
                text=f"Prediction: {classLabel}",
                fg="black",
            )

    except Exception as exc:
        resultLabel.config(
            text="Error: " + str(exc),
            fg="black",
        )
        confidenceLabel.config(text="")


def clearFields():
    """Clear all dropdowns and reset output labels."""
    for col in featureColumns:
        inputBoxes[col].set("")

    resultLabel.config(
        text="Prediction will appear here",
        fg="black",
    )

    confidenceLabel.config(text="")


buttonFrame = ttk.Frame(mainFrame)
buttonFrame.pack(pady=15)

predictButton = ttk.Button(
    buttonFrame,
    text="Predict",
    command=predictMushroom,
)
predictButton.grid(row=0, column=0, padx=6)

clearButton = ttk.Button(
    buttonFrame,
    text="Clear",
    command=clearFields,
)
clearButton.grid(row=0, column=1, padx=6)


# Show a startup message if the model could not be loaded.
# This keeps the app open and tells you what path it expected.
if model_load_error is not None:
    resultLabel.config(
        text="Model could not be loaded",
        fg="black",
    )
    confidenceLabel.config(
        text=f"Expected: {MODEL_PATH}"
    )


window.mainloop()
