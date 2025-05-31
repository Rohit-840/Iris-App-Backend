from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
from pathlib import Path
import joblib
import logging

router = APIRouter()
logging.basicConfig(level=logging.INFO)

# 1) Compute project root (…/iris-fastapi)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# 2) Look for any .pkl whose name contains "DTC" under src/models (or fallback to the root)
model_dir = PROJECT_ROOT / "src" / "models"
candidates = list(model_dir.glob("*DTC*.pkl"))
if not candidates:
    candidates = list(PROJECT_ROOT.glob("*DTC*.pkl"))

if not candidates:
    raise RuntimeError(f"No DTC_model.pkl found under {model_dir} or {PROJECT_ROOT}")

MODEL_PATH = candidates[0]
logging.info(f"Loading model from {MODEL_PATH}")

# 3) Load with joblib
try:
    model = joblib.load(str(MODEL_PATH))
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# sanity check
if not hasattr(model, "predict"):
    raise RuntimeError(f"Loaded object is a {type(model)}, not a classifier!")

# Optional: base path for your species images
IMAGES_DIR = PROJECT_ROOT / "src" / "Images"
# species_image_map = {
#     "Setosa": IMAGES_DIR / "setosa.webp",
#     "Versicolor": IMAGES_DIR / "versicolor.webp",
#     "Virginica": IMAGES_DIR / "virginica.webp",
# }

# 3) Public image URLs for each species
image_url_map = {
    "Setosa":      "https://upload.wikimedia.org/wikipedia/commons/a/a7/Irissetosa1.jpg",
    "Versicolor":  "https://upload.wikimedia.org/wikipedia/commons/2/27/Blue_Flag%2C_Ottawa.jpg",
    "Virginica":   "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg",
}

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width:  float
    petal_length: float
    petal_width:  float

@router.post("/predict")
def predict_iris(iris: IrisInput):
    arr = np.array([[iris.sepal_length,
                    iris.sepal_width,
                    iris.petal_length,
                    iris.petal_width]],
                dtype=np.float64)
    try:
        raw_pred = model.predict(arr)[0]

        print(raw_pred)

        # If it’s numeric, map 0/1/2 → name
        if isinstance(raw_pred, (int, np.integer, float, np.floating)):
            names = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
            species = names.get(int(raw_pred), "Unknown")

        # If it’s a string like "Iris-setosa" or "Iris_setosa"
        elif isinstance(raw_pred, str):
            # remove any prefix up to a hyphen or underscore, then title-case
            species = raw_pred.split("-", 1)[-1].split("_", 1)[-1].capitalize()

        else:
            species = str(raw_pred)

        # build response
        response = {"species": species}

        # # (Optional) include image URL if it exists
        # img_path = species_image_map.get(species)
        # if img_path and img_path.is_file():
        #     response["image_path"] = str(img_path)

        # attach a public image URL if we have one
        if species in image_url_map:
            response["image_url"] = image_url_map[species]

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))