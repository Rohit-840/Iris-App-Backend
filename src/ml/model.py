import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier

class IrisModel:
    def __init__(self, model_path: str):
        """Initialize model with path to pickle file"""
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: str) -> DecisionTreeClassifier:
        """Load the pickled model file"""
        try:
            with open(model_path, 'rb') as file:
                model = pickle.load(file)
                if not hasattr(model, 'predict'):
                    raise AttributeError("Loaded model does not have predict method")
                return model
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")

    def predict(self, features: np.ndarray) -> int:
        """Make prediction using loaded model"""
        try:
            prediction = self.model.predict(features)
            return int(prediction[0])
        except Exception as e:
            raise Exception(f"Prediction error: {str(e)}")

def load_model(model_path: str) -> IrisModel:
    """Helper function to load model"""
    return IrisModel(model_path)