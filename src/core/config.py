import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    MODEL_PATH = os.getenv("MODEL_PATH", "DTC_model.pkl")
    API_V1_STR = "/api/v1"