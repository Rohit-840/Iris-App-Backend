from pydantic import BaseModel
from typing import List

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class IrisResponse(BaseModel):
    species: str
    probability: float

class IrisSpeciesInfo(BaseModel):
    name: str
    regions: List[str]
    climate: str
    temperature: str
    habitat: str
    altitude: str
    characteristics: str