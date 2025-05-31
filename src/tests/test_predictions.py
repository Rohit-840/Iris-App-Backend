import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_predict_species_setosa():
    response = client.post("/predict", json={
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    })
    assert response.status_code == 200
    assert response.json() == {"species": "Iris Setosa"}

def test_predict_species_versicolor():
    response = client.post("/predict", json={
        "sepal_length": 5.9,
        "sepal_width": 3.0,
        "petal_length": 4.2,
        "petal_width": 1.5
    })
    assert response.status_code == 200
    assert response.json() == {"species": "Iris Versicolor"}

def test_predict_species_virginica():
    response = client.post("/predict", json={
        "sepal_length": 6.5,
        "sepal_width": 3.0,
        "petal_length": 5.2,
        "petal_width": 2.0
    })
    assert response.status_code == 200
    assert response.json() == {"species": "Iris Virginica"}

def test_predict_invalid_data():
    response = client.post("/predict", json={
        "sepal_length": "invalid",
        "sepal_width": 3.0,
        "petal_length": 5.2,
        "petal_width": 2.0
    })
    assert response.status_code == 422  # Unprocessable Entity