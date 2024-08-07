import pytest
import json
from app import app


@pytest.fixture
def client():
    app.config['TESTING'] = True

    with app.test_client() as client:
        yield client


def test_predict(client):
    test_data = {'features': [0, 80, 0, 1, 3, 25, 6, 140]}
    response = client.post('/predict', data=json.dumps(test_data),
                           content_type='application/json')
    assert response.status_code == 200
    response_data = response.get_json()
    assert 'prediction' in response_data
    assert isinstance(response_data['prediction'], int)
