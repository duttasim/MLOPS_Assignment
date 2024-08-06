import pytest
import json
from app import app


@pytest.fixture
def client():
    app.config['TESTING'] = True

    with app.test_client() as client:
        yield client


def test_predict(client):
    test_data = {'features': [3.1, 4.2, 3.0, 3.5]}
    response = client.post('/predict', data=json.dumps(test_data),
                           content_type='application/json')
    assert response.status_code == 200
    response_data = response.get_json()
    assert 'prediction' in response_data
    assert isinstance(response_data['prediction'], int)
