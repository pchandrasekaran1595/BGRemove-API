from fastapi.testclient import TestClient

from main import app, VERSION

client = TestClient(app)

def test_get_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "statusText" : "Root Endpoint of Background Removal/Replacement API",
        "statusCode" : 200,
        "version" : VERSION,
    }


def test_get_version():
    response = client.get("/version")
    assert response.status_code == 200
    assert response.json() == {
        "statusText" : "Background Removal/Replacement API Version Fetch Successful",
        "statusCode" : 200,
        "version" : VERSION,
    }


def test_get_remove_bg():
    response = client.get("/remove")
    assert response.status_code == 200
    assert response.json() == {
        "statusText" : "Background Removal Endpoint",
        "statusCode" : 200,
        "version" : VERSION,
    }


def test_get_replace_bg():
    response = client.get("/replace")
    assert response.status_code == 200
    assert response.json() == {
        "statusText" : "Background Replacement Endpoint",
        "statusCode" : 200,
        "version" : VERSION,
    }


def test_get_remove_bg_li():
    response = client.get("/remove/li")
    assert response.status_code == 200
    assert response.json() == {
        "statusText" : "Background Removal Endpoint (Lightweight Model)",
        "statusCode" : 200,
        "version" : VERSION,
    }


def test_get_replace_bg_li():
    response = client.get("/replace/li")
    assert response.status_code == 200
    assert response.json() == {
        "statusText" : "Background Replacement Endpoint (Lightweight Model)",
        "statusCode" : 200,
        "version" : VERSION,
    }