from fastapi.testclient import TestClient
import main

def test_health():
    response = TestClient(main.app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_memories_endpoint():
    response = TestClient(main.app).get("/api/users/no_such_user/memories")
    assert response.status_code == 200
    assert response.json()["user_id"] == "no_such_user"
