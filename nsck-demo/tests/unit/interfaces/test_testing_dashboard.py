"""
Test Testing Dashboard API
Verifies: All endpoints for the comprehensive testing dashboard including
chat, game simulations, monitoring, and export functionality.
"""
import sys
import os
import json


from python.interfaces.testing_dashboard import app


def _client():
    """Create a Flask test client."""
    app.config["TESTING"] = True
    return app.test_client()


# ---------------------------------------------------------------------------
# Dashboard Home
# ---------------------------------------------------------------------------

def test_dashboard_home():
    """Test that the testing dashboard page loads."""
    client = _client()
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"NSCK Testing Dashboard" in resp.data
    assert b"Chat &amp; Test" in resp.data or b"Chat" in resp.data
    assert b"Game Simulations" in resp.data
    assert b"System Monitor" in resp.data


# ---------------------------------------------------------------------------
# Chat API
# ---------------------------------------------------------------------------

def test_chat_message():
    """Test sending a chat message."""
    client = _client()
    resp = client.post("/api/chat",
        data=json.dumps({"message": "What is a rose?"}),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert "answer" in data
    assert "confidence" in data
    assert "reasoning_trace" in data
    assert "emotion" in data
    assert data["confidence"] >= 0


def test_chat_with_sample_text():
    """Test chat with a sample text provided."""
    client = _client()
    resp = client.post("/api/chat",
        data=json.dumps({
            "message": "What is the main topic?",
            "sample_text": "The red rose blooms in the garden."
        }),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert "answer" in data
    assert "reasoning_trace" in data
    assert len(data["reasoning_trace"]) > 0


def test_chat_empty_rejected():
    """Test that empty chat messages are rejected."""
    client = _client()
    resp = client.post("/api/chat",
        data=json.dumps({"message": ""}),
        content_type="application/json",
    )
    assert resp.status_code == 400


def test_chat_history():
    """Test retrieving chat history."""
    client = _client()
    # Send a message first
    client.post("/api/chat",
        data=json.dumps({"message": "Hello"}),
        content_type="application/json",
    )
    resp = client.get("/api/chat/history")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "history" in data
    assert len(data["history"]) > 0


# ---------------------------------------------------------------------------
# Game Simulation API
# ---------------------------------------------------------------------------

def test_game_start_snake():
    """Test starting a snake game."""
    client = _client()
    resp = client.post("/api/game/start",
        data=json.dumps({"game_type": "snake", "auto_play": False}),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert "session_id" in data
    assert data["game_type"] == "snake"
    assert "state" in data


def test_game_start_pong():
    """Test starting a pong game."""
    client = _client()
    resp = client.post("/api/game/start",
        data=json.dumps({"game_type": "pong", "auto_play": False}),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["game_type"] == "pong"


def test_game_start_maze():
    """Test starting a maze game."""
    client = _client()
    resp = client.post("/api/game/start",
        data=json.dumps({"game_type": "maze", "auto_play": False}),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["game_type"] == "maze"


def test_game_step():
    """Test stepping a game simulation."""
    client = _client()
    # Start a game
    start_resp = client.post("/api/game/start",
        data=json.dumps({"game_type": "snake", "auto_play": False}),
        content_type="application/json",
    )
    session_id = start_resp.get_json()["session_id"]

    # Step it
    resp = client.post("/api/game/step",
        data=json.dumps({"session_id": session_id}),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert "action" in data
    assert "reward" in data
    assert "score" in data
    assert data["step"] == 1


def test_game_multiple_steps():
    """Test running multiple game steps."""
    client = _client()
    start_resp = client.post("/api/game/start",
        data=json.dumps({"game_type": "snake", "auto_play": False}),
        content_type="application/json",
    )
    session_id = start_resp.get_json()["session_id"]

    for i in range(5):
        resp = client.post("/api/game/step",
            data=json.dumps({"session_id": session_id}),
            content_type="application/json",
        )
        data = resp.get_json()
        assert data["step"] == i + 1


def test_game_state():
    """Test getting game state."""
    client = _client()
    start_resp = client.post("/api/game/start",
        data=json.dumps({"game_type": "pong", "auto_play": False}),
        content_type="application/json",
    )
    session_id = start_resp.get_json()["session_id"]

    resp = client.get(f"/api/game/state?session_id={session_id}")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "state" in data


def test_game_sessions_list():
    """Test listing game sessions."""
    client = _client()
    resp = client.get("/api/game/sessions")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "sessions" in data


def test_game_stop():
    """Test stopping a game."""
    client = _client()
    start_resp = client.post("/api/game/start",
        data=json.dumps({"game_type": "snake", "auto_play": False}),
        content_type="application/json",
    )
    session_id = start_resp.get_json()["session_id"]

    resp = client.post("/api/game/stop",
        data=json.dumps({"session_id": session_id}),
        content_type="application/json",
    )
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "stopped"


def test_game_invalid_type():
    """Test starting an invalid game type."""
    client = _client()
    resp = client.post("/api/game/start",
        data=json.dumps({"game_type": "invalid_game"}),
        content_type="application/json",
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Monitoring API
# ---------------------------------------------------------------------------

def test_monitor_endpoint():
    """Test the full monitoring endpoint."""
    client = _client()
    resp = client.get("/api/monitor")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "emotion" in data
    assert "self_model" in data
    assert "system_stats" in data
    assert "current" in data["emotion"]
    assert "valence" in data["emotion"]
    assert "arousal" in data["emotion"]
    assert "blend" in data["emotion"]
    assert "mood" in data["emotion"]


def test_stats_endpoint():
    """Test the stats endpoint."""
    client = _client()
    resp = client.get("/api/stats")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "queries_processed" in data


def test_logs_endpoint():
    """Test the logs endpoint."""
    client = _client()
    resp = client.get("/api/logs")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "logs" in data


def test_knowledge_endpoint():
    """Test the knowledge base endpoint."""
    client = _client()
    resp = client.get("/api/knowledge")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "knowledge" in data


def test_context_stats():
    """Test context engine stats endpoint."""
    client = _client()
    resp = client.get("/api/context/stats")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "registered_concepts" in data


# ---------------------------------------------------------------------------
# Export API
# ---------------------------------------------------------------------------

def test_export_text():
    """Test full text export."""
    client = _client()
    # Generate some data first
    client.post("/api/chat",
        data=json.dumps({"message": "Hello world"}),
        content_type="application/json",
    )
    resp = client.get("/api/export/text")
    assert resp.status_code == 200
    assert resp.content_type == "text/plain; charset=utf-8"
    text = resp.data.decode("utf-8")
    assert "NSCK COGNITIVE SYSTEM" in text
    assert "SYSTEM STATISTICS" in text
    assert "EMOTIONAL STATE" in text
    assert "KNOWLEDGE BASE" in text
    assert "CHAT HISTORY" in text
    assert "ACTIVITY LOG" in text
    assert "END OF EXPORT" in text


def test_export_json():
    """Test JSON export."""
    client = _client()
    resp = client.get("/api/export/json")
    assert resp.status_code == 200
    assert resp.content_type == "application/json"
    data = json.loads(resp.data)
    assert "exported_at" in data
    assert "system_stats" in data
    assert "knowledge_base" in data
    assert "emotion" in data
    assert "chat_history" in data
    assert "activity_log" in data


# ---------------------------------------------------------------------------
# Correction and Feedback
# ---------------------------------------------------------------------------

def test_correction_endpoint():
    """Test knowledge correction."""
    client = _client()
    resp = client.post("/api/correct",
        data=json.dumps({
            "concept": "fire",
            "relation": "causes",
            "wrong_target": "danger",
            "correct_target": "heat_and_light",
        }),
        content_type="application/json",
    )
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


def test_feedback_endpoint():
    """Test reward feedback."""
    client = _client()
    resp = client.post("/api/feedback",
        data=json.dumps({
            "task_tag": "general",
            "reward": 1.0,
            "concepts": ["test"],
        }),
        content_type="application/json",
    )
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Process endpoint (multimodal)
# ---------------------------------------------------------------------------

def test_process_text():
    """Test processing text through the cognitive pipeline."""
    client = _client()
    resp = client.post("/api/process", data={
        "text": "The red rose is beautiful",
        "task_tag": "general",
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert "answer" in data
    assert "reasoning_trace" in data
    assert "confidence" in data


# ---------------------------------------------------------------------------
# Integration: Chat + Game + Monitor
# ---------------------------------------------------------------------------

def test_full_integration_flow():
    """Test a complete flow: chat, start game, step, monitor, export."""
    client = _client()

    # 1. Chat
    resp = client.post("/api/chat",
        data=json.dumps({"message": "Tell me about learning"}),
        content_type="application/json",
    )
    assert resp.status_code == 200

    # 2. Start game
    resp = client.post("/api/game/start",
        data=json.dumps({"game_type": "snake", "auto_play": False}),
        content_type="application/json",
    )
    session_id = resp.get_json()["session_id"]

    # 3. Step game
    for _ in range(3):
        resp = client.post("/api/game/step",
            data=json.dumps({"session_id": session_id}),
            content_type="application/json",
        )
        assert resp.status_code == 200

    # 4. Monitor
    resp = client.get("/api/monitor")
    assert resp.status_code == 200
    mon = resp.get_json()
    assert "snake" in mon["self_model"]

    # 5. Export
    resp = client.get("/api/export/text")
    assert resp.status_code == 200
    text = resp.data.decode("utf-8")
    assert "GAME SESSIONS" in text
    assert "snake" in text.lower()


if __name__ == "__main__":
    test_dashboard_home()
    test_chat_message()
    test_chat_with_sample_text()
    test_chat_empty_rejected()
    test_chat_history()
    test_game_start_snake()
    test_game_start_pong()
    test_game_start_maze()
    test_game_step()
    test_game_multiple_steps()
    test_game_state()
    test_game_sessions_list()
    test_game_stop()
    test_game_invalid_type()
    test_monitor_endpoint()
    test_stats_endpoint()
    test_logs_endpoint()
    test_knowledge_endpoint()
    test_context_stats()
    test_export_text()
    test_export_json()
    test_correction_endpoint()
    test_feedback_endpoint()
    test_process_text()
    test_full_integration_flow()
    print("\n🎉 All Testing Dashboard Tests Passed!")
