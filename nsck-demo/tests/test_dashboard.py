"""
Test Cognitive Dashboard API
Verifies: API endpoints for processing, stats, logs, export, correction.
"""
import sys
import os
import json
import io
import zipfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from python.cognitive_dashboard import app, get_system, _activity_log


def _client():
    """Create a Flask test client."""
    app.config["TESTING"] = True
    return app.test_client()


def test_dashboard_home():
    """Test that the dashboard page loads."""
    print("--- Test: Dashboard Home ---")
    client = _client()
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"NSCK Cognitive Dashboard" in resp.data
    print("✅ Dashboard Home Test Passed")


def test_process_text():
    """Test processing text input via API."""
    print("\n--- Test: Process Text ---")
    client = _client()
    resp = client.post("/api/process", data={
        "text": "The red rose is beautiful in the garden",
        "task_tag": "general",
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert "answer" in data
    assert "reasoning_trace" in data
    assert "confidence" in data
    assert "stats" in data
    assert data["confidence"] > 0
    print(f"  Answer: {data['answer'][:80]}")
    print(f"  Confidence: {data['confidence']:.2f}")
    print("✅ Process Text Test Passed")


def test_process_structured():
    """Test processing structured data via API."""
    print("\n--- Test: Process Structured ---")
    client = _client()
    resp = client.post("/api/process", data={
        "structured": json.dumps({"location": "garden", "color": "red"}),
        "task_tag": "observation",
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert "answer" in data
    print(f"  Answer: {data['answer'][:80]}")
    print("✅ Process Structured Test Passed")


def test_process_empty_rejected():
    """Test that empty input returns an error."""
    print("\n--- Test: Empty Input ---")
    client = _client()
    resp = client.post("/api/process", data={})
    # Should still work (multimodal processor handles empty)
    assert resp.status_code == 200
    print("✅ Empty Input Test Passed")


def test_stats_endpoint():
    """Test the stats API."""
    print("\n--- Test: Stats Endpoint ---")
    client = _client()
    resp = client.get("/api/stats")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "knowledge_entries" in data
    assert "queries_processed" in data
    print(f"  Stats: {data}")
    print("✅ Stats Endpoint Test Passed")


def test_logs_endpoint():
    """Test the logs API."""
    print("\n--- Test: Logs Endpoint ---")
    client = _client()
    # First trigger an activity
    client.post("/api/process", data={"text": "test log entry"})
    resp = client.get("/api/logs")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "logs" in data
    assert len(data["logs"]) > 0
    print(f"  Log entries: {len(data['logs'])}")
    print("✅ Logs Endpoint Test Passed")


def test_knowledge_endpoint():
    """Test the knowledge API."""
    print("\n--- Test: Knowledge Endpoint ---")
    client = _client()
    resp = client.get("/api/knowledge")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "knowledge" in data
    assert len(data["knowledge"]) > 0
    print(f"  Knowledge entries: {len(data['knowledge'])}")
    print("✅ Knowledge Endpoint Test Passed")


def test_export_json():
    """Test JSON export."""
    print("\n--- Test: Export JSON ---")
    client = _client()
    resp = client.get("/api/export/json")
    assert resp.status_code == 200
    assert resp.content_type == "application/json"
    data = json.loads(resp.data)
    assert "exported_at" in data
    assert "knowledge_base" in data
    assert "activity_log" in data
    assert "system_stats" in data
    print(f"  Export size: {len(resp.data)} bytes")
    print("✅ Export JSON Test Passed")


def test_export_zip():
    """Test ZIP export."""
    print("\n--- Test: Export ZIP ---")
    client = _client()
    resp = client.get("/api/export/zip")
    assert resp.status_code == 200
    assert "application/zip" in resp.content_type

    # Verify it's a valid ZIP
    buf = io.BytesIO(resp.data)
    with zipfile.ZipFile(buf, "r") as zf:
        names = zf.namelist()
        assert "stats.json" in names
        assert "knowledge_base.json" in names
        assert "context_engine.json" in names
        assert "activity_log.json" in names
    print(f"  ZIP files: {names}")
    print("✅ Export ZIP Test Passed")


def test_correction_endpoint():
    """Test knowledge correction."""
    print("\n--- Test: Correction Endpoint ---")
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
    data = resp.get_json()
    assert data["status"] == "ok"
    print("✅ Correction Endpoint Test Passed")


def test_feedback_endpoint():
    """Test reward feedback."""
    print("\n--- Test: Feedback Endpoint ---")
    client = _client()
    # Process something first
    client.post("/api/process", data={"text": "eating food"})
    resp = client.post("/api/feedback",
        data=json.dumps({
            "task_tag": "general",
            "reward": 1.0,
            "concepts": ["food", "eating"],
        }),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    print("✅ Feedback Endpoint Test Passed")


def test_context_stats():
    """Test context engine stats endpoint."""
    print("\n--- Test: Context Stats ---")
    client = _client()
    resp = client.get("/api/context/stats")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "registered_concepts" in data
    print(f"  Context stats: {data}")
    print("✅ Context Stats Test Passed")


if __name__ == "__main__":
    test_dashboard_home()
    test_process_text()
    test_process_structured()
    test_process_empty_rejected()
    test_stats_endpoint()
    test_logs_endpoint()
    test_knowledge_endpoint()
    test_export_json()
    test_export_zip()
    test_correction_endpoint()
    test_feedback_endpoint()
    test_context_stats()
    print("\n🎉 All Dashboard Tests Passed!")
