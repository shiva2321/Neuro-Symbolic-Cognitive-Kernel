"""
Test Suite for Unified Dashboard
==================================
Basic tests to verify dashboard functionality and logging system.
"""

import pytest
import sys
import os
from pathlib import Path

# Add project paths - unified_dashboard is in nsck-demo/python
dashboard_path = Path(__file__).parent / "nsck-demo" / "python"
sys.path.insert(0, str(dashboard_path))

def test_dashboard_import():
    """Test that the unified dashboard can be imported."""
    try:
        import unified_dashboard
        assert unified_dashboard.app is not None
        print(f"✓ Successfully imported unified_dashboard from {dashboard_path}")
    except ImportError as e:
        pytest.fail(f"Failed to import unified_dashboard: {e}")


def test_structured_logger_creation():
    """Test that StructuredLogger can be instantiated."""
    from unified_dashboard import StructuredLogger
    
    logger = StructuredLogger(max_entries=100)
    assert logger is not None
    assert len(logger.entries) == 0


def test_structured_logger_logging():
    """Test basic logging functionality."""
    from unified_dashboard import StructuredLogger
    
    logger = StructuredLogger(max_entries=100)
    
    # Log a test event
    logger.log("test", "Test message", "INFO", metadata={"test_key": "test_value"})
    
    # Verify it was logged
    assert len(logger.entries) == 1
    entry = logger.entries[0]
    
    assert entry["category"] == "test"
    assert entry["message"] == "Test message"
    assert entry["severity"] == "INFO"
    assert entry["metadata"]["test_key"] == "test_value"


def test_structured_logger_filtering():
    """Test log filtering by category and severity."""
    from unified_dashboard import StructuredLogger
    
    logger = StructuredLogger(max_entries=100)
    
    # Log various events
    logger.log("system", "System message", "INFO")
    logger.log("game", "Game message", "DEBUG")
    logger.log("error", "Error message", "ERROR")
    logger.log("system", "Another system message", "WARNING")
    
    # Test category filtering
    system_logs = logger.get_recent(category="system")
    assert len(system_logs) == 2
    
    game_logs = logger.get_recent(category="game")
    assert len(game_logs) == 1
    
    # Test severity filtering
    error_logs = logger.get_recent(severity="ERROR")
    assert len(error_logs) == 1


def test_structured_logger_export_json():
    """Test JSON export functionality."""
    from unified_dashboard import StructuredLogger
    import json
    
    logger = StructuredLogger(max_entries=100)
    logger.log("test", "Export test", "INFO", metadata={"data": 123})
    
    # Export as JSON
    json_str = logger.export_json()
    data = json.loads(json_str)
    
    assert "entries" in data
    assert "stats" in data
    assert "session_start" in data
    assert len(data["entries"]) == 1


def test_structured_logger_export_csv():
    """Test CSV export functionality."""
    from unified_dashboard import StructuredLogger
    
    logger = StructuredLogger(max_entries=100)
    logger.log("test", "CSV test", "INFO")
    
    # Export as CSV
    csv_str = logger.export_csv()
    lines = csv_str.split("\n")
    
    assert len(lines) >= 2  # Header + at least one entry
    assert "timestamp,category,severity,message,metadata" in lines[0]


def test_structured_logger_stats():
    """Test statistics tracking."""
    from unified_dashboard import StructuredLogger
    
    logger = StructuredLogger(max_entries=100)
    
    # Log events
    logger.log("system", "Message 1", "INFO")
    logger.log("game", "Message 2", "DEBUG")
    logger.log("system", "Message 3", "WARNING")
    
    stats = logger.get_stats()
    
    assert stats["total_events"] == 3
    assert "system_INFO" in stats["by_category"]
    assert stats["by_category"]["system_INFO"] == 1
    assert "log_file" in stats


def test_flask_app_exists():
    """Test that Flask app is properly configured."""
    from unified_dashboard import app
    
    assert app is not None
    assert app.name == "unified_dashboard"


def test_api_routes_registered():
    """Test that key API routes are registered."""
    from unified_dashboard import app
    
    # Get all registered routes
    routes = [str(rule) for rule in app.url_map.iter_rules()]
    
    # Check for key endpoints
    assert any("/api/logs" in route for route in routes)
    assert any("/api/chat" in route for route in routes)
    assert any("/api/game/start" in route for route in routes)
    assert any("/api/learn/upload" in route for route in routes)


def test_learned_policy():
    """Test dashboard policy learner."""
    from unified_dashboard import LearnedPolicy
    
    policy = LearnedPolicy()
    
    # Train on some state-action pairs
    policy.train("state1", "action1")
    policy.train("state1", "action1")  # Reinforce
    policy.train("state1", "action2")
    
    # Should predict most common action
    predicted = policy.predict("state1")
    assert predicted == "action1"
    
    # Unknown state should return None
    assert policy.predict("unknown_state") is None
    
    # Test stats
    stats = policy.get_stats()
    assert stats["states_seen"] == 1
    assert stats["total_experiences"] == 3


def test_game_initialization():
    """Test game state initialization functions."""
    from unified_dashboard import _init_snake, _init_pong, _init_maze
    
    # Test Snake
    snake = _init_snake()
    assert snake["type"] == "snake"
    assert "head" in snake
    assert "body" in snake
    assert "food" in snake
    assert snake["score"] == 0
    
    # Test Pong
    pong = _init_pong()
    assert pong["type"] == "pong"
    assert "ball_x" in pong
    assert "ball_y" in pong
    assert "p1_y" in pong
    
    # Test Maze
    maze = _init_maze()
    assert maze["type"] == "maze"
    assert "player" in maze
    assert "exit" in maze
    assert "walls" in maze


def test_app_routes_accessible():
    """Test that app routes return responses."""
    from unified_dashboard import app
    
    client = app.test_client()
    
    # Test index route
    response = client.get("/")
    assert response.status_code == 200
    assert b"NSCK" in response.data or b"nsck" in response.data.lower()
    
    # Test stats endpoint
    response = client.get("/api/stats")
    assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
