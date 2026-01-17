#!/usr/bin/env python
"""
Quick diagnostic for Ollama setup and model availability.
Helps debug connectivity and model issues before running the full trainer.
"""
from __future__ import annotations
import json
import sys
from http.client import HTTPConnection


def check_ollama_connection(host: str = "localhost", port: int = 11434) -> bool:
    """Check if Ollama HTTP API is reachable."""
    try:
        conn = HTTPConnection(host, port, timeout=5)
        conn.request("GET", "/api/tags", headers={})
        resp = conn.getresponse()
        conn.close()
        return resp.status == 200
    except Exception as e:
        print(f"✗ Cannot connect to Ollama at {host}:{port}: {e}")
        return False


def list_models(host: str = "localhost", port: int = 11434) -> list:
    """List available models from Ollama."""
    try:
        conn = HTTPConnection(host, port, timeout=5)
        conn.request("GET", "/api/tags", headers={})
        resp = conn.getresponse()
        if resp.status != 200:
            print(f"✗ Ollama API error: {resp.status}")
            return []
        data = json.loads(resp.read().decode("utf-8"))
        models = [m.get("name", "unknown") for m in data.get("models", [])]
        conn.close()
        return models
    except Exception as e:
        print(f"✗ Error listing models: {e}")
        return []


def main():
    print("=" * 70)
    print(" OLLAMA DIAGNOSTIC")
    print("=" * 70)
    print()

    host = "localhost"
    port = 11434

    print(f"Checking Ollama at http://{host}:{port}...")
    if not check_ollama_connection(host, port):
        print()
        print("✗ Ollama is NOT reachable.")
        print()
        print("Solutions:")
        print("1. Start Ollama with: ollama serve")
        print("2. Or check if Ollama is running as a background service.")
        print("3. Verify firewall allows localhost:11434.")
        print()
        return 1

    print("✓ Ollama is reachable.")
    print()

    print("Available models:")
    models = list_models(host, port)
    if models:
        for m in models:
            print(f"  • {m}")
    else:
        print("  (none)")
        print()
        print("You need to pull a model first. Examples:")
        print("  ollama pull llama2")
        print("  ollama pull llama3.1")
        print("  ollama pull mistral")
        print("  ollama pull neural-chat")
        print()
        return 1

    print()
    print("✓ Setup looks good. You can run the trainer now.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
