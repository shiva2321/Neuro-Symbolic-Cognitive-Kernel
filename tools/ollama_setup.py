#!/usr/bin/env python
"""
Quick setup script: Check Ollama, list available models, and help pull one if needed.
"""
from __future__ import annotations
import json
import sys
import subprocess
import os
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
            return []
        data = json.loads(resp.read().decode("utf-8"))
        models = [m.get("name", "unknown") for m in data.get("models", [])]
        conn.close()
        return models
    except Exception:
        return []


def pull_model(model: str) -> bool:
    """Try to pull a model using subprocess.
    Use the OLLAMA_CLI_PATH environment variable if provided to specify the full path to the ollama executable.
    """
    cli = os.environ.get("OLLAMA_CLI_PATH", "ollama")
    try:
        print(f"\nPulling model '{model}' using CLI: {cli} ... (this may take a few minutes)")
        result = subprocess.run(
            [cli, "pull", model],
            capture_output=True,
            text=True,
            timeout=900
        )
        if result.returncode == 0:
            print(f"✓ Successfully pulled '{model}'")
            return True
        else:
            # Print stderr and a helpful hint
            print(f"✗ Failed to pull '{model}': {result.stderr.strip()}")
            if cli != "ollama":
                print(f"(Tried CLI at: {cli})")
            return False
    except FileNotFoundError:
        print(f"✗ '{cli}' command not found. Make sure Ollama is installed and the CLI path is correct.")
        if cli == "ollama":
            print("Try installing Ollama or set the full path via the OLLAMA_CLI_PATH environment variable.")
        else:
            print("Verify the file exists and is executable.")
        return False
    except subprocess.TimeoutExpired:
        print("✗ Timeout pulling model (took > 15 min). Try manually: ollama pull llama3.1")
        return False
    except Exception as e:
        print(f"✗ Error pulling model: {e}")
        return False


def main():
    print("=" * 70)
    print(" OLLAMA SETUP & MODEL CHECKER")
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

    print("Checking available models...")
    models = list_models(host, port)
    if models:
        print(f"✓ Found {len(models)} model(s):")
        for m in models:
            print(f"  • {m}")
        print()
        print("You're ready to use the trainer!")
        return 0
    else:
        print("✗ No models found.")
        print()
        print("Available models to pull:")
        suggestions = [
            ("llama3.1", "Latest Llama 3.1 (faster, recommended)"),
            ("llama2", "Llama 2 (stable, good for most tasks)"),
            ("mistral", "Mistral (lightweight, fast)"),
            ("neural-chat", "Neural Chat (optimized for conversations)"),
            ("orca-mini", "Orca Mini (very small, quick start)"),
        ]
        for model_name, desc in suggestions:
            print(f"  • {model_name:20} - {desc}")
        print()

        # Ask user which to pull
        print("Which model would you like to pull? (Enter name, or press Enter to skip)")
        choice = input("Model name [llama3.1]: ").strip() or "llama3.1"

        if pull_model(choice):
            print()
            print("✓ Setup complete! You can now run the trainer.")
            return 0
        else:
            print()
            print("Pull failed. You can try manually:")
            print(f"  ollama pull {choice}")
            print()
            print("If you have the ollama binary but it is not in PATH, set the full path through the environment variable:")
            print("  $env:OLLAMA_CLI_PATH = 'C:\\path\\to\\ollama.exe'")
            print("Then re-run this script.")
            return 1


if __name__ == "__main__":
    sys.exit(main())
