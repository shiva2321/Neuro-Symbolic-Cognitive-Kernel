"""
Quick test to verify the dashboard can start
"""
import sys

print("Testing dashboard startup...")
print("=" * 70)

# Test 1: Backend import
print("\n1. Testing backend import...")
try:
    from utils.graph_backend import get_available_backends
    backends = get_available_backends()
    print(f"   ✓ Available backends: {backends}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Test 2: NCGN imports
print("\n2. Testing NCGN imports...")
try:
    from ncgn import DualSystemArchitecture, LinguisticGraph
    print(f"   ✓ NCGN modules imported")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Test 3: Dashboard imports
print("\n3. Testing dashboard imports...")
try:
    from flask import Flask
    from flask_socketio import SocketIO
    print(f"   ✓ Flask and SocketIO imported")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Test 4: Dashboard utilities
print("\n4. Testing dashboard utilities...")
try:
    from dashboard_utils import MetricsMonitor, HardwareProfiler
    print(f"   ✓ Dashboard utilities imported")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ All tests passed!")
print("\nYou can now start the dashboard:")
print("  python ncgn_dashboard.py")
print("\nOr run it directly:")
print("  C:\\Users\\shiva\\AppData\\Local\\Programs\\Python\\Python312\\python.exe ncgn_dashboard.py")
print("=" * 70)

