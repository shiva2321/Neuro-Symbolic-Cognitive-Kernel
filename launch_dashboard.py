"""
Dashboard Launcher
Quick launcher for the Neural Network Dashboard
"""

import os
import sys
import webbrowser
import time
import subprocess

def check_dependencies():
    """Check if required packages are installed"""
    required = ['flask', 'PyPDF2', 'docx']
    missing = []

    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        print("⚠️  Missing required packages:")
        for pkg in missing:
            print(f"   - {pkg}")
        print("\nInstalling missing packages...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + missing)
        print("✅ Packages installed!\n")

def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║          🧠 NEURAL NETWORK DASHBOARD LAUNCHER 🧠             ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # Check dependencies
    print("Checking dependencies...")
    check_dependencies()

    # Create necessary directories
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)

    print("✅ Directories created")
    print("\n" + "="*62)
    print("Starting dashboard server...")
    print("="*62)
    print("\n📍 Dashboard URL: http://localhost:5000")
    print("\n⚙️  Features:")
    print("   • Upload PDF, TXT, DOC, Code files")
    print("   • Train neural network with custom parameters")
    print("   • Query the system with natural language")
    print("   • Visualize graph networks interactively")
    print("   • Export graphs in multiple formats")
    print("\n💡 Tip: The browser will open automatically in 3 seconds")
    print("    Or manually navigate to: http://localhost:5000")
    print("\n⏸️  Press Ctrl+C to stop the server")
    print("="*62 + "\n")

    # Open browser after a delay
    def open_browser():
        time.sleep(3)
        try:
            webbrowser.open('http://localhost:5000')
        except:
            pass

    import threading
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()

    # Start the Flask app
    try:
        from dashboard import app
        app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)
    except KeyboardInterrupt:
        print("\n\n" + "="*62)
        print("Dashboard stopped. Thank you for using the system!")
        print("="*62)
    except Exception as e:
        print(f"\n❌ Error starting dashboard: {e}")
        print("\nTroubleshooting:")
        print("  1. Check if port 5000 is available")
        print("  2. Ensure all dependencies are installed")
        print("  3. Try running: python dashboard.py")

if __name__ == '__main__':
    main()

