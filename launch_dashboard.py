#!/usr/bin/env python3
"""
NSCK Unified Dashboard Launcher
=================================

Quick launch script for the NSCK Unified Scientific Dashboard.

Usage:
    python launch_dashboard.py [--port PORT] [--host HOST] [--debug]

Examples:
    python launch_dashboard.py
    python launch_dashboard.py --port 5001
    python launch_dashboard.py --host 0.0.0.0 --debug

Features:
- Automatic environment setup
- Graceful shutdown handling
- Session logging with unique identifiers
- Performance monitoring
"""

import sys
import os
import argparse
import signal
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent / "nsck-demo" / "python"))

def main():
    parser = argparse.ArgumentParser(
        description="Launch NSCK Unified Scientific Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                         # Launch on default port 5000
  %(prog)s --port 8080             # Launch on port 8080
  %(prog)s --host 0.0.0.0          # Listen on all interfaces
  %(prog)s --debug                 # Enable debug mode with auto-reload
  
The dashboard will be available at http://localhost:PORT/
All sessions are logged to: /workspaces/Node_network/logs/
        """
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=5000,
        help="Port to run the dashboard on (default: 5000)"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1, use 0.0.0.0 for external access)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with auto-reload"
    )
    
    args = parser.parse_args()
    
    # Ensure logs directory exists
    log_dir = Path("/workspaces/Node_network/logs")
    log_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("NSCK UNIFIED SCIENTIFIC DASHBOARD")
    print("=" * 80)
    print(f"Starting dashboard server...")
    print(f"  Host: {args.host}")
    print(f"  Port: {args.port}")
    print(f"  Debug Mode: {args.debug}")
    print(f"  Log Directory: {log_dir}")
    print("")
    print(f"Dashboard URL: http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}/")
    print("")
    print("Press Ctrl+C to stop the server")
    print("=" * 80)
    print("")
    
    # Import and run the app
    try:
        from unified_dashboard import app, get_logger
        
        # Setup graceful shutdown
        def signal_handler(sig, frame):
            print("\n\n" + "=" * 80)
            print("Shutting down dashboard...")
            logger = get_logger()
            stats = logger.get_stats()
            print(f"  Session Duration: {stats['session_duration_seconds']:.1f} seconds")
            print(f"  Total Events Logged: {stats['total_events']}")
            print(f"  Log File: {stats['log_file']}")
            print("=" * 80)
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        # Run the Flask app
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            threaded=True
        )
        
    except ImportError as e:
        print(f"ERROR: Could not import unified_dashboard: {e}")
        print("\nMake sure you're running this from the project root directory.")
        print("Expected location: /workspaces/Node_network/")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to start dashboard: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
