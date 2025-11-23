"""Launch the token-level visualization streamlit app.

This script provides an easy way to start the streamlit app for visualizing
token-level probabilities and NLLs from validation pickle files.

Usage:
    python run_token_viz_app.py
    
    # Or with a specific port:
    python run_token_viz_app.py --port 8502
"""

import subprocess
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description="Launch token visualization streamlit app")
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port to run streamlit on (default: 8501)"
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't automatically open browser"
    )
    args = parser.parse_args()
    
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "src/analysis/token_visualization_app.py",
        "--server.port",
        str(args.port),
    ]
    
    if args.no_browser:
        cmd.extend(["--server.headless", "true"])
    
    print(f"🚀 Launching streamlit app on port {args.port}...")
    print(f"📊 URL: http://localhost:{args.port}")
    print("\nPress Ctrl+C to stop the app\n")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down streamlit app...")


if __name__ == "__main__":
    main()


