#!/usr/bin/env python3
"""Script to build and optionally serve the documentation locally.

Usage:
    python docs/build_docs.py          # Build documentation
    python docs/build_docs.py --serve  # Build and serve documentation
"""

import argparse
import http.server
import os
import shutil
import socketserver
import subprocess
import sys
import webbrowser

from pathlib import Path


def check_dependencies():
    """Check for pandoc dependency."""
    print("🔍 Checking dependencies...")

    # Check for pandoc
    try:
        subprocess.run(["pandoc", "--version"], capture_output=True, check=True)
        print("✅ Pandoc found in PATH")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Pandoc not found. Please install it from: https://pandoc.org/installing.html")
        sys.exit(1)


def check_images():
    """Check that images exist in docs/_static/images/."""
    print("🖼️  Checking images...")

    docs_dir = Path(__file__).parent
    images_dir = docs_dir / "_static" / "images"

    if not images_dir.exists():
        print(f"   ⚠️  Warning: Images directory not found at {images_dir}")
        print("   Please ensure images are in the docs/_static/images/ directory.")
        return

    # Count image files
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp"}
    image_count = 0

    for image_file in images_dir.iterdir():
        if image_file.is_file() and image_file.suffix.lower() in image_extensions:
            image_count += 1

    if image_count > 0:
        print(f"   ✅ Found {image_count} image(s) in docs/_static/images/")
    else:
        print("   ⚠️  No image files found in docs/_static/images/")


def clear_build_cache():
    """Clear only the build directory for a fresh build."""
    docs_dir = Path(__file__).parent
    build_dir = docs_dir / "_build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
        print("   Removed docs/_build directory")


def build_docs():
    """Build the documentation using Sphinx."""
    print("Building documentation...")

    # Clear build directory for fresh build
    clear_build_cache()

    # Check dependencies first
    check_dependencies()

    # Check that images exist
    check_images()

    try:
        # Ensure we're in the docs directory for sphinx-build
        docs_dir = Path(__file__).parent
        current_dir = Path.cwd()
        if current_dir != docs_dir:
            os.chdir(docs_dir)

        print("Building documentation with sphinx-build...")
        result = subprocess.run(
            ["uv", "run", "sphinx-build", "-q", "-b", "html", ".", "_build/html"],
            check=True,
            capture_output=True,
            text=True,
        )
        print("✅ Documentation built successfully!")

        # Show only important warnings if any
        if result.stderr:
            stderr_lines = result.stderr.split("\n")
            important_warnings = [line for line in stderr_lines if "ERROR" in line or "CRITICAL" in line]
            if important_warnings:
                print("⚠️  Important warnings:")
                for warning in important_warnings[:5]:  # Show max 5
                    print(f"   {warning}")

        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error building documentation: {e}")
        print("STDOUT:", e.stdout[-1000:] if e.stdout else "No stdout")  # Last 1000 chars only
        print("STDERR:", e.stderr[-1000:] if e.stderr else "No stderr")  # Last 1000 chars only
        return False


def find_available_port(start_port: int, max_attempts: int = 10) -> int:
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socketserver.TCPServer(("", port), None) as _:
                return port
        except OSError:
            continue
    raise OSError(f"Could not find an available port in range {start_port}-{start_port + max_attempts - 1}")


def serve_docs(port: int = 8000):
    """Serve the documentation locally."""
    build_dir = Path("_build/html")
    if not build_dir.exists():
        print("❌ Build directory not found. Please build the documentation first.")
        return

    os.chdir(build_dir)

    try:
        available_port = find_available_port(port)
        if available_port != port:
            print(f"⚠️ Port {port} is in use. Using port {available_port} instead.")

        class Handler(http.server.SimpleHTTPRequestHandler):
            def log_message(self, fmt, *args):
                # Suppress log messages
                pass

        with socketserver.TCPServer(("", available_port), Handler) as httpd:
            url = f"http://localhost:{available_port}"
            print(f"🌐 Serving documentation at {url}")
            print("Press Ctrl+C to stop the server")

            # Open browser
            webbrowser.open(url)

            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\n👋 Server stopped")
    except OSError as e:
        print(f"❌ Error starting server: {e}")
        print("Please try again with a different port using --port option")


def main():
    parser = argparse.ArgumentParser(description="Build and serve scXpand documentation")
    parser.add_argument("--serve", action="store_true", help="Serve documentation after building")
    parser.add_argument("--port", type=int, default=8000, help="Port for serving (default: 8000)")
    parser.add_argument("--no-build", action="store_true", help="Skip building, just serve")

    args = parser.parse_args()

    if not args.no_build:
        success = build_docs()
        if not success and args.serve:
            print("❌ Cannot serve documentation due to build errors")
            return

    if args.serve:
        serve_docs(args.port)


if __name__ == "__main__":
    main()
