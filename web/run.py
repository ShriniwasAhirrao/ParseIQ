"""Start the ParseIQ Web UI — backend + frontend dev servers.

Usage:
    python web/run.py              # Start backend in production mode (port 8000)
    python web/run.py --dev        # Start backend + frontend dev server
"""
import os
import sys
import subprocess

# Ensure project root is importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    dev_mode = "--dev" in sys.argv

    host = "0.0.0.0" if dev_mode else "127.0.0.1"

    print("=" * 50)
    print("  ParseIQ Web UI")
    print("=" * 50)
    print(f"  Backend:  http://localhost:8000")
    print(f"  API docs: http://localhost:8000/docs")
    if dev_mode:
        print(f"  Frontend: http://localhost:5173")
        print(f"  Mode:     development (hot-reload on)")
    else:
        print(f"  Mode:     production")
    print("=" * 50)
    print()

    procs = []

    # Start frontend dev server in background (if --dev)
    if dev_mode:
        frontend_dir = os.path.join(ROOT, "web", "frontend")
        procs.append(subprocess.Popen(
            "npm run dev",
            cwd=frontend_dir,
            shell=True,
        ))

    # Start backend (blocking)
    try:
        import uvicorn
        uvicorn.run(
            "web.api.main:app",
            host=host,
            port=8000,
            reload=dev_mode,
            reload_dirs=[os.path.join(ROOT, "web", "api"), os.path.join(ROOT, "parseiq")] if dev_mode else None,
        )
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            p.terminate()


if __name__ == "__main__":
    main()
