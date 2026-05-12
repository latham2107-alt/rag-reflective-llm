import sys
import argparse
import uvicorn


def build_index():
    from src.indexing.build_index import build_index
    build_index()


def run_demo():
    from src.demo.cli_demo import run_demo
    run_demo()


def run_eval():
    from src.eval.run_eval import run_evaluation
    run_evaluation()


def run_web(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    print(f" Starting RAG vs Reflective Agent web server...")
    print(f" Web interface: http://{host}:{port}")
    print(f" API docs: http://{host}:{port}/docs")
    print(f" Health check: http://{host}:{port}/api/health")
    print("-" * 50)

    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


def main():
    parser = argparse.ArgumentParser(
        description="RAG vs Reflective Agent - Academic QA System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py build-index           Build the vector index from corpus
  python run.py demo                  Run interactive CLI demo
  python run.py eval                  Run evaluation benchmark
  python run.py web                   Run web interface and API server
  python run.py web --host 127.0.0.1  Run on localhost only
  python run.py web --port 3000       Run on port 3000
  python run.py web --reload          Run with auto-reload for development
        """
    )

    parser.add_argument(
        "command",
        choices=["build-index", "demo", "eval", "web"],
        help="Command to run"
    )

    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind the web server to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the web server to (default: 8000)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )

    args = parser.parse_args()

    if args.command == "build-index":
        build_index()
    elif args.command == "demo":
        run_demo()
    elif args.command == "eval":
        run_eval()
    elif args.command == "web":
        run_web(host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
