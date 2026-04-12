"""
FastAPI entry point for the Invoice NER backend.
"""

import os
import logging
import argparse
from dotenv import load_dotenv
import uvicorn

from src import app, load_model, DEVICE, MODEL_PATH

load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Run the API server."""
    parser = argparse.ArgumentParser(description="Invoice NER API")
    parser.add_argument(
        "--debug", action="store_true", help="Run in debug mode with auto-reload"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "7860")),
        help="Port to run on (default: from PORT env or 7860)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("HOST", "0.0.0.0"),
        help="Host to bind to (default: from HOST env or 0.0.0.0)",
    )
    args = parser.parse_args()

    if not args.debug:
        load_model()

    print("\n" + "=" * 60)
    print("🚀 Starting Invoice NER API")
    if args.debug:
        print("🐛 DEBUG MODE - Auto-reload enabled")
    print("=" * 60)
    print(f"📱 Device: {DEVICE}")
    print(f"🔧 Model: {MODEL_PATH}")
    print(f"🌐 Open your browser to: http://{args.host}:{args.port}")
    print(f"📊 Log Level: {os.getenv('LOG_LEVEL', 'INFO')}")
    print("=" * 60 + "\n")

    uvicorn.run(
        "app:app" if args.debug else app,
        host=args.host,
        port=args.port,
        reload=args.debug,
        log_level="debug" if args.debug else "info",
    )


if __name__ == "__main__":
    main()
