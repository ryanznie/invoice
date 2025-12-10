"""
FastAPI + Gradio app for Invoice NER model testing
Accepts: Image + Text file (JSON with words and bboxes)

This is the main entry point that orchestrates the modular components in src/
"""

import os
import logging
from dotenv import load_dotenv
import gradio as gr

# Import from src modules
from src import app, create_gradio_interface, load_model, DEVICE, MODEL_PATH

load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================================
# GRADIO INTERFACE SETUP
# ============================================================================

# Create Gradio interface from src module
demo = create_gradio_interface()


# Mount Gradio app to FastAPI
app = gr.mount_gradio_app(app, demo, path="/")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    import argparse

    # Parse arguments
    parser = argparse.ArgumentParser(description="Invoice NER App")
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

    # Load model before starting server (unless in debug mode with reload)
    if not args.debug:
        load_model()

    print("\n" + "=" * 60)
    print("🚀 Starting Invoice NER App")
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
