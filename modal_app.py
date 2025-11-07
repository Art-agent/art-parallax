"""
Modal deployment for Parallax - Distributed LLM Inference Engine

This Modal app deploys Parallax with:
- Scheduler (FastAPI server for orchestration)
- Worker nodes (GPU/CPU inference workers)
- Chat UI (React frontend)
- Model storage using Modal Volumes

To deploy:
1. Install Modal: pip install modal
2. Authenticate: modal setup
3. Deploy: modal deploy modal_app.py
4. Run: modal run modal_app.py
"""

import modal
import os
from pathlib import Path
from typing import Optional

# Create the Modal app
app = modal.App("parallax-llm-inference")

# Docker image with all dependencies
parallax_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install([
        "build-essential",
        "curl",
        "git",
        "wget",
        "libzmq3-dev",
        "nodejs",
        "npm",
    ])
    .pip_install([
        "fastapi==0.115.6",
        "uvicorn[standard]==0.25.0",
        "uvloop==0.21.0",
        "pydantic==2.10.5",
        "httpx==0.28.1",
        "pyzmq==26.2.0",
        "aiofiles==24.1.0",
        "sse-starlette==2.2.1",
        "websockets==14.1",
        "python-multipart==0.0.20",
        "python-jose[cryptography]==3.3.0",
        "passlib[bcrypt]==1.7.4",
        "email-validator==2.2.0",
        "rich==13.9.4",
        "click==8.1.8",
        "psutil==6.1.0",
        "numpy",
        "torch",  # For GPU inference
        "transformers",  # For model loading
        "sglang[all]",  # For GPU inference engine
    ])
    .run_commands([
        # Build the frontend
        "cd /app/src/frontend && npm install && npm run build",
    ])
)

# Volume for storing models (persistent across runs)
model_volume = modal.Volume.from_name(
    "parallax-models",
    create_if_missing=True
)

# Network file system for shared state
nfs = modal.NetworkFileSystem.from_name(
    "parallax-shared-state",
    create_if_missing=True
)

# Secret for configuration (create this in Modal dashboard)
secret = modal.Secret.from_name("parallax-config")


@app.cls(
    image=parallax_image,
    gpu="A10G",  # For GPU inference, adjust as needed
    volumes={
        "/models": model_volume,
        "/shared": nfs,
    },
    secrets=[secret],
    concurrency_limit=10,
    container_idle_timeout=300,
    allow_concurrent_inputs=100,
)
class ParallaxScheduler:
    """Central scheduler that orchestrates worker nodes"""

    @modal.build()
    def build(self):
        """Build-time setup"""
        print("Building Parallax Scheduler...")

    @modal.enter()
    def startup(self):
        """Initialize scheduler on container start"""
        import sys
        sys.path.append("/app")

        # Import Parallax modules
        from src.backend.main import app as fastapi_app
        from src.backend.core import config

        self.app = fastapi_app
        self.config = config

        # Set up environment from Modal secrets
        os.environ.update({
            "PARALLAX_PORT": os.environ.get("PARALLAX_PORT", "3001"),
            "PARALLAX_HOST": "0.0.0.0",
            "MODEL_PATH": "/models",
            "SHARED_STATE_PATH": "/shared",
            "ENABLE_AUTH": os.environ.get("ENABLE_AUTH", "false"),
            "JWT_SECRET": os.environ.get("JWT_SECRET", "your-secret-key-here"),
        })

        print("Scheduler initialized successfully")

    @modal.web_endpoint(method="GET", docs=True)
    async def health(self):
        """Health check endpoint"""
        return {"status": "healthy", "service": "scheduler"}

    @modal.web_endpoint(method="POST")
    async def schedule_task(self, task: dict):
        """Schedule a task to worker nodes"""
        from src.backend.services.scheduler import SchedulerService

        scheduler = SchedulerService()
        result = await scheduler.schedule(task)
        return result

    @modal.asgi_app()
    def fastapi(self):
        """Serve the FastAPI application"""
        return self.app


@app.cls(
    image=parallax_image,
    gpu="A10G",  # GPU for inference
    volumes={
        "/models": model_volume,
        "/shared": nfs,
    },
    secrets=[secret],
    concurrency_limit=5,
    container_idle_timeout=600,
)
class ParallaxWorker:
    """Worker node that performs actual inference"""

    @modal.build()
    def build(self):
        """Build-time setup"""
        print("Building Parallax Worker...")

    @modal.enter()
    def startup(self):
        """Initialize worker on container start"""
        import sys
        sys.path.append("/app")

        from src.parallax.worker import Worker
        from src.parallax.config import WorkerConfig

        # Configure worker
        self.config = WorkerConfig(
            model_path="/models",
            device="cuda" if modal.gpu.is_available() else "cpu",
            max_batch_size=int(os.environ.get("MAX_BATCH_SIZE", "32")),
            max_sequence_length=int(os.environ.get("MAX_SEQ_LENGTH", "2048")),
        )

        self.worker = Worker(self.config)
        self.worker.start()

        print(f"Worker initialized on {self.config.device}")

    @modal.method()
    async def process_inference(self, prompt: str, **kwargs):
        """Process an inference request"""
        result = await self.worker.generate(
            prompt=prompt,
            max_tokens=kwargs.get("max_tokens", 512),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
            stream=kwargs.get("stream", False),
        )
        return result

    @modal.method()
    async def load_model(self, model_name: str):
        """Load a specific model"""
        await self.worker.load_model(model_name)
        return {"status": "success", "model": model_name}


@app.cls(
    image=parallax_image,
    volumes={
        "/shared": nfs,
    },
    secrets=[secret],
    container_idle_timeout=300,
)
class ParallaxChatUI:
    """React frontend for chat interface"""

    @modal.build()
    def build(self):
        """Build the frontend"""
        print("Building Chat UI...")

    @modal.enter()
    def startup(self):
        """Start the chat UI server"""
        import sys
        sys.path.append("/app")

        # Set environment variables
        os.environ.update({
            "REACT_APP_API_URL": os.environ.get("SCHEDULER_URL", "https://parallax-scheduler.modal.run"),
            "PORT": "3002",
        })

        print("Chat UI initialized")

    @modal.asgi_app()
    def serve_ui(self):
        """Serve the React frontend"""
        from fastapi import FastAPI
        from fastapi.staticfiles import StaticFiles
        from fastapi.responses import FileResponse

        app = FastAPI()

        # Serve static files
        app.mount("/static", StaticFiles(directory="/app/src/frontend/build/static"), name="static")

        @app.get("/")
        async def root():
            return FileResponse("/app/src/frontend/build/index.html")

        @app.get("/{path:path}")
        async def catch_all(path: str):
            file_path = f"/app/src/frontend/build/{path}"
            if os.path.exists(file_path):
                return FileResponse(file_path)
            return FileResponse("/app/src/frontend/build/index.html")

        return app


@app.function(
    image=parallax_image,
    volumes={
        "/models": model_volume,
    },
    secrets=[secret],
    timeout=3600,
)
def download_model(model_name: str = "meta-llama/Llama-2-7b-chat-hf"):
    """Download and cache a model to the volume"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Downloading model: {model_name}")

    # Download to volume
    cache_dir = f"/models/{model_name.replace('/', '_')}"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        use_auth_token=os.environ.get("HF_TOKEN"),
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        use_auth_token=os.environ.get("HF_TOKEN"),
        torch_dtype="auto",
        device_map="auto",
    )

    print(f"Model {model_name} downloaded successfully to {cache_dir}")
    return {"status": "success", "model": model_name, "path": cache_dir}


@app.function(
    image=parallax_image,
    secrets=[secret],
)
def test_deployment():
    """Test the deployment with a simple inference"""
    import httpx
    import asyncio

    async def test():
        scheduler_url = os.environ.get("SCHEDULER_URL", "https://parallax-scheduler.modal.run")

        # Test health endpoint
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{scheduler_url}/health")
            print(f"Health check: {response.json()}")

            # Test inference
            response = await client.post(
                f"{scheduler_url}/v1/chat/completions",
                json={
                    "model": "llama-2-7b-chat",
                    "messages": [
                        {"role": "user", "content": "Hello, how are you?"}
                    ],
                    "max_tokens": 50,
                },
            )
            print(f"Inference test: {response.json()}")

    asyncio.run(test())


@app.local_entrypoint()
def main(
    action: str = "deploy",
    model: Optional[str] = None,
    workers: int = 2,
):
    """
    Main entrypoint for Modal deployment

    Args:
        action: Action to perform (deploy, test, download-model)
        model: Model to download (for download-model action)
        workers: Number of worker instances to spawn
    """

    if action == "deploy":
        print("Deploying Parallax on Modal...")
        print(f"Spawning {workers} worker instances...")

        # Deploy scheduler
        scheduler = ParallaxScheduler()
        print("✓ Scheduler deployed")

        # Deploy workers
        for i in range(workers):
            worker = ParallaxWorker()
            print(f"✓ Worker {i+1} deployed")

        # Deploy chat UI
        chat_ui = ParallaxChatUI()
        print("✓ Chat UI deployed")

        print("\n🚀 Parallax deployed successfully!")
        print("\nAccess points:")
        print("- Scheduler API: https://parallax-scheduler.modal.run")
        print("- Chat UI: https://parallax-chat.modal.run")
        print("\nTo test: modal run modal_app.py --action test")

    elif action == "test":
        print("Testing Parallax deployment...")
        test_deployment.remote()

    elif action == "download-model":
        if not model:
            model = "meta-llama/Llama-2-7b-chat-hf"
        print(f"Downloading model: {model}")
        result = download_model.remote(model)
        print(f"Result: {result}")

    else:
        print(f"Unknown action: {action}")
        print("Available actions: deploy, test, download-model")


if __name__ == "__main__":
    # For local testing
    main()