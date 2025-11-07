import sys
import modal

image = modal.Image.from_dockerfile("./docker/Dockerfile.hopper")

app = modal.App("art-parallax", image=image)


@app.function(gpu="H100")
@modal.asgi_app()
def fastapi_app():
  from src.backend.main import app
  return app