import modal
import os
from threading import Thread
import time
import subprocess
import requests

# Call docker image to serve to modal
parallax_image = (
  # We will use hopper for inference since it is cheaper
  modal.Image.from_dockerfile("./docker/Dockerfile.hopper")
)

app = modal.App( #type: ignore
  name="art-parallax-inference", image=parallax_image
)

volume = modal.Volume.from_name("my-persisted-volume", create_if_missing=True)

@app.function(gpu="H100", volumes={"/parallax": volume}, container_idle_timeout=600, allow_concurrent_inputs=64, timeout=60)
@modal.asgi_app() 
def parallax_node():
  """Starts the parallax scheduler and exposes AI endpoint"""
  repo_dir = "/parallax"
  if not os.path.exists(repo_dir):
    subprocess.run(
      ["git", "clone", "https://github.com/Art-agent/art-parallax.git", repo_dir],
      check=True,
    )
    subprocess.run(
      ["pip", "install", "-e", f"{repo_dir}[gpu]"], shell=True, check=True
    )
  else:
    print("Parallax already installed at /parallax")
    
  def _run_parallax():
    cmd = [
      "parallax",
      "run",
      "-m", "Qwen/Qwen3-0.6B",      # Change model here
      "--host", "0.0.0.0",
      "-u",                         # Disable telemetry
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=repo_dir)
  
    thread = Thread(target=_run_parallax, daemon=True)
    thread.start()
    
@app.function(
    gpu="A100-80GB:4",  # Must match above
    timeout=1800,
)
@modal.fastapi_endpoint(method="POST")
def chat(payload: dict):
    """
    Proxies requests to internal Parallax server.
    Fully streaming, OpenAI-compatible.
    """
    internal_url = "http://localhost:3001/v1/chat/completions"
    try:
        resp = requests.post(internal_url, json=payload, stream=True, timeout=1800)
        resp.raise_for_status()
    except Exception as e:
        return {"error": str(e)}, 500

    # Stream response back
    def generate():
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                yield chunk

    return generate(), resp.status_code, resp.headers.items()