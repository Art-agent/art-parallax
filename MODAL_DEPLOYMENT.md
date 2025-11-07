# Deploying Parallax on Modal

This guide provides comprehensive instructions for deploying the Parallax distributed LLM inference engine on Modal.

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Configuration](#configuration)
5. [Deployment Options](#deployment-options)
6. [API Usage](#api-usage)
7. [Monitoring & Management](#monitoring--management)
8. [Troubleshooting](#troubleshooting)

## Overview

Parallax is deployed on Modal as a distributed system with three main components:

- **Scheduler**: Central orchestration server (FastAPI)
- **Workers**: GPU/CPU inference nodes
- **Chat UI**: React frontend interface

Modal provides:
- Automatic scaling and GPU provisioning
- Persistent model storage via Volumes
- Shared state via Network File System
- Secure secrets management
- Public HTTPS endpoints

## Prerequisites

1. **Install Modal CLI**:
```bash
pip install modal
```

2. **Authenticate with Modal**:
```bash
modal setup
```

3. **Create a Modal account** at [modal.com](https://modal.com)

4. **Optional: HuggingFace Token** (for gated models):
   - Get token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Add to Modal secrets (see Configuration section)

## Quick Start

### 1. Deploy with Default Settings

```bash
# Deploy all components
modal deploy modal_app.py

# Or run interactively
modal run modal_app.py --action deploy --workers 2
```

### 2. Download Models

```bash
# Download default model (Llama-2-7b-chat)
modal run modal_app.py --action download-model

# Download specific model
modal run modal_app.py --action download-model --model "mistralai/Mixtral-8x7B-v0.1"
```

### 3. Test the Deployment

```bash
# Run built-in tests
modal run modal_app.py --action test
```

### 4. Access Your Deployment

Your deployment will be available at:
- **API Endpoint**: `https://<your-username>-parallax-scheduler.modal.run`
- **Chat Interface**: `https://<your-username>-parallax-chat.modal.run`
- **Health Check**: `https://<your-username>-parallax-scheduler.modal.run/health`

## Configuration

### Environment Variables

Create secrets in the Modal dashboard:

1. Go to [modal.com/secrets](https://modal.com/secrets)
2. Create a new secret named `parallax-config`
3. Add these environment variables:

```env
# Required
PARALLAX_PORT=3001
JWT_SECRET=your-secure-jwt-secret-here

# Optional - for authenticated models
HF_TOKEN=your-huggingface-token

# Optional - customization
ENABLE_AUTH=true
MAX_BATCH_SIZE=32
MAX_SEQ_LENGTH=2048
MODEL_NAME=meta-llama/Llama-2-7b-chat-hf

# Optional - P2P networking
RELAY_SERVER_URL=wss://your-relay.example.com
P2P_ENABLED=true
```

### GPU Configuration

Edit `modal_app.py` to adjust GPU types:

```python
# For different GPU types
@app.cls(
    gpu="A100",  # Options: "T4", "A10G", "A100", "H100"
    # Or specify count
    gpu=modal.gpu.A100(count=2),  # Multiple GPUs
)
```

### Resource Limits

Adjust in `modal_app.py`:

```python
@app.cls(
    # Concurrency settings
    concurrency_limit=10,  # Max concurrent requests
    container_idle_timeout=600,  # Seconds before shutdown
    allow_concurrent_inputs=100,  # Max queued requests

    # Memory (optional)
    memory=32768,  # MB of RAM

    # CPU (optional)
    cpu=8,  # Number of CPU cores
)
```

## Deployment Options

### Option 1: Simple API Server

Deploy just the API without UI:

```python
# In modal_app.py, modify main() function
def main():
    scheduler = ParallaxScheduler()
    worker = ParallaxWorker()
    print("API deployed at: https://...")
```

### Option 2: Multi-Worker Setup

Deploy with multiple workers for higher throughput:

```bash
modal run modal_app.py --workers 5
```

### Option 3: Development Mode

For testing with live updates:

```bash
# Deploy with hot reload
modal serve modal_app.py

# Watch logs
modal logs -f
```

### Option 4: Custom Model Deployment

```python
# Create a custom deployment script
import modal
from modal_app import app, ParallaxWorker

@app.function()
def deploy_custom():
    # Deploy with specific model
    worker = ParallaxWorker()
    worker.load_model.remote("your-custom/model-name")
```

## API Usage

### OpenAI-Compatible Endpoint

```python
import openai

# Configure client
client = openai.OpenAI(
    base_url="https://your-parallax-scheduler.modal.run/v1",
    api_key="your-api-key"  # If auth is enabled
)

# Chat completion
response = client.chat.completions.create(
    model="llama-2-7b-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! How are you?"}
    ],
    temperature=0.7,
    max_tokens=150,
    stream=True  # Streaming supported
)

for chunk in response:
    print(chunk.choices[0].delta.content, end="")
```

### Direct HTTP API

```bash
# Chat completion
curl -X POST https://your-parallax-scheduler.modal.run/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-2-7b-chat",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'

# Health check
curl https://your-parallax-scheduler.modal.run/health

# List models
curl https://your-parallax-scheduler.modal.run/v1/models
```

### Streaming Example

```python
import requests
import json

url = "https://your-parallax-scheduler.modal.run/v1/chat/completions"
headers = {"Content-Type": "application/json"}
data = {
    "model": "llama-2-7b-chat",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": True
}

with requests.post(url, json=data, headers=headers, stream=True) as response:
    for line in response.iter_lines():
        if line:
            if line.startswith(b"data: "):
                data = json.loads(line[6:])
                if data != "[DONE]":
                    print(data["choices"][0]["delta"].get("content", ""), end="")
```

## Monitoring & Management

### View Logs

```bash
# All logs
modal logs

# Follow logs
modal logs -f

# Specific function logs
modal logs ParallaxScheduler
```

### Monitor Resources

```bash
# View app status
modal app list

# Get deployment info
modal app describe parallax-llm-inference
```

### Scale Workers

```python
# In modal_app.py, add scaling function
@app.function()
def scale_workers(count: int):
    """Dynamically scale worker count"""
    for i in range(count):
        ParallaxWorker().spawn()
    return f"Scaled to {count} workers"
```

### Stop Deployment

```bash
# Stop all containers
modal app stop parallax-llm-inference
```

## Troubleshooting

### Common Issues

**1. GPU Out of Memory**
```python
# Reduce batch size in configuration
MAX_BATCH_SIZE=16
MAX_SEQ_LENGTH=1024
```

**2. Model Download Fails**
```bash
# Check HuggingFace token
modal secret create parallax-config HF_TOKEN=your-token

# Retry with specific cache
modal run modal_app.py --action download-model --model "model-name"
```

**3. Connection Timeouts**
```python
# Increase timeout in modal_app.py
container_idle_timeout=1200  # 20 minutes
```

**4. Authentication Errors**
```bash
# Regenerate JWT secret
modal secret update parallax-config JWT_SECRET=new-secret-key
```

### Debug Mode

Add debug logging:

```python
@modal.enter()
def startup(self):
    import logging
    logging.basicConfig(level=logging.DEBUG)
    # ... rest of initialization
```

### Performance Tuning

For optimal performance:

1. **Use appropriate GPU**: A100 for large models, T4 for cost-efficiency
2. **Enable caching**: Models are cached in volumes
3. **Batch requests**: Use MAX_BATCH_SIZE for throughput
4. **Monitor usage**: Check Modal dashboard for metrics

## Advanced Features

### Custom Inference Pipeline

```python
@app.function(gpu="A100")
def custom_inference(prompt: str, pipeline_config: dict):
    """Custom inference with specific settings"""
    from transformers import pipeline

    # Custom pipeline
    pipe = pipeline(
        "text-generation",
        model=pipeline_config["model"],
        device_map="auto",
        torch_dtype="auto",
    )

    result = pipe(prompt, **pipeline_config["params"])
    return result
```

### Multi-Model Deployment

```python
# Deploy multiple models
models = [
    "meta-llama/Llama-2-7b-chat-hf",
    "mistralai/Mixtral-8x7B-v0.1",
    "microsoft/phi-2"
]

for model in models:
    download_model.remote(model)
```

### Webhook Integration

```python
@app.function()
async def webhook_handler(request: dict):
    """Handle webhooks for async processing"""
    # Process request
    result = await process_inference(request["prompt"])

    # Send to callback URL
    import httpx
    async with httpx.AsyncClient() as client:
        await client.post(request["callback_url"], json=result)
```

## Cost Optimization

1. **Use spot instances** when available
2. **Set appropriate idle timeouts**
3. **Use T4 GPUs for smaller models**
4. **Enable request batching**
5. **Cache models in volumes**

## Security Best Practices

1. **Always use secrets** for sensitive data
2. **Enable authentication** in production
3. **Use HTTPS endpoints** (Modal provides by default)
4. **Rotate JWT secrets** regularly
5. **Limit concurrency** to prevent abuse

## Support

- **Modal Documentation**: [docs.modal.com](https://docs.modal.com)
- **Modal Discord**: [discord.gg/modal](https://discord.gg/modal)
- **Parallax Issues**: GitHub Issues
- **Modal Support**: support@modal.com

## Next Steps

1. Deploy your first model
2. Test the API endpoints
3. Configure authentication
4. Set up monitoring
5. Scale based on usage

Happy deploying! 🚀