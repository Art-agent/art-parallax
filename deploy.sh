#!/bin/bash

# Parallax Modal Deployment Script
# This script helps deploy Parallax on Modal with various configurations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
ACTION="deploy"
WORKERS=2
MODEL=""
GPU_TYPE="A10G"
ENVIRONMENT="production"

# Function to print colored output
print_color() {
    color=$1
    message=$2
    echo -e "${color}${message}${NC}"
}

# Function to show help
show_help() {
    echo "Parallax Modal Deployment Script"
    echo ""
    echo "Usage: ./deploy.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -a, --action ACTION       Action to perform (deploy|test|download|stop|logs|scale)"
    echo "  -w, --workers COUNT       Number of worker instances (default: 2)"
    echo "  -m, --model MODEL_NAME    Model to download"
    echo "  -g, --gpu GPU_TYPE        GPU type (T4|A10G|A100|H100, default: A10G)"
    echo "  -e, --env ENVIRONMENT     Environment (development|production, default: production)"
    echo "  -h, --help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./deploy.sh                           # Deploy with defaults"
    echo "  ./deploy.sh -a deploy -w 4 -g A100    # Deploy with 4 workers on A100 GPUs"
    echo "  ./deploy.sh -a download -m 'meta-llama/Llama-2-13b-chat-hf'"
    echo "  ./deploy.sh -a test                   # Test deployment"
    echo "  ./deploy.sh -a logs                   # View logs"
    echo "  ./deploy.sh -a scale -w 10            # Scale to 10 workers"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -a|--action)
            ACTION="$2"
            shift 2
            ;;
        -w|--workers)
            WORKERS="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -g|--gpu)
            GPU_TYPE="$2"
            shift 2
            ;;
        -e|--env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check if Modal is installed
check_modal() {
    if ! command -v modal &> /dev/null; then
        print_color "$RED" "Error: Modal CLI is not installed."
        echo "Please install it with: pip install modal"
        exit 1
    fi
}

# Check if authenticated
check_auth() {
    if ! modal token id &> /dev/null; then
        print_color "$YELLOW" "Not authenticated with Modal."
        echo "Running modal setup..."
        modal setup
    fi
}

# Update modal_app.py with GPU type
update_gpu_config() {
    if [ "$GPU_TYPE" != "A10G" ]; then
        print_color "$YELLOW" "Updating GPU configuration to $GPU_TYPE..."
        sed -i.bak "s/gpu=\"A10G\"/gpu=\"$GPU_TYPE\"/g" modal_app.py
        print_color "$GREEN" "✓ GPU configuration updated"
    fi
}

# Deploy action
deploy() {
    print_color "$GREEN" "🚀 Deploying Parallax on Modal..."
    echo "Configuration:"
    echo "  - Environment: $ENVIRONMENT"
    echo "  - Workers: $WORKERS"
    echo "  - GPU Type: $GPU_TYPE"
    echo ""

    update_gpu_config

    if [ "$ENVIRONMENT" == "development" ]; then
        print_color "$YELLOW" "Starting in development mode (live reload)..."
        modal serve modal_app.py
    else
        print_color "$YELLOW" "Deploying to production..."
        modal deploy modal_app.py

        print_color "$YELLOW" "Starting workers..."
        modal run modal_app.py --action deploy --workers "$WORKERS"

        print_color "$GREEN" "✅ Deployment complete!"
        echo ""
        echo "Access your deployment at:"
        echo "  API: https://your-username-parallax-scheduler.modal.run"
        echo "  Chat: https://your-username-parallax-chat.modal.run"
    fi
}

# Test action
test() {
    print_color "$YELLOW" "🧪 Testing Parallax deployment..."
    modal run modal_app.py --action test
    print_color "$GREEN" "✅ Tests complete!"
}

# Download model action
download_model() {
    if [ -z "$MODEL" ]; then
        MODEL="meta-llama/Llama-2-7b-chat-hf"
    fi

    print_color "$YELLOW" "📥 Downloading model: $MODEL"
    modal run modal_app.py --action download-model --model "$MODEL"
    print_color "$GREEN" "✅ Model downloaded successfully!"
}

# Stop deployment
stop() {
    print_color "$YELLOW" "🛑 Stopping Parallax deployment..."
    modal app stop parallax-llm-inference
    print_color "$GREEN" "✅ Deployment stopped!"
}

# View logs
view_logs() {
    print_color "$YELLOW" "📋 Viewing logs..."
    modal logs -f parallax-llm-inference
}

# Scale workers
scale() {
    print_color "$YELLOW" "⚖️ Scaling to $WORKERS workers..."

    # Create a temporary Python script for scaling
    cat > scale_workers.py << EOF
import modal
from modal_app import app, ParallaxWorker

@app.function()
def scale_workers():
    workers = []
    for i in range($WORKERS):
        worker = ParallaxWorker()
        workers.append(worker)
        print(f"Worker {i+1} deployed")
    return f"Scaled to $WORKERS workers"

if __name__ == "__main__":
    with app.run():
        result = scale_workers.remote()
        print(result)
EOF

    modal run scale_workers.py
    rm scale_workers.py

    print_color "$GREEN" "✅ Scaled to $WORKERS workers!"
}

# Main execution
main() {
    check_modal
    check_auth

    case $ACTION in
        deploy)
            deploy
            ;;
        test)
            test
            ;;
        download)
            download_model
            ;;
        stop)
            stop
            ;;
        logs)
            view_logs
            ;;
        scale)
            scale
            ;;
        *)
            print_color "$RED" "Unknown action: $ACTION"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main