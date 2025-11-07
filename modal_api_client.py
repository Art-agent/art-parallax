"""
Example client for interacting with Parallax deployed on Modal

This script demonstrates how to call the Parallax API from your agent
or any Python application.

Usage:
    python modal_api_client.py

You can integrate this into your agent by importing the ParallaxClient class.
"""

import os
import json
import asyncio
from typing import List, Dict, Optional, AsyncGenerator
import httpx
from dataclasses import dataclass


@dataclass
class Message:
    role: str
    content: str


class ParallaxClient:
    """Client for interacting with Parallax on Modal"""

    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
        timeout: int = 120,
    ):
        """
        Initialize Parallax client

        Args:
            base_url: Modal deployment URL (e.g., https://username-parallax-scheduler.modal.run)
            api_key: API key if authentication is enabled
            timeout: Request timeout in seconds
        """
        self.base_url = base_url or os.environ.get(
            "PARALLAX_API_URL",
            "https://your-username-parallax-scheduler.modal.run"
        )
        self.api_key = api_key or os.environ.get("PARALLAX_API_KEY")
        self.timeout = timeout

        # Remove trailing slash
        self.base_url = self.base_url.rstrip("/")

        # Setup headers
        self.headers = {"Content-Type": "application/json"}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

    async def health_check(self) -> Dict:
        """Check if the API is healthy"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()

    async def list_models(self) -> List[Dict]:
        """List available models"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                f"{self.base_url}/v1/models",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()["data"]

    async def chat_completion(
        self,
        messages: List[Message],
        model: str = "llama-2-7b-chat",
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.9,
        stream: bool = False,
    ) -> Dict:
        """
        Generate a chat completion

        Args:
            messages: List of conversation messages
            model: Model to use for generation
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            stream: Whether to stream the response

        Returns:
            Completion response dict
        """
        # Convert messages to dict format
        messages_dict = [{"role": m.role, "content": m.content} for m in messages]

        request_data = {
            "model": model,
            "messages": messages_dict,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": stream,
        }

        if stream:
            return await self._stream_completion(request_data)
        else:
            return await self._regular_completion(request_data)

    async def _regular_completion(self, request_data: Dict) -> Dict:
        """Handle non-streaming completion"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/v1/chat/completions",
                json=request_data,
                headers=self.headers,
            )
            response.raise_for_status()
            return response.json()

    async def _stream_completion(self, request_data: Dict) -> AsyncGenerator:
        """Handle streaming completion"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json=request_data,
                headers=self.headers,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data != "[DONE]":
                            try:
                                yield json.loads(data)
                            except json.JSONDecodeError:
                                continue

    async def generate(
        self,
        prompt: str,
        model: str = "llama-2-7b-chat",
        temperature: float = 0.7,
        max_tokens: int = 512,
        system_prompt: str = None,
    ) -> str:
        """
        Simple generation helper

        Args:
            prompt: User prompt
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            system_prompt: Optional system prompt

        Returns:
            Generated text
        """
        messages = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=prompt))

        response = await self.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )

        return response["choices"][0]["message"]["content"]

    async def stream_generate(
        self,
        prompt: str,
        model: str = "llama-2-7b-chat",
        temperature: float = 0.7,
        max_tokens: int = 512,
        system_prompt: str = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream generation helper

        Args:
            prompt: User prompt
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            system_prompt: Optional system prompt

        Yields:
            Generated text chunks
        """
        messages = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=prompt))

        async for chunk in await self.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        ):
            if chunk.get("choices") and chunk["choices"][0].get("delta"):
                content = chunk["choices"][0]["delta"].get("content", "")
                if content:
                    yield content


# Example usage functions
async def example_basic():
    """Basic example of using the client"""
    client = ParallaxClient()

    # Check health
    print("Checking API health...")
    health = await client.health_check()
    print(f"Health: {health}")

    # Generate text
    print("\nGenerating text...")
    response = await client.generate(
        prompt="Write a short poem about distributed computing",
        temperature=0.8,
        max_tokens=100,
    )
    print(f"Response: {response}")


async def example_streaming():
    """Example of streaming generation"""
    client = ParallaxClient()

    print("Streaming generation...")
    async for chunk in client.stream_generate(
        prompt="Tell me a story about a robot learning to code",
        temperature=0.7,
        max_tokens=200,
    ):
        print(chunk, end="", flush=True)
    print()


async def example_conversation():
    """Example of multi-turn conversation"""
    client = ParallaxClient()

    messages = [
        Message(role="system", content="You are a helpful AI assistant."),
        Message(role="user", content="What is Modal?"),
        Message(role="assistant", content="Modal is a cloud platform for running Python code, particularly ML workloads, with automatic scaling and GPU support."),
        Message(role="user", content="How does it compare to AWS Lambda?"),
    ]

    print("Continuing conversation...")
    response = await client.chat_completion(
        messages=messages,
        temperature=0.7,
        max_tokens=200,
    )

    print(f"Response: {response['choices'][0]['message']['content']}")


async def example_for_agent():
    """Example specifically for agent integration"""

    # Initialize client with your deployment URL
    client = ParallaxClient(
        base_url="https://your-username-parallax-scheduler.modal.run",
        api_key="your-api-key-if-needed"  # Only if auth is enabled
    )

    # Your agent's context
    agent_context = """
    You are an AI agent helping users with coding tasks.
    Current task: Implement a sorting algorithm
    """

    # User request
    user_request = "Can you help me implement quicksort in Python?"

    # Generate response
    response = await client.generate(
        prompt=user_request,
        system_prompt=agent_context,
        temperature=0.3,  # Lower temperature for code generation
        max_tokens=500,
    )

    print("Agent response:")
    print(response)

    return response


# Integration class for your agent
class ParallaxAgentIntegration:
    """
    Integration class for using Parallax in your agent

    Example usage in your agent:
        llm = ParallaxAgentIntegration()
        response = await llm.think("How do I solve this problem?")
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize with optional configuration"""
        config = config or {}
        self.client = ParallaxClient(
            base_url=config.get("base_url"),
            api_key=config.get("api_key"),
        )
        self.model = config.get("model", "llama-2-7b-chat")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 512)

    async def think(self, prompt: str, context: str = None) -> str:
        """
        Agent thinking/reasoning method

        Args:
            prompt: The question or task
            context: Additional context for the agent

        Returns:
            The agent's response
        """
        return await self.client.generate(
            prompt=prompt,
            system_prompt=context,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    async def analyze_code(self, code: str, question: str) -> str:
        """Analyze code and answer questions about it"""
        prompt = f"""
        Analyze the following code and answer the question.

        Code:
        ```
        {code}
        ```

        Question: {question}
        """
        return await self.think(prompt, "You are a code analysis expert.")

    async def generate_code(self, description: str, language: str = "python") -> str:
        """Generate code based on description"""
        prompt = f"Generate {language} code for: {description}"
        return await self.think(
            prompt,
            f"You are an expert {language} programmer. Generate clean, efficient code."
        )

    async def fix_error(self, code: str, error: str) -> str:
        """Fix an error in code"""
        prompt = f"""
        Fix the error in this code:

        Code:
        ```
        {code}
        ```

        Error:
        {error}
        """
        return await self.think(prompt, "You are a debugging expert.")


async def main():
    """Run examples"""
    print("=" * 50)
    print("Parallax Modal Client Examples")
    print("=" * 50)

    # Uncomment the examples you want to run:

    # await example_basic()
    # await example_streaming()
    # await example_conversation()
    await example_for_agent()


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())