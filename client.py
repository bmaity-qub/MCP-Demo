"""
MCP Client Demo.

Connects to server.py over stdio, discovers its tools at runtime (matches
"Live demo 01: Dynamic tool discovery"), calls search_documents with the
user's question, and passes the retrieved context to Groq to produce a
grounded answer (matches "Live demo 02: From question to grounded answer").

Why Groq: very low-latency inference (good for a live demo where a long
spinner kills the room), a generous free tier, and an OpenAI-compatible
API so the rest of this script barely changes.

Run this with:
    python client.py "How does MCP reduce integration maintenance?"

Requires:
    - server.py in the same folder (make sure it logs to stderr, not
      stdout -- stdout must carry ONLY JSON-RPC messages for MCP's stdio
      transport to parse correctly)
    - A free Groq API key: https://console.groq.com/keys
    - That key saved in a .env file (see .env.example) -- never typed
      live or hardcoded, so it's safe to have on screen during a demo.
"""

import os
import sys
import asyncio
import requests
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()  # reads .env in this folder, sets env vars from it

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
# Groq's free-tier catalog shifts without much notice (llama-3.1-8b-instant
# and llama-3.3-70b-versatile were both moved behind "Contact Sales" very
# recently, and existing free keys now get a 404 for them). Rather than
# pin one name and hope, try a short list of currently-live, non-reasoning
# instruct models in order -- first one that responds wins. Check
# https://console.groq.com/docs/models the morning of your KT and reorder
# this list if anything here has moved again.
GROQ_MODELS = [
    "openai/gpt-oss-20b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.1-8b-instant",  # left as a last-resort try, harmless if 404s
]


def ask_llm(question: str, context: str) -> str:
    """Send the retrieved context + question to Groq, trying GROQ_MODELS
    in order until one actually responds. Returns (answer, model_used)."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at "
            "https://console.groq.com/keys, then copy .env.example to .env "
            "and paste the key in there."
        )

    system_prompt = (
        "You are a helpful assistant. Answer the user's question using ONLY "
        "the provided context. If the context does not contain the answer, "
        "say so. Keep the answer to 2-3 sentences."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    last_error = None
    for model in GROQ_MODELS:
        try:
            response = requests.post(
                GROQ_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.2,
                },
                timeout=30,
            )
            if response.status_code == 404:
                # This specific model isn't available on this key/tier
                # right now -- try the next one instead of failing the demo.
                last_error = f"{model}: 404 (not available on this key)"
                continue
            response.raise_for_status()
            data = response.json()
            if "error" in data:
                last_error = f"{model}: {data['error']}"
                continue
            return data["choices"][0]["message"]["content"].strip(), model
        except requests.exceptions.RequestException as e:
            last_error = f"{model}: {e}"
            continue

    raise RuntimeError(
        f"All Groq models failed. Last error: {last_error}\n"
        "Run this to see what's actually live on your key:\n"
        '  curl https://api.groq.com/openai/v1/models '
        '-H "Authorization: Bearer $GROQ_API_KEY"'
    )


async def main(question: str):
    server_params = StdioServerParameters(
        command="python",
        args=["server.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 1. Initialize session
            await session.initialize()
            print("→ initialize session")

            # 2. Discover tools at runtime
            print("→ tools/list")
            tools_result = await session.list_tools()
            tool_names = [t.name for t in tools_result.tools]
            print(f"✓ discovered {len(tool_names)} tool(s): {tool_names}\n")

            # 3. Invoke the search_documents tool with the user's question
            print(f"INPUT\n{question}\n")
            print("TOOL TRACE")
            print("  search_documents")
            print(f"  query: {question}")
            print("  top_k: 5\n")

            result = await session.call_tool(
                "search_documents",
                arguments={"query": question, "top_k": 5},
            )
            retrieved_text = result.content[0].text
            print("RETRIEVED CONTEXT")
            print(retrieved_text + "\n")

            # 4. Compose the final, grounded answer via Groq
            try:
                answer, model_used = ask_llm(question, retrieved_text)
                print(f"GROUNDED OUTPUT  (model: {model_used})")
                print(answer)
            except RuntimeError as e:
                print("GROUNDED OUTPUT")
                print(f"[!] {e}")


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "How does MCP reduce integration maintenance?"
    asyncio.run(main(q))