"""
MCP Server: Document Search over a local FAISS index.

Exposes ONE tool over the Model Context Protocol:

    search_documents(query: str, top_k: int = 3) -> list of matching chunks

Run this with:
    python server.py

It communicates over stdio, which is what client.py expects. This file
itself needs no cloud service, API key, or cost to run -- retrieval is
fully local. (client.py separately calls Groq's free tier to generate the
final answer from what this server retrieves.)

NOTE: with stdio transport, stdout is reserved for JSON-RPC messages only.
All logging below goes to stderr.
"""

import os
import glob
import sys
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from mcp.server.fastmcp import FastMCP

DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")
CHUNK_SIZE = 500  # characters per chunk, kept simple for the demo
OVERLAP = 100     # characters shared between neighbouring chunks


def log(msg: str) -> None:
    """Log to stderr so stdout stays clean for MCP JSON-RPC."""
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# 1. Build a tiny in-memory FAISS index from the docs/ folder at startup.
# ---------------------------------------------------------------------------

log("[mcp-server] loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # small, free, runs on CPU

chunks = []
for path in glob.glob(os.path.join(DOCS_DIR, "*.txt")):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    for i in range(0, len(text), CHUNK_SIZE - OVERLAP):
        piece = text[i : i + CHUNK_SIZE].strip()
        if piece:
            chunks.append({"source": os.path.basename(path), "text": piece})

log(f"[mcp-server] indexing {len(chunks)} chunks from {DOCS_DIR}")

embeddings = embedder.encode([c["text"] for c in chunks], convert_to_numpy=True)
faiss.normalize_L2(embeddings)

dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)  # cosine similarity via normalized inner product
index.add(embeddings)

log("[mcp-server] index ready.")

# ---------------------------------------------------------------------------
# 2. Expose search as an MCP tool.
# ---------------------------------------------------------------------------

mcp = FastMCP("document-search")


@mcp.tool()
def search_documents(query: str, top_k: int = 5) -> str:
    """
    Search the internal knowledge base for chunks relevant to the query.

    Args:
        query: The natural-language question or search phrase.
        top_k: Number of top matching chunks to return (default 3).

    Returns:
        A formatted string of the top matching chunks with their source file.
    """
    q_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)

    scores, idxs = index.search(q_emb, top_k)

    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], idxs[0]), start=1):
        if idx == -1:
            continue
        chunk = chunks[idx]
        results.append(
            f"[{rank}] (source: {chunk['source']}, score: {score:.3f})\n{chunk['text']}"
        )

    if not results:
        return "No relevant documents found."

    return "\n\n".join(results)


if __name__ == "__main__":
    log("[mcp] server ready on stdio")
    mcp.run(transport="stdio")