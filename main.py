"""
serve/main.py
─────────────
FastAPI server — OpenAI-compatible streaming + non-streaming chat.
Wraps llama-cpp-python running the Gemma 3 27B GGUF on RTX 3060.

Install:
  pip install fastapi uvicorn "llama-cpp-python[server]"

Run:
  python main.py --model ~/gemma3-forge-Q4_K_M.gguf [--gpu-layers 25]
"""

import argparse
import json
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from llama_cpp import Llama
from pydantic import BaseModel, Field

# ═════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ─────────────
# This is the single most impactful lever for output quality.
# Rules are ordered by priority. Be explicit about EVERYTHING.
# ═════════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are Forge, an elite precision coding assistant with deep expertise in Python, JavaScript, Java, C++, and C.

## ABSOLUTE CODE RULES — never break these:
1. Every function signature MUST include ALL parameter names. Never write `def func(arr,):` — always write `def func(arr, target):`.
2. Every comparison MUST have both sides. Never write `if x ==:` — always write `if x == target:`.
3. Every variable used in the body MUST be declared in the signature or defined before use.
4. Code blocks MUST be syntactically complete and immediately executable. No ellipsis, no placeholders.
5. Never write TODO, FIXME, pass (unless it is the intended no-op), or "implement this".
6. All code MUST be inside a properly fenced code block with the language tag: ```python, ```javascript, etc.
7. Function and variable names MUST be descriptive and follow the language's convention (snake_case for Python, camelCase for JavaScript, PascalCase for Java classes).

## RESPONSE STRUCTURE — always follow this exact order:
1. **One-sentence summary** of what you are about to write or what the bug is.
2. **Complete code** in a fenced block — no truncation, no line skipping.
3. **Brief explanation** (3–6 bullet points) of the key decisions made.
4. **Edge cases** — list at least 2 inputs that need special handling, and show how your code handles them.

## WHEN WRITING ALGORITHMS:
- State the time complexity (Big-O) and space complexity on the line immediately after the code block.
- Show a concrete worked example: input → expected output → your function's output.
- If the algorithm has a known optimal version, implement that, not a naive version.

## WHEN DEBUGGING:
- Line 1: "**Root cause:**" followed by the exact bug in one sentence.
- Then show the corrected code with inline comments marking every changed line with `# FIXED:`.
- Then explain WHY the bug occurred (not just what it was).

## LANGUAGE AND FORMATTING:
- Write in clear, grammatically correct English. No abbreviations like "pls", "u", "bc".
- Use proper punctuation. End all sentences with a period.
- Mathematical notation: use proper symbols — O(n log n), not O(n log n) or O(nlogn).
- When stating complexity, always write both time AND space: "Time: O(n log n), Space: O(1)".

## WHAT YOU NEVER DO:
- Never produce partial functions that trail off mid-definition.
- Never skip lines with a comment like "# rest of the code here".
- Never write a variable on the right-hand side of an assignment that was not previously defined.
- Never assume the user will fill anything in — your code must run as-is."""

# ═════════════════════════════════════════════════════════════════════════════
# SAMPLING PARAMETERS — tuned specifically for code generation quality
# ═════════════════════════════════════════════════════════════════════════════
# temperature=0.1   → highly deterministic; correct variable names every time
#                     (0.2 adds randomness that corrupts identifiers like "target")
# top_k=40          → explicitly cap to top-40 tokens; blocks garbage candidates
# top_p=0.95        → nucleus sampling; combined with top_k for dual filtering
# min_p=0.05        → reject any token below 5% of the top token's probability
#                     THIS is what fixes the "def_search(arr,):" class of bugs —
#                     the missing identifier tokens have very low probability and
#                     min_p forces the model to pick the next most likely token
# repeat_penalty=1.1 → gentle repetition penalty; prevents looping on same token
CODE_SAMPLING = dict(
    temperature=0.1,
    top_k=40,
    top_p=0.95,
    min_p=0.05,
    repeat_penalty=1.1,
)

# ── Server defaults ───────────────────────────────────────────────────────────
DEFAULT_MODEL = "./output/gemma3-forge-Q4_K_M.gguf"
DEFAULT_PORT  = 8080
CONTEXT_SIZE  = 4096
GPU_LAYERS    = 25
N_THREADS     = 8

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Forge API",
    version="1.0.0",
    description="Local Gemma 3 27B coding assistant ",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

llm: Optional[Llama] = None


# ── Request models ────────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages:       list[Message]
    max_tokens:     int   = Field(default=1_024, ge=1,   le=4_096)
    temperature:    float = Field(default=0.1,   ge=0.0, le=2.0)   # default now 0.1
    top_p:          float = Field(default=0.95,  ge=0.0, le=1.0)
    top_k:          int   = Field(default=40,    ge=1)
    min_p:          float = Field(default=0.05,  ge=0.0, le=1.0)
    repeat_penalty: float = Field(default=1.1,   ge=1.0, le=2.0)
    stream:         bool  = True
    stop:           list[str] = []


# ═════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDER
# ──────────────
# ROOT CAUSE OF MISSING VARIABLE NAMES:
# The GGUF was exported with a tokenizer hotpatch that forced Gemma 3 to use
# the llama-bpe vocabulary (notebook Cell 6: return "llama-bpe").
# Gemma 3 uses its own SentencePiece vocab. Tokens valid in Gemma's vocab but
# absent in llama-bpe decode to empty string → variable names like "distances",
# "neighbor", "weight" vanish silently from the output.
#
# FIX: bypass chat_format="gemma" (which relies on the broken tokenizer path)
# and manually format the prompt using Gemma 3's raw special tokens.
# llama-cpp-python passes this as a raw string prompt to create_completion(),
# which skips the broken token-id-to-string mapping used by create_chat_completion().
# ═════════════════════════════════════════════════════════════════════════════

# Gemma 3 special tokens (hardcoded — not read from tokenizer_config.json)
_BOS            = "<bos>"
_START_OF_TURN  = "<start_of_turn>"
_END_OF_TURN    = "<end_of_turn>"
_USER           = "user"
_MODEL          = "model"

def _build_raw_prompt(request: ChatRequest) -> str:
    """
    Format conversation as a raw Gemma 3 prompt string.
    Using create_completion() with this raw string instead of
    create_chat_completion() bypasses the broken llama-bpe token mapping
    that was causing variable names to be silently dropped.

    Gemma 3 format:
      <bos><start_of_turn>user
      {system}

      {user_msg}<end_of_turn>
      <start_of_turn>model
      {assistant_msg}<end_of_turn>
      ...
      <start_of_turn>model

    The system prompt is merged into the first user turn because Gemma 3
    has no dedicated system role in its template.
    """
    parts = [_BOS]
    first_user = True

    for m in request.messages:
        role    = m.role
        content = m.content

        if role == "system":
            continue   # handled by prepending to first user turn below

        if role == "user":
            if first_user:
                # Merge system prompt into first user message
                merged = f"{SYSTEM_PROMPT}\n\n{content}"
                first_user = False
            else:
                merged = content
            parts.append(
                f"{_START_OF_TURN}{_USER}\n{merged}{_END_OF_TURN}\n"
            )

        elif role == "assistant":
            parts.append(
                f"{_START_OF_TURN}{_MODEL}\n{content}{_END_OF_TURN}\n"
            )

    # Open the model's turn — it will continue from here
    parts.append(f"{_START_OF_TURN}{_MODEL}\n")
    return "".join(parts)


def _sampling_kwargs(request: ChatRequest) -> dict:
    return dict(
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        min_p=request.min_p,
        repeat_penalty=request.repeat_penalty,
        max_tokens=request.max_tokens,
        stop=request.stop or [_END_OF_TURN],
    )


# ── Corruption detector ───────────────────────────────────────────────────────
import re as _re

_CORRUPTION_PATTERNS = [
    _re.compile(r"for\s*,\s*in\s"),         # for, in   (missing loop vars)
    _re.compile(r"\w+\s*=\s*\[[\w\]]*\]\s*\+\s*$", _re.M),  # x = [y] +  (truncated expr)
    _re.compile(r"if\s+\w+\s*[<>=!]+\s*\["), # if x <[  (missing rhs variable)
    _re.compile(r"^\s*return\s*$", _re.M),   # bare return in non-void fn
    _re.compile(r"lambda\s+\w+\s*:\s*\["),   # lambda x: [  (missing dict ref)
    _re.compile(r"def\s+\w+\s*\(\s*,"),      # def func(,  (missing first param)
]

def _is_corrupted(text: str) -> tuple[bool, str]:
    """Return (is_corrupted, reason) for the response text."""
    for pat in _CORRUPTION_PATTERNS:
        m = pat.search(text)
        if m:
            return True, f"Corruption pattern detected: {repr(m.group()[:60])}"
    return False, ""


# ── Streaming SSE generator ───────────────────────────────────────────────────
async def _sse_stream(request: ChatRequest) -> AsyncGenerator[str, None]:
    rid     = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    prompt  = _build_raw_prompt(request)

    stream = llm.create_completion(
        prompt=prompt,
        stream=True,
        echo=False,
        **_sampling_kwargs(request),
    )

    for chunk in stream:
        content = chunk["choices"][0].get("text", "")
        finish  = chunk["choices"][0].get("finish_reason")

        payload = {
            "id": rid, "object": "chat.completion.chunk",
            "created": created, "model": "gemma3-forge",
            "choices": [{"index": 0,
                         "delta": {"content": content} if content else {},
                         "finish_reason": finish}],
        }
        yield f"data: {json.dumps(payload)}\n\n"

    yield "data: [DONE]\n\n"


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": "gemma3-forge",
            "gpu_layers": GPU_LAYERS, "ctx": CONTEXT_SIZE,
            "prompt_mode": "raw_gemma3_template"}

@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": "gemma3-forge", "object": "model"}]}

@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    if llm is None:
        raise HTTPException(503, "Model not loaded")

    if request.stream:
        return StreamingResponse(
            _sse_stream(request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ── Non-streaming (evaluate.py, curl, etc.) ───────────────────────────────
    prompt = _build_raw_prompt(request)
    result = llm.create_completion(
        prompt=prompt,
        stream=False,
        echo=False,
        **_sampling_kwargs(request),
    )

    # Detect tokenizer corruption and add a warning header if found
    text = result["choices"][0].get("text", "")
    corrupted, reason = _is_corrupted(text)
    if corrupted:
        print(f"  ⚠  Tokenizer corruption detected: {reason}")
        print(f"     This means the GGUF needs re-exporting with a clean llama.cpp.")
        print(f"     See: https://github.com/ggerganov/llama.cpp (use b3447+)")

    # Reformat to look like a chat completion response for compatibility
    return {
        "id":      f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object":  "chat.completion",
        "created": int(time.time()),
        "model":   "gemma3-forge",
        "choices": [{
            "index":         0,
            "message":       {"role": "assistant", "content": text},
            "finish_reason": result["choices"][0].get("finish_reason"),
        }],
        "usage": result.get("usage", {}),
    }


# ── Model loader ──────────────────────────────────────────────────────────────
def _load_model(model_path: str):
    global llm
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Model not found: {p}\n"
            f"  Run merge_and_export.py on the training machine,\n"
            f"  download the .gguf, then point --model at it."
        )

    size_gb = p.stat().st_size / 1e9
    print(f"\n  File       : {p.name}  ({size_gb:.1f} GB)")
    print(f"  GPU layers : {GPU_LAYERS}")
    print(f"  Context    : {CONTEXT_SIZE} tokens")
    print(f"  Threads    : {N_THREADS}")
    print(f"  Prompt mode: raw Gemma 3 template ")

    # chat_format=None — we format the prompt manually via _build_raw_prompt()
    # This bypasses the broken tokenizer path caused by the GGUF export hotpatch.
    llm = Llama(
        model_path=str(p),
        n_ctx=CONTEXT_SIZE,
        n_gpu_layers=GPU_LAYERS,
        n_threads=N_THREADS,
        verbose=False,
        chat_format=None,      # ← was "gemma" — caused token drops via bad vocab mapping
    )
    print("  ✓ Model loaded and ready")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    global GPU_LAYERS, CONTEXT_SIZE
    
    ap = argparse.ArgumentParser(description="Forge inference server")
    ap.add_argument("--model",      default=DEFAULT_MODEL)
    ap.add_argument("--port",       default=DEFAULT_PORT,  type=int)
    ap.add_argument("--host",       default="127.0.0.1")
    ap.add_argument("--gpu-layers", default=GPU_LAYERS,    type=int, dest="gpu_layers",
                    help="Layers to offload to GPU. RTX 3060 12 GB → 25. Raise if under budget.")
    ap.add_argument("--ctx",        default=CONTEXT_SIZE,  type=int,
                    help="Context window size (default 4096)")
    args = ap.parse_args()

    GPU_LAYERS   = args.gpu_layers
    CONTEXT_SIZE = args.ctx

    print("=" * 52)
    print("  Forge — Coding Assistant Server  (Gemma 3 27B)")
    print("=" * 52)
    _load_model(args.model)
    print(f"\n  Streaming  : http://{args.host}:{args.port}/v1/chat/completions")
    print(f"  Health     : http://{args.host}:{args.port}/health")
    print(f"  API docs   : http://{args.host}:{args.port}/docs\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
