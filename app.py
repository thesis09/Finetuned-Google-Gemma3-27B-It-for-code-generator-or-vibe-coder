"""
app.py
──────
Streamlit Web UI for the Forge coding assistant.
Connects to serve/main.py via streaming HTTP.

Install:
  pip install streamlit requests

Run (with main.py already running):
  streamlit run app.py
"""

import json
import re

import requests
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
SERVER_URL = "http://127.0.0.1:8080"
CHAT_URL   = f"{SERVER_URL}/v1/chat/completions"
TIMEOUT    = (10, 300)   # (connect, read) seconds — never None

# ── Sampling params ───────────────────────────────────────────────────────────
DEFAULT_TEMPERATURE    = 0.1
DEFAULT_TOP_K          = 40
DEFAULT_TOP_P          = 0.95
DEFAULT_MIN_P          = 0.05
DEFAULT_REPEAT_PENALTY = 1.1

# ── Code-request detector ─────────────────────────────────────────────────────
CODE_KEYWORDS = {
    "write", "implement", "create", "build", "make",
    "code", "function", "algorithm", "class", "method",
    "debug", "fix", "find the bug", "what's wrong",
    "explain", "how does", "sort", "search", "tree",
    "binary", "linked list", "stack", "queue", "graph",
}

def _is_code_request(text: str) -> bool:
    lowered = text.lower()
    return any(kw in lowered for kw in CODE_KEYWORDS)

def _augment_prompt(text: str) -> str:
    """Append completeness instructions for code requests."""
    if not _is_code_request(text):
        return text
    return (
        text.strip() + "\n\n"
        "Important: Write the COMPLETE implementation with ALL parameter names "
        "in the function signature. Every variable used inside the function MUST "
        "appear in the signature or be defined before use. "
        "Do not truncate, skip, or use placeholders. "
        "The code must be immediately runnable."
    )

# ── Code detection — pattern-based, not backtick-only ─────────────────────────
# FIX: old check was `"```" in full_response` which is False when the model
# outputs correct code without fences → warning fired every single time.
# New check looks for actual code syntax patterns in the response text.
_CODE_LINE_RE = re.compile(
    r'^(\s*)(def |class |for |while |if |elif |else:|return |import |from |'
    r'try:|except|finally:|with |async |await |yield |'
    r'#include|public |private |static |void |int |float |char |bool |'
    r'function |const |let |var )',
    re.MULTILINE,
)

def _contains_code(text: str) -> bool:
    """Return True if response has fenced blocks OR any recognisable code lines."""
    return "```" in text or bool(_CODE_LINE_RE.search(text))

# ── Language sniffer ──────────────────────────────────────────────────────────
def _sniff_language(code: str) -> str:
    if re.search(r'^\s*(def |import |from )\w+', code, re.M):
        return "python"
    if re.search(r'#include\s*[<"]|std::|cout\s*<<', code, re.M):
        return "cpp"
    if re.search(r'public\s+class\s+\w+|System\.out\.print', code, re.M):
        return "java"
    if re.search(r'(function\s+\w+|const\s+\w+\s*=|=>)', code, re.M):
        return "javascript"
    if re.search(r'#include\s*<stdio\.h>|printf\s*\(|malloc\s*\(', code, re.M):
        return "c"
    return "python"

# ── Post-processor: wrap bare code in fences ──────────────────────────────────
def _wrap_bare_code(text: str) -> str:
    """
    When the model outputs syntactically correct code WITHOUT backtick fences
    (common at low temperatures), detect the code block boundaries and wrap
    them automatically so Streamlit renders it with syntax highlighting.

    If the response already has ``` fences, this function is a no-op.
    """
    if "```" in text:
        return text   # already fenced — nothing to do

    lines      = text.splitlines()
    first_code = None
    last_code  = None

    for i, line in enumerate(lines):
        is_code = bool(_CODE_LINE_RE.match(line)) or (
            first_code is not None and re.match(r'^\s{4,}\S', line)
        )
        if is_code:
            if first_code is None:
                first_code = i
            last_code = i

    if first_code is None:
        return text   # no code detected — return unchanged

    # Include trailing blank lines that are part of the code block
    while last_code + 1 < len(lines) and lines[last_code + 1].strip() == "":
        last_code += 1

    lang    = _sniff_language("\n".join(lines[first_code : last_code + 1]))
    wrapped = (
        lines[:first_code]
        + [f"```{lang}"]
        + lines[first_code : last_code + 1]
        + ["```"]
        + lines[last_code + 1 :]
    )
    return "\n".join(wrapped)

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Forge — Coding Assistant",
    page_icon="⚡",
    layout="wide",
)

# ── Sidebar: settings + info ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Forge")
    st.markdown("**Gemma 3 27B · Q4_K_M · Local**")
    st.markdown("Python · JavaScript · Java · C++ · C · SQL")
    st.divider()

    st.markdown("### ⚙️ Generation settings")
    temperature = st.slider(
        "Temperature",
        min_value=0.0, max_value=1.0,
        value=DEFAULT_TEMPERATURE, step=0.05,
        help="Lower = more deterministic code. 0.1 recommended for algorithms.",
    )
    max_tokens = st.slider(
        "Max tokens",
        min_value=256, max_value=4096,
        value=1024, step=128,
        help="Maximum length of the response.",
    )
    repeat_penalty = st.slider(
        "Repeat penalty",
        min_value=1.0, max_value=1.5,
        value=DEFAULT_REPEAT_PENALTY, step=0.05,
        help="Penalises repeated tokens. 1.1 is gentle but effective.",
    )

    st.divider()

    # Server health
    try:
        info = requests.get(f"{SERVER_URL}/health", timeout=2).json()
        st.success(f"🟢 Server online\n\nGPU layers: {info.get('gpu_layers', '?')}")
    except Exception:
        st.error(
            "🔴 Server offline\n\n"
            "Run:\n```\npython main.py --model gemma3-forge-Q4_K_M.gguf\n```"
        )

    st.divider()

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("### 💡 Examples")
    examples = [
        "Write a binary search algorithm in Python with full type hints.",
        "Implement a linked list with insert, delete, and reverse in C.",
        "Write a debounce function in JavaScript with JSDoc.",
        "Write a generic Stack class in Java.",
        "Debug: `for i in range(1, len(arr)):` skips the first element.",
        "What is the time complexity of merge sort and why?",
        "Implement Dijkstra's shortest path algorithm in Python.",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=ex):
            st.session_state._inject = ex
            st.rerun()

# ── Main chat area ────────────────────────────────────────────────────────────
st.markdown("## ⚡ Forge — Precision Coding Assistant")
st.caption(
    "Gemma 3 27B fine-tuned on 40K curated coding samples. "
    "Runs 100% locally on your RTX 3060."
)
st.divider()

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle injected example from sidebar
injected = st.session_state.pop("_inject", None)

# ── Prompt handling ───────────────────────────────────────────────────────────
user_input = st.chat_input("Ask Forge to write or debug code…") or injected

if user_input:
    # Show raw user message
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Build the augmented prompt (extra instructions for code tasks)
    augmented = _augment_prompt(user_input)
    # Build payload messages: history (unaugmented) + augmented current message
    api_messages = st.session_state.messages[:-1] + [
        {"role": "user", "content": augmented}
    ]

    with st.chat_message("assistant"):
        placeholder   = st.empty()
        full_response = ""

        try:
            resp = requests.post(
                CHAT_URL,
                json={
                    "messages":       api_messages,
                    "stream":         True,
                    "max_tokens":     max_tokens,
                    "temperature":    temperature,
                    "top_k":          DEFAULT_TOP_K,
                    "top_p":          DEFAULT_TOP_P,
                    "min_p":          DEFAULT_MIN_P,
                    "repeat_penalty": repeat_penalty,
                },
                stream=True,
                timeout=TIMEOUT,
            )

            if resp.status_code != 200:
                st.error(
                    f"⚠️ Server error HTTP {resp.status_code}\n\n"
                    f"`{resp.text[:400]}`"
                )
            else:
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue

                    line = raw_line.decode("utf-8")
                    if not line.startswith("data: "):
                        continue

                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data_str)
                        token = chunk["choices"][0].get("delta", {}).get("content", "")

                        # Sanitize GGUF tokeniser artefacts
                        token = re.sub(r"\[UNK_BYTE_[^\]]+\]", " ", token)

                        full_response += token
                        # Stream raw text while generating — fencing happens after
                        placeholder.markdown(full_response + "▌")
                    except json.JSONDecodeError:
                        pass

                # ── Post-process and render final response ─────────────────
                if full_response.strip():
                    # FIX: wrap any bare code blocks the model forgot to fence.
                    # Old code checked `"```" in full_response` which was False
                    # when the model output correct code without backtick fences,
                    # triggering a false warning on every single response.
                    # Now we fix the formatting instead of complaining about it.
                    display_response = _wrap_bare_code(full_response)

                    placeholder.markdown(display_response)

                    # Save the post-processed version so future turns render correctly
                    st.session_state.messages.append({
                        "role":    "assistant",
                        "content": display_response,
                    })
                else:
                    placeholder.warning(
                        "*(No response — model may have hit a stop token early. "
                        "Try increasing Max tokens in the sidebar.)*"
                    )

        except requests.exceptions.ConnectTimeout:
            st.error(
                "⚠️ Cannot connect to Forge server (timed out after 10 s).  \n"
                "Make sure `python main.py` is running."
            )
        except requests.exceptions.ReadTimeout:
            st.error(
                "⚠️ Server stopped responding mid-stream (> 300 s).  \n"
                "The model may have stalled. Restart `python main.py`."
            )
        except requests.exceptions.RequestException as e:
            st.error(f"⚠️ Network error: {e}")