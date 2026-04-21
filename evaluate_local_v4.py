"""
eval/evaluate_local_v4.py  —  Forge Full Evaluation Suite (Local / RTX 3060)
─────────────────────────────────────────────────────────────────────────────
RTX 3060 12 GB · Q4_K_M GGUF · llama-cpp-python (no HuggingFace transformers)

Difference from evaluate_h100_v4.py:
  • Loads the GGUF file via llama-cpp-python instead of the 109 GB bfloat16
    safetensors via HuggingFace transformers. The H100 version would OOM
    immediately on a 12 GB card.
  • GPU layers = 25 (partial offload — fits in 12 GB). Raise to 35 if you
    have headroom, lower to 15 if you get CUDA out-of-memory errors.
  • Uses the raw Gemma 3 prompt template from main.py (hardcoded special
    tokens) instead of apply_chat_template(). This is the same fix that
    main.py uses to prevent silent variable-name drops from the llama-bpe
    tokenizer hotpatch in the GGUF export.
  • Context window = 4096 (same as main.py). H100 version used 8192+.
  • Inference is slower (~20-40s per problem vs ~9s on H100). HumanEval 164
    problems will take ~2.5 hours. MBPP 100 problems ~1.5 hours. Plan ahead.
  • No torch / transformers / bitsandbytes needed. Just llama-cpp-python.

main.py vs this file:
  main.py          = FastAPI server for interactive use (Open WebUI, VS Code,
                     curl). Run it when you want to CHAT with Forge.
  this file        = Standalone benchmark script. Run it when you want SCORES.
  They are separate. You never need main.py running during eval.

Install:
  pip install llama-cpp-python evalplus datasets tqdm
  # For GPU support (CUDA):
  CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall

Run:
  python eval/evaluate_local_v4.py --model ./gemma3-forge-Q4_K_M.gguf
  python eval/evaluate_local_v4.py --model ./gemma3-forge-Q4_K_M.gguf --gpu-layers 20
  python eval/evaluate_local_v4.py --model ./gemma3-forge-Q4_K_M.gguf --skip-humaneval
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from tqdm import tqdm

# ── CLI args ──────────────────────────────────────────────────────────────────
# Parsed before anything else so --help works without loading the model.
def _parse_args():
    ap = argparse.ArgumentParser(description="Forge local evaluation suite")
    ap.add_argument(
        "--model",
        default="./gemma3-forge-Q4_K_M.gguf",
        help="Path to the Q4_K_M GGUF file (default: ./gemma3-forge-Q4_K_M.gguf)",
    )
    ap.add_argument(
        "--gpu-layers", dest="gpu_layers", type=int, default=25,
        help="Layers to offload to GPU. RTX 3060 12GB → 25. Raise/lower as needed.",
    )
    ap.add_argument(
        "--ctx", type=int, default=4096,
        help="Context window tokens (default: 4096)",
    )
    ap.add_argument(
        "--threads", type=int, default=8,
        help="CPU threads for layers that run on CPU (default: 8)",
    )
    ap.add_argument(
        "--skip-humaneval", action="store_true",
        help="Skip HumanEval (164 problems, ~2.5 hrs). Useful for quick debug runs.",
    )
    ap.add_argument(
        "--skip-humaneval-plus", action="store_true",
        help="Skip HumanEval+ (another ~2.5 hrs on top of HumanEval).",
    )
    ap.add_argument(
        "--skip-mbpp", action="store_true",
        help="Skip MBPP (100 problems, ~1.5 hrs).",
    )
    ap.add_argument(
        "--humaneval-n", type=int, default=164,
        help="Number of HumanEval problems to run (default: 164 = full set).",
    )
    ap.add_argument(
        "--mbpp-n", type=int, default=100,
        help="Number of MBPP problems to run (default: 100).",
    )
    ap.add_argument(
        "--debug-n", type=int, default=50,
        help="Number of DebugBench samples (default: 50).",
    )
    return ap.parse_args()

ARGS = _parse_args()

# ── Config ────────────────────────────────────────────────────────────────────
GGUF_PATH    = ARGS.model
RESULTS_FILE = Path("eval/results_local_v4.json")

# Generation settings — identical to main.py for consistency
TEMPERATURE    = 0.1
TOP_K          = 40
TOP_P          = 0.95
MIN_P          = 0.05    # main.py uses this; H100 version didn't — helps quality
REPEAT_PENALTY = 1.1
MAX_NEW_TOKENS = 768     # for spot checks / debug
CODE_MAX_TOKENS = 512   # for HumanEval / MBPP (shorter problems)

# ── Gemma 3 raw prompt tokens (from main.py) ──────────────────────────────────
# We build prompts manually instead of using chat_format="gemma" because the
# GGUF was exported with a llama-bpe tokenizer hotpatch. Using llama.cpp's
# built-in Gemma chat formatter runs through the broken vocab mapping and
# silently drops variable names (e.g. "distances", "target" become empty string).
# Building the raw string and passing to create_completion() bypasses this.
_BOS           = "<bos>"
_START_OF_TURN = "<start_of_turn>"
_END_OF_TURN   = "<end_of_turn>"

# ── System prompt (identical to main.py) ─────────────────────────────────────
SYSTEM_PROMPT = """You are Forge, an elite precision coding assistant with deep expertise in Python, JavaScript, Java, C++, and C.

## ABSOLUTE CODE RULES — never break these:
1. Every function signature MUST include ALL parameter names.
2. Every comparison MUST have both sides.
3. Every variable used in the body MUST be declared in the signature or defined before use.
4. Code blocks MUST be syntactically complete and immediately executable.
5. Never write TODO, FIXME, or "implement this".
6. All code MUST be inside a properly fenced code block with the language tag.
7. Function and variable names MUST be descriptive and follow language conventions.

## RESPONSE STRUCTURE:
1. One-sentence summary.
2. Complete code in a fenced block.
3. Brief explanation (3-6 bullet points).
4. Edge cases — at least 2.

## WHEN WRITING ALGORITHMS:
- State time and space complexity after the code block.
- Show a worked example.

## WHEN DEBUGGING:
- Root cause in one sentence.
- Corrected code with # FIXED: comments.
- Explain WHY the bug occurred.

## LANGUAGE AND FORMATTING:
- Grammatically correct English. Proper punctuation.
- Complexity: O(n log n). Always state both time AND space."""

# ── Global model handle ───────────────────────────────────────────────────────
_llm = None


# ══════════════════════════════════════════════════════════════════════════════
#  Model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_model():
    global _llm
    from llama_cpp import Llama

    p = Path(GGUF_PATH)
    if not p.exists():
        print(f"  ✗ GGUF not found: {p}")
        print("    Pass --model /path/to/gemma3-forge-Q4_K_M.gguf")
        sys.exit(1)

    size_gb = p.stat().st_size / 1e9
    print(f"  File       : {p.name}  ({size_gb:.1f} GB)")
    print(f"  GPU layers : {ARGS.gpu_layers}  (raise if VRAM allows, lower on OOM)")
    print(f"  Context    : {ARGS.ctx} tokens")
    print(f"  Threads    : {ARGS.threads}")
    print(f"  Loading …")

    # chat_format=None — we build the raw Gemma 3 prompt ourselves.
    # This is the same fix as main.py to avoid silent token drops.
    _llm = Llama(
        model_path=str(p),
        n_ctx=ARGS.ctx,
        n_gpu_layers=ARGS.gpu_layers,
        n_threads=ARGS.threads,
        verbose=False,
        chat_format=None,
    )
    print("  ✓ Model loaded\n")


# ══════════════════════════════════════════════════════════════════════════════
#  Prompt builders  (raw Gemma 3 template — same logic as main.py)
# ══════════════════════════════════════════════════════════════════════════════

def _build_prompt(user_content: str, with_system: bool = True) -> str:
    """
    Build a single-turn raw Gemma 3 prompt string.
    with_system=True  → prepend the full Forge system prompt (DebugBench, Spot checks)
    with_system=False → bare user message only (HumanEval, HumanEval+, MBPP)
    """
    if with_system:
        user_content = f"{SYSTEM_PROMPT}\n\n{user_content}"

    return (
        f"{_BOS}"
        f"{_START_OF_TURN}user\n{user_content}{_END_OF_TURN}\n"
        f"{_START_OF_TURN}model\n"
    )


def _call(prompt: str, max_tokens: int) -> str:
    """Single call to llama-cpp-python create_completion()."""
    result = _llm.create_completion(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        min_p=MIN_P,
        repeat_penalty=REPEAT_PENALTY,
        stop=[_END_OF_TURN],
        echo=False,
        stream=False,
    )
    return result["choices"][0].get("text", "").strip()


def ask(user_prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """Full Forge ask — includes system prompt. For DebugBench + Spot checks."""
    return _call(_build_prompt(user_prompt, with_system=True), max_new_tokens)


def _ask_humaneval(fn_prompt: str) -> str:
    """HumanEval / HumanEval+ — no system prompt, minimal instruction."""
    user_msg = (
        "Complete this Python function. "
        "Return ONLY the completed function inside a ```python``` block. "
        "Do not add any explanation.\n\n"
        f"```python\n{fn_prompt}\n```"
    )
    return _call(_build_prompt(user_msg, with_system=False), CODE_MAX_TOKENS)


def _ask_mbpp(description: str, func_name: str) -> str:
    """
    MBPP — no system prompt, function name injected explicitly.
    MBPP test assertions hardcode a specific name (e.g. assert min_cost(...)).
    Without injecting the name the model picks its own → NameError every time.
    This was the root cause of the 9% score in v3.
    """
    user_msg = (
        f"Write a Python function named `{func_name}` that solves this task:\n\n"
        f"{description}\n\n"
        f"The function MUST be named exactly `{func_name}`.\n"
        f"Return ONLY the function inside a ```python``` block. "
        f"Do not add any explanation or example calls."
    )
    return _call(_build_prompt(user_msg, with_system=False), CODE_MAX_TOKENS)


# ══════════════════════════════════════════════════════════════════════════════
#  Code helpers
# ══════════════════════════════════════════════════════════════════════════════

def extract_python_code(text: str) -> str:
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    idx = text.find("def ")
    return text[idx:].strip() if idx != -1 else text.strip()


def extract_func_name_from_tests(test_list: list) -> str:
    """Pull the expected function name from the first MBPP assert statement."""
    for test in test_list:
        m = re.search(r'assert\s+([a-zA-Z_]\w*)\s*\(', test)
        if m:
            return m.group(1)
    return "solution"


_CODE_LINE_RE = re.compile(
    r'^(\s*)(def |class |for |while |if |elif |else:|return |import |from |'
    r'try:|except|finally:|with |async |await |yield |'
    r'#include|public |private |static |void |int |float |char |bool |'
    r'function |const |let |var )',
    re.MULTILINE,
)

def _sniff_language(code: str) -> str:
    if re.search(r'^\s*(def |import |from )\w+', code, re.M):      return "python"
    if re.search(r'#include\s*[<"]|std::|cout',   code, re.M):     return "cpp"
    if re.search(r'public\s+class\s+\w+',          code, re.M):    return "java"
    if re.search(r'(function\s+\w+|const\s+\w+=|=>)', code, re.M): return "javascript"
    if re.search(r'#include\s*<stdio|printf\s*\(', code, re.M):    return "c"
    return "python"

def _wrap_bare_code(text: str) -> str:
    if "```" in text:
        return text
    lines = text.splitlines()
    first_code = last_code = None
    for i, line in enumerate(lines):
        is_code = bool(_CODE_LINE_RE.match(line)) or (
            first_code is not None and re.match(r'^\s{4,}\S', line)
        )
        if is_code:
            if first_code is None:
                first_code = i
            last_code = i
    if first_code is None:
        return text
    while last_code + 1 < len(lines) and lines[last_code + 1].strip() == "":
        last_code += 1
    lang = _sniff_language("\n".join(lines[first_code:last_code + 1]))
    return "\n".join(
        lines[:first_code] + [f"```{lang}"]
        + lines[first_code:last_code + 1] + ["```"]
        + lines[last_code + 1:]
    )


def _smart_assemble(fn_prompt: str, model_response: str) -> str:
    """
    Assemble final HumanEval code from stub + model response.
    Case A: model returned full function with def → use directly
    Case B: model returned body only              → indent + attach to stub
    Case C: empty / garbled                       → stub + pass
    """
    m = re.search(r"```(?:python)?\n(.*?)```", model_response, re.DOTALL)
    code = m.group(1).strip() if m else model_response.strip()

    if not code:
        return fn_prompt.rstrip() + "\n    pass"

    fn_name_match = re.search(r"def\s+(\w+)\s*\(", fn_prompt)
    if fn_name_match and f"def {fn_name_match.group(1)}" in code:
        return code  # Case A

    if not code.lstrip().startswith("def "):
        indented = "\n".join(
            ("    " + line) if line.strip() else ""
            for line in code.splitlines()
        )
        return fn_prompt.rstrip() + "\n" + indented  # Case B

    return fn_prompt.rstrip() + "\n    pass"  # Case C


def run_python_code(code: str, test: str, timeout: int = 15) -> bool:
    """
    Execute code + test in a subprocess.
    Timeout is 15s here vs 10s on H100 — CPU inference is slower and the
    generated code itself may run on CPU, so give it more time.
    """
    full = code + "\n\n" + test
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(full)
        fname = f.name
    try:
        res = subprocess.run(
            [sys.executable, fname],
            capture_output=True,
            timeout=timeout,
        )
        return res.returncode == 0
    except Exception:
        return False
    finally:
        Path(fname).unlink(missing_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  1. HumanEval — full 164 problems
# ══════════════════════════════════════════════════════════════════════════════

def eval_humaneval(n: int = 164) -> dict:
    print(f"\n[HumanEval] {n} problems  (~{n * 25 // 60} hrs estimated on RTX 3060) ...")
    try:
        from datasets import load_dataset
        try:
            ds = load_dataset("openai/human-eval", split="test").select(range(n))
        except Exception:
            ds = load_dataset("openai_humaneval", split="test").select(range(n))
    except Exception as e:
        print(f"  Could not load: {e}")
        return {}

    passed  = 0
    results = []

    for row in tqdm(ds, desc="  humaneval"):
        fn_prompt = row["prompt"]
        test_code = row["test"]
        task_id   = row["task_id"]

        response  = _ask_humaneval(fn_prompt)
        full_code = _smart_assemble(fn_prompt, response)
        ok        = run_python_code(full_code, test_code)
        if ok:
            passed += 1
        results.append({"task_id": task_id, "passed": ok})

        if len(results) % 10 == 0:
            print(f"    {len(results)}/{n}  pass@1 so far: {passed/len(results):.1%}")

    pass_at_1 = passed / n
    print(f"  pass@1 = {pass_at_1:.2%}  ({passed}/{n})")
    return {"pass@1": pass_at_1, "passed": passed, "total": n, "results": results}


# ══════════════════════════════════════════════════════════════════════════════
#  2. HumanEval+ — 80x more test cases per problem
# ══════════════════════════════════════════════════════════════════════════════

def _load_evalplus():
    """Try all known evalplus import layouts."""
    try:
        from evalplus.data     import get_human_eval_plus
        from evalplus.evaluate import evaluate_functional_correctness
        return get_human_eval_plus, evaluate_functional_correctness
    except ImportError:
        pass
    try:
        from evalplus.data import get_human_eval_plus
        from evalplus.eval import evaluate_functional_correctness
        return get_human_eval_plus, evaluate_functional_correctness
    except ImportError:
        pass
    try:
        from evalplus.data import get_human_eval_plus
        return get_human_eval_plus, None   # CLI fallback
    except ImportError:
        pass
    return None, None


def eval_humaneval_plus(n: int = 164) -> dict:
    print(f"\n[HumanEval+] {n} problems (evalplus — extended test cases) ...")

    get_hep, eval_fc = _load_evalplus()
    if get_hep is None:
        print("  ✗ evalplus not importable.  Run: pip install --upgrade evalplus")
        return {}

    problems     = get_hep()
    problem_list = list(problems.items())[:n]
    samples      = []

    for task_id, problem in tqdm(problem_list, desc="  humaneval+"):
        fn_prompt = problem["prompt"]
        response  = _ask_humaneval(fn_prompt)
        full_code = _smart_assemble(fn_prompt, response)
        samples.append({"task_id": task_id, "solution": full_code})

    import os
    with tempfile.TemporaryDirectory() as tmp:
        samples_path = os.path.join(tmp, "samples.jsonl")
        with open(samples_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

        if eval_fc is not None:
            results = eval_fc(
                sample_file=samples_path,
                k=[1],
                problems=problems,
                base_only=False,
            )
        else:
            cli = subprocess.run(
                [sys.executable, "-m", "evalplus.evaluate",
                 "--dataset", "humaneval", "--samples", samples_path],
                capture_output=True, text=True,
            )
            print(cli.stdout)
            if cli.returncode != 0:
                print(f"  ✗ evalplus CLI error:\n{cli.stderr}")
                return {}
            base_m = re.search(r'pass@1.*?:\s*([\d.]+)', cli.stdout)
            plus_m = re.search(r'plus.*?pass@1.*?:\s*([\d.]+)', cli.stdout, re.I)
            b = float(base_m.group(1)) if base_m else 0.0
            p = float(plus_m.group(1)) if plus_m else 0.0
            print(f"  base pass@1 : {b:.2%}   plus pass@1 : {p:.2%}   gap : {b-p:.2%}")
            return {"base_pass@1": round(b, 4), "plus_pass@1": round(p, 4), "gap": round(b-p, 4), "n": n}

    base = results.get("pass@1",      0.0)
    plus = results.get("plus_pass@1", 0.0)
    gap  = base - plus
    print(f"  base pass@1 : {base:.2%}   plus pass@1 : {plus:.2%}   gap : {gap:.2%}  ", end="")
    if   gap <= 0.05: print("(excellent)")
    elif gap <= 0.12: print("(normal)")
    else:             print("(high — check harder cases)")
    return {"base_pass@1": round(base, 4), "plus_pass@1": round(plus, 4), "gap": round(gap, 4), "n": n}



# ══════════════════════════════════════════════════════════════════════════════
#  3. DebugBench
# ══════════════════════════════════════════════════════════════════════════════

def _get_buggy_fixed(row: dict) -> tuple:
    buggy_keys = ["buggy_code", "bug_code", "code", "input"]
    fixed_keys = ["solution", "fixed_code", "correct_code", "target", "output"]
    buggy = next((row[k] for k in buggy_keys if k in row and row[k]), "")
    fixed = next((row[k] for k in fixed_keys if k in row and row[k]), "")
    return buggy, fixed

ALLOWED_LANGS = {"python", "c", "cpp", "c++", "java", "javascript", "js"}

def eval_debug(n: int = 50) -> dict:
    print(f"\n[DebugBench] {n} samples ...")
    try:
        from datasets import load_dataset
        ds = load_dataset("Rtian/DebugBench", split="test").select(range(n))
    except Exception as e:
        print(f"  Could not load: {e}")
        return {}

    if len(ds) > 0:
        print(f"  Fields : {list(ds[0].keys())}")

    correct = processed = skipped = 0
    results = []

    for row in tqdm(ds, desc="  debugbench"):
        lang = (row.get("language") or row.get("lang") or "").lower()
        if lang not in ALLOWED_LANGS:
            skipped += 1
            continue

        buggy, fixed = _get_buggy_fixed(row)
        if not buggy or not fixed:
            skipped += 1
            continue

        bug_type  = (row.get("bug_type") or row.get("error_type") or "bug").lower()
        processed += 1

        response     = ask(
            f"Find the bug in this {lang} code and show the fixed version:\n\n"
            f"```{lang}\n{buggy}\n```",
            max_new_tokens=512,
        )
        fixed_tokens = set(fixed.lower().split()[:30])
        resp_tokens  = set(response.lower().split())
        overlap      = len(fixed_tokens & resp_tokens) / max(len(fixed_tokens), 1)
        ok           = overlap > 0.5
        if ok:
            correct += 1
        results.append({"bug_type": bug_type, "overlap": round(overlap, 3), "passed": ok})

    accuracy = correct / processed if processed > 0 else 0.0
    print(f"  accuracy = {accuracy:.2%}  ({correct}/{processed} correct, {skipped} skipped)")
    return {"accuracy": accuracy, "correct": correct, "processed": processed, "skipped": skipped}


# ══════════════════════════════════════════════════════════════════════════════
#  4. Spot checks
# ══════════════════════════════════════════════════════════════════════════════

SPOT_CHECKS = [
    {
        "name":            "list_flattening",
        "prompt":          "Write a Python function that flattens a nested list of arbitrary depth.",
        "expect_keywords": ["def", "flatten", "isinstance", "list", "yield"],
    },
    {
        "name":   "sql_injection",
        "prompt": (
            "What is the security bug in this code and how do you fix it?\n\n"
            "```python\n"
            "user_input = input('Enter name: ')\n"
            "query = 'SELECT * FROM users WHERE name = ' + user_input\n"
            "cursor.execute(query)\n"
            "```"
        ),
        "expect_keywords": ["sql injection", "parameterized", "placeholder"],
    },
    {
        "name":   "off_by_one",
        "prompt": (
            "This loop skips the first element. Debug and fix it:\n\n"
            "```python\n"
            "arr = [10, 20, 30]\n"
            "for i in range(1, len(arr)):\n"
            "    print(arr[i])\n"
            "```"
        ),
        "expect_keywords": ["range(0", "range(len", "first", "index"],
    },
    {
        "name":   "async_await",
        "prompt": (
            "Write an async Python function that fetches JSON from 3 URLs "
            "concurrently using aiohttp and asyncio.gather."
        ),
        "expect_keywords": ["async def", "await", "asyncio.gather", "aiohttp"],
    },
    {
        "name":   "time_complexity",
        "prompt": (
            "This function is O(n²). Rewrite it to be O(n):\n\n"
            "```python\n"
            "def has_duplicate(arr):\n"
            "    for i in range(len(arr)):\n"
            "        for j in range(i+1, len(arr)):\n"
            "            if arr[i] == arr[j]:\n"
            "                return True\n"
            "    return False\n"
            "```"
        ),
        "expect_keywords": ["set(", "O(n)", "seen"],
    },
]

def eval_spot_checks() -> dict:
    print(f"\n[Spot Checks] {len(SPOT_CHECKS)} prompts ...")
    results = []
    for check in SPOT_CHECKS:
        response  = ask(check["prompt"], max_new_tokens=MAX_NEW_TOKENS)
        full_text = response.lower()
        keywords  = check["expect_keywords"]
        hits      = sum(1 for kw in keywords if kw.lower() in full_text)
        total_k   = len(keywords)
        score     = hits / total_k
        matched   = [kw for kw in keywords if kw.lower() in full_text]
        results.append({
            "name":             check["name"],
            "score":            round(score, 3),
            "hits":             hits,
            "total":            total_k,
            "matched":          matched,
            "response_preview": _wrap_bare_code(response)[:400],
        })
        sym = "✓" if score >= 0.6 else "~" if score >= 0.3 else "✗"
        print(f"  {sym} {check['name']:25s}  {score:.0%}  ({hits}/{total_k})  matched={matched}")

    mean = sum(r["score"] for r in results) / len(results)
    print(f"  Mean: {mean:.1%}")
    return {"mean_score": round(mean, 3), "checks": results}


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 62)
    print("  Forge — Full Evaluation Suite  (Local · GGUF · v4)")
    print("=" * 62)
    print(f"  Model      : {GGUF_PATH}")
    print(f"  GPU layers : {ARGS.gpu_layers}  |  Context : {ARGS.ctx}  |  Threads : {ARGS.threads}")
    print(f"  Temp={TEMPERATURE}  top_k={TOP_K}  top_p={TOP_P}  min_p={MIN_P}  rep_pen={REPEAT_PENALTY}")

    skipped = []
    if ARGS.skip_humaneval:       skipped.append("HumanEval")
    if ARGS.skip_humaneval_plus:  skipped.append("HumanEval+")
    if ARGS.skip_mbpp:            skipped.append("MBPP")
    if skipped:
        print(f"  Skipping   : {', '.join(skipped)}")

    # Estimate total time
    total_mins = 0
    if not ARGS.skip_humaneval:       total_mins += ARGS.humaneval_n * 25 // 60
    if not ARGS.skip_humaneval_plus:  total_mins += ARGS.humaneval_n * 25 // 60
    if not ARGS.skip_mbpp:            total_mins += ARGS.mbpp_n * 15 // 60
    total_mins += 30  # DebugBench + spots
    print(f"  Est. time  : ~{total_mins} min  (varies by GPU layer count + problem difficulty)")
    print()

    # [1/5] Load
    print("[1/5] Loading model ...")
    load_model()

    # [2/5] Smoke test
    print("[2/5] Smoke test ...")
    smoke = ask("Reply with only the word: ready", max_new_tokens=10)
    print(f"  Model says: {repr(smoke[:80])}")
    if not smoke.strip():
        print("  ✗ Empty response. Check --model path and --gpu-layers.")
        sys.exit(1)
    print("  ✓ Ready\n")

    t0 = time.time()

    # [3/5] HumanEval
    if not ARGS.skip_humaneval:
        print("[3/5] Running HumanEval ...")
        he = eval_humaneval(n=ARGS.humaneval_n)
    else:
        print("[3/5] HumanEval — SKIPPED")
        he = {}

    # [4/5] HumanEval+
    if not ARGS.skip_humaneval_plus:
        print("[4/5] Running HumanEval+ ...")
        hep = eval_humaneval_plus(n=ARGS.humaneval_n)
    else:
        print("[4/5] HumanEval+ — SKIPPED")
        hep = {}

    # [5/5] DebugBench + Spot checks
    print("[5/5] Running DebugBench + Spot checks ...")
    db = eval_debug(n=ARGS.debug_n)
    sp = eval_spot_checks()

    elapsed = time.time() - t0

    # Collect scores
    he_score = he.get("pass@1",       0.0)
    hep_base = hep.get("base_pass@1", he_score)
    hep_plus = hep.get("plus_pass@1", 0.0)
    hep_gap  = hep.get("gap",         0.0)
    mb_score = mb.get("pass@1",       0.0)
    db_score = db.get("accuracy",     0.0)
    sp_score = sp.get("mean_score",   0.0)

    summary = {
        "model":                   GGUF_PATH,
        "precision":               "Q4_K_M",
        "inference_mode":          "llama_cpp_local",
        "gpu_layers":              ARGS.gpu_layers,
        "temperature":             TEMPERATURE,
        "humaneval_pass@1":        he_score,
        "humaneval_plus_base@1":   hep_base,
        "humaneval_plus_plus@1":   hep_plus,
        "humaneval_he_vs_hep_gap": hep_gap,
        "debug_accuracy":          db_score,
        "spot_mean_coverage":      sp_score,
        "eval_time_minutes":       round(elapsed / 60, 1),
        "full": {"humaneval": he, "humaneval_plus": hep, "debug": db, "spot": sp},
    }

    RESULTS_FILE.parent.mkdir(exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 62}")
    print(f"  RESULTS  (Q4_K_M · {ARGS.gpu_layers} GPU layers)")
    print(f"{'=' * 62}")
    if he:
        print(f"  HumanEval     pass@1  : {he_score:.2%}  ({he.get('passed',0)}/{he.get('total',0)})")
    if hep:
        print(f"  HumanEval+    base @1 : {hep_base:.2%}")
        print(f"  HumanEval+    plus @1 : {hep_plus:.2%}  (gap: {hep_gap:.2%})")
        
    print(f"  DebugBench    accuracy: {db_score:.2%}  ({db.get('correct',0)}/{db.get('processed',0)})")
    print(f"  Spot checks   mean    : {sp_score:.1%}")
    print(f"  Total time            : {summary['eval_time_minutes']} min")
    print(f"\n  Saved → {RESULTS_FILE}")

    print(f"\n{'─' * 62}")
    print("  INTERPRETATION")
    print(f"{'─' * 62}")
    if hep and hep_gap:
        if   hep_gap <= 0.05: print("  HE vs HE+ gap ≤5%   → Solves correctly, not pattern-matching.")
        elif hep_gap <= 0.12: print("  HE vs HE+ gap 5-12% → Normal. Most edge cases handled.")
        else:                 print("  HE vs HE+ gap >12%  → Check harder test cases.")
    note = "  NOTE: Q4_K_M scores may be 1-3% lower than bfloat16 (H100) scores"
    print(note)
    print("        due to quantization. This is expected and normal.")
    print()


if __name__ == "__main__":
    main()
