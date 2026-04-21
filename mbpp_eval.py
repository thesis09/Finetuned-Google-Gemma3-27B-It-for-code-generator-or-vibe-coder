"""
eval/mbpp_eval.py  —  MBPP only, local RTX 3060
─────────────────────────────────────────────────
Run:
  python mbpp_eval.py --model ./gemma3-forge-Q4_K_M.gguf
  python mbpp_eval.py --model ./gemma3-forge-Q4_K_M.gguf --n 50
  python mbpp_eval.py --model ./gemma3-forge-Q4_K_M.gguf --gpu-layers 20
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

# ── Args ──────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--model",      default="./gemma3-forge-Q4_K_M.gguf")
ap.add_argument("--gpu-layers", dest="gpu_layers", type=int, default=25)
ap.add_argument("--ctx",        type=int, default=4096)
ap.add_argument("--threads",    type=int, default=8)
ap.add_argument("--n",          type=int, default=100,
                help="Number of MBPP problems (max 257 in sanitized test split)")
ARGS = ap.parse_args()

# ── Gemma 3 raw prompt tokens ─────────────────────────────────────────────────
_BOS           = "<bos>"
_START_OF_TURN = "<start_of_turn>"
_END_OF_TURN   = "<end_of_turn>"

# ── Model ─────────────────────────────────────────────────────────────────────
_llm = None

def load_model():
    global _llm
    from llama_cpp import Llama
    p = Path(ARGS.model)
    if not p.exists():
        print(f"✗ GGUF not found: {p}")
        sys.exit(1)
    print(f"  Loading {p.name}  ({p.stat().st_size/1e9:.1f} GB) …")
    _llm = Llama(
        model_path=str(p),
        n_ctx=ARGS.ctx,
        n_gpu_layers=ARGS.gpu_layers,
        n_threads=ARGS.threads,
        verbose=False,
        chat_format=None,   # raw prompt — avoids broken llama-bpe token mapping
    )
    print("  ✓ Loaded\n")

# ── Inference ─────────────────────────────────────────────────────────────────
def _ask(description: str, func_name: str) -> str:
    """
    Ask the model for a function with a specific name.
    Injecting the name is essential — MBPP tests hardcode it in the asserts.
    Without this the model picks its own name → NameError → 9% pass rate.
    """
    user_msg = (
        f"Write a Python function named `{func_name}` that solves this task:\n\n"
        f"{description}\n\n"
        f"The function MUST be named exactly `{func_name}`.\n"
        f"The function MUST only take the arguments that are passed in the test, "
        f"with NO extra parameters beyond what the problem describes.\n"  # ← add this
        f"Return ONLY the function inside a ```python``` block. "
        f"No explanation, no example calls."
    )
    prompt = (
        f"{_BOS}"
        f"{_START_OF_TURN}user\n{user_msg}{_END_OF_TURN}\n"
        f"{_START_OF_TURN}model\n"
    )
    result = _llm.create_completion(
        prompt=prompt,
        max_tokens=512,
        temperature=0.1,
        top_k=40,
        top_p=0.95,
        min_p=0.05,
        repeat_penalty=1.1,
        stop=[_END_OF_TURN],
        echo=False,
        stream=False,
    )
    return result["choices"][0].get("text", "").strip()

# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    idx = text.find("def ")
    return text[idx:].strip() if idx != -1 else text.strip()

def extract_func_name(test_list: list) -> str:
    for test in test_list:
        m = re.search(r'assert\s+([a-zA-Z_]\w*)\s*\(', test)
        if m:
            return m.group(1)
    return "solution"

def run_code(code: str, test: str, timeout: int = 15) -> bool:
    full = code + "\n\n" + test
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(full)
        fname = f.name
    try:
        res = subprocess.run([sys.executable, fname],
                             capture_output=True, timeout=timeout)
        return res.returncode == 0
    except Exception:
        return False
    finally:
        Path(fname).unlink(missing_ok=True)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 52)
    print("  Forge — MBPP Evaluation  (Local · GGUF)")
    print("=" * 52)
    print(f"  Model      : {ARGS.model}")
    print(f"  GPU layers : {ARGS.gpu_layers}  |  n = {ARGS.n} problems")
    print(f"  Est. time  : ~{ARGS.n * 15 // 60} hrs on RTX 3060")
    print()

    load_model()

    from datasets import load_dataset
    try:
        ds = load_dataset(
            "google-research-datasets/mbpp", "sanitized", split="test"
        ).select(range(ARGS.n))
    except Exception:
        ds = load_dataset("mbpp", split="test").select(range(ARGS.n))

    passed  = 0
    failed  = []
    results = []
    t0      = time.time()

    for row in tqdm(ds, desc="  mbpp"):
        description  = row.get("text", row.get("prompt", ""))
        test_list    = row.get("test_list", row.get("tests", []))
        test_imports = row.get("test_imports", [])
        task_id      = row.get("task_id", len(results))

        if not description or not test_list:
            results.append({"task_id": task_id, "passed": False, "skipped": True})
            continue

        func_name     = extract_func_name(test_list)
        imports_block = "\n".join(test_imports) + "\n" if test_imports else ""
        test_code     = imports_block + "\n".join(test_list)

        response = _ask(description, func_name)
        code     = extract_code(response)
        ok       = run_code(code, test_code)

        if ok:
            passed += 1
        else:
            failed.append({"task_id": task_id, "func_name": func_name,
                           "description": description[:80]})

        results.append({"task_id": task_id, "func_name": func_name, "passed": ok})

        if len(results) % 10 == 0:
            pct = passed / len(results)
            elapsed_m = (time.time() - t0) / 60
            remaining = (ARGS.n - len(results)) * (elapsed_m / len(results))
            print(f"    {len(results):>3}/{ARGS.n}  pass@1: {pct:.1%}  "
                  f"elapsed: {elapsed_m:.0f}m  remaining: ~{remaining:.0f}m")

    elapsed   = time.time() - t0
    pass_at_1 = passed / ARGS.n

    # Save results
    Path("eval").mkdir(exist_ok=True)
    out = {
        "model":       ARGS.model,
        "precision":   "Q4_K_M",
        "n":           ARGS.n,
        "passed":      passed,
        "pass@1":      round(pass_at_1, 4),
        "time_minutes": round(elapsed / 60, 1),
        "failed":      failed,
        "results":     results,
    }
    out_path = Path("eval/mbpp_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    # Print summary
    print(f"\n{'=' * 52}")
    print(f"  MBPP RESULTS")
    print(f"{'=' * 52}")
    print(f"  pass@1     : {pass_at_1:.2%}  ({passed}/{ARGS.n})")
    print(f"  failed     : {len(failed)}")
    print(f"  time       : {elapsed/60:.1f} min")
    print(f"  Saved      → {out_path}")

    # Show failed task IDs so you can inspect them
    if failed:
        print(f"\n  Failed task IDs:")
        for f_ in failed:
            print(f"    [{f_['task_id']}]  {f_['func_name']}()  —  {f_['description']}")

    print()


if __name__ == "__main__":
    main()
