# ⚡ Forge — Local Coding Assistant (Gemma 3 27B QLoRA)

> A full end-to-end LLM fine-tuning pipeline: dataset curation → QLoRA training → GGUF export → FastAPI server → Streamlit UI → custom eval harness.  
> Runs a 27B parameter model on a single RTX 3060 12 GB.

---

## What is this

Forge is a QLoRA fine-tune of Google's **Gemma 3 27B-IT** model, conditioned for structured code generation across Python, JavaScript, Java, C++, C and SQL. The model produces complete, runnable code with consistent formatting, complexity analysis, and debugging explanations.

This repository contains the full pipeline — not just the model weights — including the data preparation notebook, training code, export utilities, inference server, Streamlit UI, and evaluation harness.

---
This is the video demo : 

(NOTE: The generation phase in the attached video is speed up 8x. Real-time inference on the 3060 took ~4 minutes.)

https://github.com/user-attachments/assets/b1b9dd71-20f5-4fee-8a74-1deaeeaa4d87





## Results

| Benchmark | Forge (this model) | Gemma 3 27B-IT base | Notes |
|---|---|---|---|
| HumanEval pass@1 | **98.78%** (162/164) | ~84% | Full 164-problem set |
| MBPP pass@1 | **71%** (~73% corrected) | ~72% | 100-problem sanitized split |
| DebugBench accuracy | **74%** (37/50) | — | Token-overlap metric, directional only |

For a complete architectural breakdown of the QLoRA training loop and how the GGUF tokenizer corruption was bypassed, read the engineering post-mortem here: https://medium.com/@kaustubh09k/i-fine-tuned-a-27-billion-parameter-model-as-a-fresher-heres-everything-that-broke-1db882563e4a

Download the model from Hugging Face:
[Link to HF repo](https://huggingface.co/KK9922/Forge-Gemma-3-27B-GGUF)

The HumanEval gain (+15pp over base) is real but partially reflects training distribution overlap with CodeAlpaca and self-oss-instruct. **MBPP is the honest generalization number** — it matches the base model, confirming no catastrophic forgetting.

---

## Repository structure

```
forge/
├── SLM_3.ipynb                  # Full pipeline notebook (data → train → export)
│   ├── Cell 1: prepare_dataset  # Downloads, filters, deduplicates 3 datasets → JSONL
│   ├── Cell 2: train            # QLoRA fine-tune on Gemma 3 27B-IT
│   └── Cell 3: merge_and_export # Merges LoRA → bfloat16, exports to GGUF
│
├── serve/
│   └── main.py                  # FastAPI inference server (OpenAI-compatible)
│
├── app.py                       # Streamlit web UI (connects to main.py)
│
└── eval/
    ├── evaluate_local_v4.py     # Full eval suite (local, GGUF-based)
    └── mbpp_eval.py             # MBPP-only benchmark script
```

---

## Pipeline overview

### 1. Dataset preparation (`SLM_3.ipynb` — Cell 1)

Three sources are downloaded, filtered, and merged:

| Source | Samples kept | Type |
|---|---|---|
| `bigcode/self-oss-instruct-sc2-exec-filter-50k` | 20,000 | Execution-verified code generation |
| `sahil2801/CodeAlpaca-20k` | 13,141 | Instruction-style coding tasks |
| `Rtian/DebugBench` | 0* | Multi-language bug fixing |

> *DebugBench contributed 0 samples due to a field-name mismatch (`solution` vs `fixed_code`) caught during evaluation. Training ran on ~33K samples from the first two sources.

**Filtering applied:**
- Minimum 80-character responses (removes one-liner garbage answers)
- Maximum 6,000-character sequences (removes unusable long samples)
- Pattern removal: `TODO`, `NotImplementedError`, trailing ellipsis, placeholder patterns
- Jaccard deduplication at 0.85 threshold (character 4-shingles, sliding window of 2,000)

Output format: ChatML JSONL with a fixed Forge system prompt at every turn.

### 2. Training (`SLM_3.ipynb` — Cell 2)

```python
BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
)
```

| Parameter | Value |
|---|---|
| Base model | `google/gemma-3-27b-it` |
| Method | QLoRA (NF4 4-bit base, bfloat16 compute) |
| LoRA rank / alpha | r=16 / α=32 |
| Training steps | 1,000 (< 1 epoch over 31K samples) |
| Effective batch size | 16 (per_device=2 × grad_accum=8) |
| Learning rate | 2e-4 · cosine decay · 3% warmup |
| Optimizer | paged_adamw_8bit |
| Sequence length | 2,048 tokens |
| Hardware | 1× H100 80 GB SXM |
| Training time | ~3 h 48 min |

Training loss dropped from 1.83 → 0.22 in the first 50 steps and plateaued for the remaining 950 steps — the model converged early. Running 3–5 epochs would likely squeeze more signal from the data.

### 3. Export (`SLM_3.ipynb` — Cell 3)

The notebook handles three post-training steps:

1. **LoRA merge** — merges adapter weights into the base model → 109 GB bfloat16 safetensors
2. **Tokenizer restore** — replaces the LoRA-modified tokenizer files with the original from `google/gemma-3-27b-it` to fix a vocabulary hash mismatch
3. **GGUF quantization** — converts to Q4_K_M GGUF (~16 GB) via `llama.cpp convert_hf_to_gguf.py`

> **Note on the tokenizer bug:** The GGUF was exported with a llama-bpe hotpatch to handle Gemma 3's SentencePiece vocabulary. This causes llama.cpp's built-in Gemma chat formatter to silently drop variable names (e.g. `distances`, `target`, `neighbor` become empty strings). The inference server and eval scripts both bypass this by building raw prompt strings instead of using the chat formatter.

### 4. Inference server (`serve/main.py`)

FastAPI server that wraps llama-cpp-python and exposes an OpenAI-compatible `/v1/chat/completions` endpoint.

**Key design decisions:**
- `chat_format=None` — disables llama.cpp's broken Gemma tokenizer path
- Raw Gemma 3 special tokens (`<bos>`, `<start_of_turn>user`, `<end_of_turn>`) are hardcoded and injected manually
- System prompt is merged into the first user turn (Gemma 3 has no dedicated system role)
- Corruption detector flags known tokenizer artifact patterns in responses
- Supports both streaming (SSE) and non-streaming responses

```bash
# Start the server
python serve/main.py --model ./gemma3-forge-Q4_K_M.gguf --gpu-layers 25

# Flags
--gpu-layers 25    # layers offloaded to GPU (RTX 3060 12 GB → 25, increase if VRAM allows)
--ctx 4096         # context window
--port 8080        # default port
--host 127.0.0.1   # bind address
```

**Sampling defaults (tuned for code):**

```
temperature=0.1    highly deterministic — correct variable names every time
top_k=40           caps candidate pool — blocks garbage tokens
top_p=0.95         nucleus sampling
min_p=0.05         rejects tokens below 5% of top token probability
                   this is what fixes the "def func(arr,):" class of bugs
repeat_penalty=1.1 gentle — prevents token loops without distorting code
```

### 5. Streamlit UI (`app.py`)

Web interface that connects to the running `main.py` server.

```bash
# With main.py already running:
streamlit run app.py
```

Features:
- Live streaming token output
- Sidebar sliders for temperature, max tokens, repeat penalty
- Server health indicator (green/red based on `/health` endpoint)
- One-click example prompts
- Automatic code fence wrapping when the model outputs bare code without backticks
- `[UNK_BYTE_*]` artifact sanitization
- Multi-turn conversation with full history

### 6. Evaluation

Two eval scripts are provided.

**`eval/evaluate_local_v4.py`** — full suite, runs locally via GGUF:
```bash
python eval/evaluate_local_v4.py --model ./gemma3-forge-Q4_K_M.gguf

# Skip slow benchmarks for a quick sanity check (~45 min)
python eval/evaluate_local_v4.py --model ./gemma3-forge-Q4_K_M.gguf \
  --skip-humaneval --skip-humaneval-plus --skip-mbpp
```

**`eval/mbpp_eval.py`** — MBPP only, faster iteration:
```bash
python eval/mbpp_eval.py --model ./gemma3-forge-Q4_K_M.gguf --n 100
```

Both scripts use the same raw Gemma 3 prompt format as `main.py` — no system prompt for code completion tasks, injected function names for MBPP.

> **MBPP function-name fix:** MBPP test assertions hardcode the expected function name (e.g. `assert min_cost(...) == 4`). Without telling the model what name to use, it picks its own and every test fails with `NameError`. The eval scripts extract the expected name from the first `assert` statement and inject it into the prompt. Initial score before this fix: 9%. After fix: 71%.

---

## Installation

### Requirements

```bash
pip install llama-cpp-python    # inference engine
pip install fastapi uvicorn     # API server
pip install streamlit requests  # web UI
pip install datasets tqdm       # eval scripts
pip install evalplus            # HumanEval+ (optional)
```

For GPU support in llama-cpp-python (CUDA):
```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall
```

For training (only needed if reproducing from the notebook):
```bash
pip install transformers trl peft bitsandbytes accelerate torch sentencepiece
```

### VRAM requirements

| Mode | VRAM |
|---|---|
| `--gpu-layers 25` (RTX 3060 12 GB) | ~10–11 GB |
| `--gpu-layers 40` (RTX 3080/3090 24 GB) | ~16 GB |
| Full GPU offload | ~16 GB |
| CPU-only | ~32 GB RAM |

---

## Quick start

```bash
# 1. Start the inference server
python serve/main.py --model ./gemma3-forge-Q4_K_M.gguf --gpu-layers 25

# 2. Open the web UI
streamlit run app.py
# → http://localhost:8501
```

Or use the API directly:
```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Write a binary search in Python."}],
    "max_tokens": 1024,
    "stream": false
  }'
```

---

## What I learned building this

Four things broke in ways that were not obvious:

**1. Eval bugs are as dangerous as model bugs.**  
HumanEval returned 0% on the first run. The eval code was prepending the function stub to the model's response even when the model had already returned the complete function — creating duplicate `def` blocks that caused `IndentationError`. Fixed with a 3-case assembly function that detects whether the model returned a full definition, a body only, or nothing.

**2. Always verify your data pipeline before training.**  
DebugBench was listed as a training source but contributed 0 samples. The loading code looked for `row["fixed_code"]` but the actual dataset field is `row["solution"]`. The pipeline silently succeeded with 0 samples. I only caught this when evaluating DebugBench and noticed the model had never seen a single bug-fix example.

**3. Benchmark numbers mean nothing without understanding the metric.**  
MBPP returned 9% — which looked like a catastrophic failure. The real cause: MBPP tests hardcode function names in their assertions. The model wrote logically correct functions under different names and failed every test with `NameError`. 
Zero model failure; 100% eval script failure. After injecting the expected name into the prompt got the result of 73%.

**4. Loss plateau ≠ done training.**  
Training loss hit 0.22 at step 50 and barely moved for the next 950 steps. The model hadn't learned everything — it had converged on what the data could teach in that first pass. Running more epochs (3–5) over the same data would likely continue improving loss gradually rather than in one sharp drop.

---

## License

Model weights are derived from `google/gemma-3-27b-it` and are subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).

Code in this repository is released under the MIT License.

---

## Citation

If you use this work, please cite the base model:

```bibtex
@misc{gemmateam2024gemma3,
  title  = {Gemma 3 Technical Report},
  author = {Gemma Team, Google DeepMind},
  year   = {2024},
  url    = {https://ai.google.dev/gemma}
}
```
