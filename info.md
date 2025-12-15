# The Architecture is:

Query → Router → Intent Classification
                      ↓
         ┌───────────┴───────────┐
         ↓                       ↓
    INTENT_COURSE           INTENT_GENERAL
         ↓                       ↓
    Engine B (Waterfall)    Engine A (3-Source RAG)
    ├── Tier 1: Exact Code  ├── BM25
    ├── Tier 2: Fuzzy Name  ├── Global Vector
    ├── Tier 3: Instructor  └── Scoped Vector + Rerank
    └── Tier 4: Semantic+BM25
         ↓                       ↓
    Course-Specific Prompt   General Prompt
         ↓                       ↓
         └───────────┬───────────┘
                     ↓
               LLM Response



command - 

cd ~/llama.cpp/build/bin

./llama-server \
  -m ~/models/Qwen3-14B-Q4_K_M.gguf \
  --host 0.0.0.0 \
  --port 3000 \
  --ctx-size 32768 \
  --n-gpu-layers 999 \
  --threads 20 \
  --rope-scaling linear
