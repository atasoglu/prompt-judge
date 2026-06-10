# prompt-judge

Iteratively compress a prompt while preserving semantic fidelity.

Multiple summarizer agents compress the prompt in parallel using different strategies (*extractive*, *abstractive*, *structure-preserving*, *role-preserving*). A judge agent scores each candidate against the original, and the best result becomes the input for the next iteration.

## Installation

```bash
uv sync
```

## Usage

```bash
# Pass prompt as argument
prompt-judge "Your long prompt here..."

# Or pipe from stdin
cat prompt.txt | prompt-judge

# Custom model and base URL (Ollama, vLLM, OpenRouter, …)
prompt-judge "..." --model llama3 --base-url http://localhost:11434/v1

# Save full JSONL log
prompt-judge "..." --output log.jsonl
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `gpt-4o-mini` | Summarizer model |
| `--judge-model` | same as `--model` | Judge model |
| `--embed-model` | — | Embedding model for cosine similarity check |
| `--base-url` | — | OpenAI-compatible endpoint |
| `--api-key` | `$OPENAI_API_KEY` | API key |
| `--iterations` | `10` | Max compression iterations |
| `--threshold` | `0.85` | Min similarity score to continue |
| `--strategies` | all | Summarizer strategies to use |
| `--instructions` | — | Extra instructions for summarizers |
| `--output`, `-o` | — | Write JSONL log to file |
| `--quiet`, `-q` | — | Suppress progress output |

## Requirements

- Python 3.11+
- `openai` SDK
- Any OpenAI-compatible API endpoint
