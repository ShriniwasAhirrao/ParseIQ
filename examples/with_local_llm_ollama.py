"""Example: LLM enrichment using a local Ollama instance.

Prerequisites:
  1. Install Ollama: https://ollama.com
  2. Pull a model:   ollama pull llama3
  3. Start server:   ollama serve   (runs on http://localhost:11434)

No API key needed — data never leaves your machine.
"""
from parseiq import Pipeline

result = Pipeline("input/input_data.json").run(
    llm=True,
    llm_provider="ollama",
    llm_model="llama3",
    # llm_base_url defaults to http://localhost:11434/v1
)

print(f"LLM grade  : {result.llm_grade}")
print(f"Quality    : {result.overall_quality_score}/100")
print(f"Output dir : {result.output_files[0] if result.output_files else 'n/a'}")
