"""Example: analyse a local JSON file (no LLM)."""
from parseiq import Pipeline

result = Pipeline("input/input_data.json").run(llm=False)

print(f"Tables     : {result.tables}")
print(f"Avg quality: {result.overall_quality_score}/100")
print(f"Anomalies  : {result.total_anomalies}")
print(f"Files      : {result.output_files}")
