"""Example: analyse data from PostgreSQL.

Install extra:  pip install parseiq[postgres]
"""
from parseiq import Pipeline

result = Pipeline.from_postgres(
    conn_string="postgresql://user:password@localhost:5432/mydb",
    query="SELECT * FROM orders LIMIT 10000",
    table_name="orders",
    output_dir="output/postgres_run",
).run(llm=False)

print(f"Quality score: {result.overall_quality_score}/100")
print(f"Anomalies    : {result.total_anomalies}")
