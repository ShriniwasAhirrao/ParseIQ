"""Example: analyse a file stored in Amazon S3.

Install extra:  pip install parseiq[s3]
Credentials are picked up from the boto3 credential chain
(env vars, ~/.aws/credentials, IAM role).
"""
from parseiq import Pipeline

result = Pipeline.from_s3(
    "s3://my-company-data/exports/customers_2024.json",
    output_dir="output/s3_run",
).run(llm=False)

print(f"Tables : {result.tables}")
print(f"Quality: {result.overall_quality_score}/100")
