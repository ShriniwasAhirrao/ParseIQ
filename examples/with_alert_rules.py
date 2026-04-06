"""Example: fire alerts when data quality rules are violated."""
from parseiq import Pipeline
from parseiq.alerts import slack_webhook

# Simple print callback (swap for slack_webhook / email in production)
def print_alert(rule_key, table, column_or_metric, actual_value):
    print(f"  ALERT  {rule_key}  |  {table}.{column_or_metric}  =  {actual_value}")

result = Pipeline("input/input_data.json").run(
    llm=False,
    alert_rules={
        # Fires if employees.salary has any negative values
        "employees.salary": {"negative_values": True},
        # Fires if email null rate exceeds 5 %
        "employees.email":  {"null_rate_gt": 5},
        # Fires if overall table quality drops below 70
        "employees":        {"quality_score_lt": 70},
    },
    on_alert=print_alert,
)

print(f"\nAlerts fired : {len(result.alerts_fired)}")
for a in result.alerts_fired:
    print(f"  {a}")
