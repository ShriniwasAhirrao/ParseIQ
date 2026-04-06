"""
parseiq.alerts — Rule-based alerting for data quality events.

Usage::

    from parseiq import Pipeline
    from parseiq.alerts import slack_webhook

    result = Pipeline("data.json").run(
        llm=False,
        alert_rules={
            "employees.email":  {"null_rate_gt": 5},
            "orders.amount":    {"negative_values": True},
            "customers":        {"quality_score_lt": 70},
        },
        on_alert=slack_webhook("https://hooks.slack.com/services/..."),
    )

Supported rule types
--------------------
    null_rate_gt        float  — fires when null % > threshold (0–100)
    quality_score_lt    float  — fires when table quality score < threshold
    negative_values     bool   — NEGATIVE_VALUES_DETECTED anomaly present
    duplicate_rows      bool   — DUPLICATE_ROWS_DETECTED anomaly present
    future_dates        bool   — FUTURE_DATE_DETECTED anomaly present
    mixed_types         bool   — MIXED_DATA_TYPES anomaly present
    outliers_detected   bool   — NUMERIC_OUTLIERS_DETECTED anomaly present
"""
from __future__ import annotations

import json
import logging
import smtplib
from email.message import EmailMessage
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

AlertCallback = Callable[[str, str, str, Any], None]


def evaluate_rules(
    alert_rules: Dict[str, Dict[str, Any]],
    raw_metadata: Dict[str, Any],
    on_alert: Optional[AlertCallback] = None,
) -> List[Dict[str, Any]]:
    """Evaluate *alert_rules* against Step-1 *raw_metadata*.

    Parameters
    ----------
    alert_rules:
        ``{"table"`` or ``"table.column": {rule_type: threshold}}``
    raw_metadata:
        The Step-1 technical metadata dict (raw_metadata.json structure).
    on_alert:
        Optional callback fired for each alert.
        Signature: ``(rule_key, table, column_or_metric, actual_value) → None``
    """
    fired: List[Dict[str, Any]] = []
    tables_meta = raw_metadata.get("tables", {})

    for rule_key, conditions in alert_rules.items():
        parts = rule_key.split(".", 1)
        table_name = parts[0]
        column_name = parts[1] if len(parts) == 2 else None

        table_meta = tables_meta.get(table_name, {})
        inner = table_meta.get("table_metadata", table_meta)

        for rule_type, threshold in conditions.items():
            alert = _check_rule(rule_key, rule_type, threshold, table_name, column_name, inner)
            if alert:
                fired.append(alert)
                if on_alert:
                    try:
                        on_alert(rule_key, alert["table"], alert["column_or_metric"], alert["actual_value"])
                    except Exception as exc:
                        logger.warning("on_alert callback raised: %s", exc)

    return fired


def _check_rule(rule_key, rule_type, threshold, table_name, column_name, inner_meta):
    attributes = inner_meta.get("attributes", {})

    if rule_type == "quality_score_lt":
        score = inner_meta.get("data_quality_score", 100)
        if score < threshold:
            return _alert(rule_key, rule_type, table_name, "data_quality_score", score)

    if rule_type == "duplicate_rows" and threshold:
        dup = inner_meta.get("data_profiling", {}).get("duplicate_analysis", {}).get("duplicate_rows", 0)
        if dup > 0:
            return _alert(rule_key, rule_type, table_name, "duplicate_rows", dup)

    if column_name:
        attr = attributes.get(column_name, {})
        flags = attr.get("anomaly_flags", [])

        if rule_type == "null_rate_gt":
            null_pct = attr.get("null_percentage", 0)
            if null_pct > threshold:
                return _alert(rule_key, rule_type, table_name, column_name, null_pct)
        if rule_type == "negative_values" and threshold and "NEGATIVE_VALUES_DETECTED" in flags:
            return _alert(rule_key, rule_type, table_name, column_name, True)
        if rule_type == "future_dates" and threshold and "FUTURE_DATE_DETECTED" in flags:
            return _alert(rule_key, rule_type, table_name, column_name, True)
        if rule_type == "mixed_types" and threshold and "MIXED_DATA_TYPES" in flags:
            return _alert(rule_key, rule_type, table_name, column_name, True)
        if rule_type == "outliers_detected" and threshold and "NUMERIC_OUTLIERS_DETECTED" in flags:
            return _alert(rule_key, rule_type, table_name, column_name, True)
    else:
        if rule_type == "null_rate_gt":
            for col, attr in attributes.items():
                if attr.get("null_percentage", 0) > threshold:
                    return _alert(rule_key, rule_type, table_name, col, attr["null_percentage"])

    return None


def _alert(rule_key, rule_type, table, col_metric, value):
    return {"rule_key": rule_key, "rule_type": rule_type,
            "table": table, "column_or_metric": col_metric, "actual_value": value}


# ---------------------------------------------------------------------------
# Built-in helpers
# ---------------------------------------------------------------------------

def slack_webhook(url: str) -> AlertCallback:
    """Return an ``on_alert`` callback that posts to a Slack Incoming Webhook.

    Example::

        on_alert=slack_webhook("https://hooks.slack.com/services/T.../B.../...")
    """
    import urllib.request

    def _cb(rule_key, table, col_metric, actual_value):
        text = (
            f":warning: *ParseIQ Alert* — `{rule_key}`\n"
            f">Table: `{table}`  |  Field: `{col_metric}`  |  Value: `{actual_value}`"
        )
        payload = json.dumps({"text": text}).encode()
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=5):
                pass
        except Exception as exc:
            logger.warning("Slack webhook delivery failed: %s", exc)

    return _cb


def email(
    smtp_host: str,
    smtp_port: int,
    sender: str,
    recipients: List[str],
    subject_prefix: str = "[ParseIQ Alert]",
    username: Optional[str] = None,
    password: Optional[str] = None,
    use_tls: bool = True,
) -> AlertCallback:
    """Return an ``on_alert`` callback that sends an SMTP email.

    Example::

        on_alert=email(
            smtp_host="smtp.gmail.com", smtp_port=587,
            sender="alerts@company.com", recipients=["team@company.com"],
            username="alerts@company.com", password="app-password",
        )
    """

    def _cb(rule_key, table, col_metric, actual_value):
        msg = EmailMessage()
        msg["Subject"] = f"{subject_prefix} {rule_key} fired"
        msg["From"] = sender
        msg["To"] = ", ".join(recipients)
        msg.set_content(
            f"ParseIQ alert fired.\n\n"
            f"Rule key    : {rule_key}\n"
            f"Table       : {table}\n"
            f"Field/Metric: {col_metric}\n"
            f"Actual value: {actual_value}\n"
        )
        try:
            with smtplib.SMTP(smtp_host, smtp_port) as smtp:
                if use_tls:
                    smtp.starttls()
                if username and password:
                    smtp.login(username, password)
                smtp.send_message(msg)
        except Exception as exc:
            logger.warning("Email alert delivery failed: %s", exc)

    return _cb
