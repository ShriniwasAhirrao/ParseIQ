"""
Generate stress_test_data.json — 14-level nested, 250+ records, rich anomalies.

Level hierarchy:
  1  enterprise
  2  continents
  3  countries
  4  regions
  5  cities
  6  offices
  7  departments
  8  teams
  9  employees
  10 assignments
  11 tasks
  12 subtasks
  13 activities
  14 activity_logs
"""
import json
import random
import os

random.seed(42)

SKILLS = ["Python","Go","Java","Rust","TypeScript","SQL","ML","Docker","Kubernetes","React",
          "AWS","GCP","Azure","Terraform","Spark","Redis","Kafka","GraphQL","Swift","Kotlin"]
TITLES = ["Analyst","Engineer","Senior Engineer","Lead Engineer","Manager","Director","VP","Intern",
          "Consultant","Architect","Principal Engineer","Fellow","CTO","CPO","Head of"]
STATUSES = ["active","inactive","pending","blocked","completed","review"]
ACTIONS = ["CREATE","UPDATE","DELETE","READ","EXPORT","IMPORT","APPROVE","REJECT"]
FOCUSES = ["frontend","backend","data","devops","qa","security","product","mobile","cloud","infra"]
CURRENCIES = ["USD","EUR","GBP","JPY","CNY","INR","AED","SGD","BRL","KRW"]
TIMEZONES = ["UTC-8","UTC-5","UTC-1","UTC+0","UTC+1","UTC+3","UTC+5:30","UTC+7","UTC+8","UTC+9"]


# ── helpers ─────────────────────────────────────────────────────────────────

def rand_date(y_start=2010, y_end=2024):
    y = random.randint(y_start, y_end)
    m = random.randint(1, 12)
    d = random.randint(1, 28)
    return f"{y}-{m:02d}-{d:02d}"


def rand_email(name_slug):
    return f"{name_slug}@megaglobal.com"


# ── level 14: activity_log ───────────────────────────────────────────────────

def make_activity_log(log_id, inject_anomaly=False):
    ts = "2099-03-15T12:00:00Z" if inject_anomaly else \
         f"2024-{random.randint(1,9):02d}-{random.randint(10,28):02d}T{random.randint(8,18):02d}:30:00Z"
    action = 999 if inject_anomaly else random.choice(ACTIONS)   # MIXED_DATA_TYPES
    ip = "999.888.777.666" if inject_anomaly else \
         f"{random.randint(1,254)}.{random.randint(0,254)}.{random.randint(0,254)}.{random.randint(1,254)}"
    return {
        "log_id":    log_id,
        "timestamp": ts,
        "action":    action,
        "success":   random.choice([True, False]),
        "ip_address": ip,
        "error_code": None if not inject_anomaly else -500,
        "session_id": f"SES-{random.randint(10000,99999)}"
    }


# ── level 13: activities ─────────────────────────────────────────────────────

def make_activity(act_id, inject_anomaly=False):
    hours = -random.uniform(1, 5) if inject_anomaly else random.uniform(0.5, 10)
    atype = random.choice(["work", "meeting", "review", "training"])
    if inject_anomaly:
        atype = random.choice([atype, 42, None])  # MIXED_DATA_TYPES
    return {
        "activity_id": act_id,
        "type":        atype,
        "hours":       round(hours, 2),
        "date":        "2028-11-11" if inject_anomaly else rand_date(2023, 2024),
        "billable":    random.choice([True, False]),
        "activity_logs": [
            make_activity_log(f"{act_id}-LG{i}", inject_anomaly=(i == 1 and inject_anomaly))
            for i in range(1, 3)
        ]
    }


# ── level 12: subtasks ───────────────────────────────────────────────────────

def make_subtask(st_id, inject_anomaly=False):
    comp = 150.0 if inject_anomaly else round(random.uniform(0, 100), 1)  # impossible %
    name = None if inject_anomaly else f"Subtask {st_id}"
    return {
        "subtask_id":     st_id,
        "name":           name,
        "status":         random.choice(STATUSES),
        "completion_pct": comp,
        "due_date":       "2099-12-31" if inject_anomaly else rand_date(2024, 2025),
        "story_points":   -random.randint(1, 13) if inject_anomaly else random.randint(1, 13),
        "activities": [
            make_activity(f"{st_id}-ACT{i}", inject_anomaly=(i == 1 and inject_anomaly))
            for i in range(1, 3)
        ]
    }


# ── level 11: tasks ──────────────────────────────────────────────────────────

def make_task(task_id, inject_anomaly=False):
    est = -random.randint(5, 40) if inject_anomaly else random.randint(5, 120)
    title = None if inject_anomaly else f"Task {task_id}"
    priority = random.choice(["low", "medium", "high", "critical"])
    if inject_anomaly:
        priority = random.choice([priority, 99, None])  # MIXED_DATA_TYPES
    return {
        "task_id":         task_id,
        "title":           title,
        "priority":        priority,
        "estimated_hours": est,
        "actual_hours":    random.randint(0, 150),
        "due_date":        "2099-06-30" if inject_anomaly else rand_date(2024, 2025),
        "assigned_to":     None if inject_anomaly else f"EMP-{random.randint(1000,9999)}",
        "subtasks": [
            make_subtask(f"{task_id}-ST{i}", inject_anomaly=(i == 1 and inject_anomaly))
            for i in range(1, 3)
        ]
    }


# ── level 10: assignments ────────────────────────────────────────────────────

def make_assignment(asgn_id, inject_anomaly=False):
    alloc = 120.0 if inject_anomaly else round(random.uniform(10, 100), 1)
    return {
        "assignment_id":  asgn_id,
        "project_code":   f"PRJ-{random.randint(100,999)}",
        "role":           random.choice(["lead","contributor","reviewer","observer"]),
        "start_date":     "2030-01-01" if inject_anomaly else rand_date(2022, 2024),
        "end_date":       None if inject_anomaly else rand_date(2024, 2025),
        "allocation_pct": alloc,
        "billing_rate":   -random.randint(50, 200) if inject_anomaly else random.randint(50, 400),
        "tasks": [
            make_task(f"{asgn_id}-TK{i}", inject_anomaly=(i == 1 and inject_anomaly))
            for i in range(1, 3)
        ]
    }


# ── level 9: employees ───────────────────────────────────────────────────────

_EMP_ANOMALIES = ["bad_email","negative_salary","outlier_salary","future_hire","null_row","mixed_is_active",None,None,None,None]

def make_employee(emp_id, anomaly=None, duplicate_prototype=None):
    if duplicate_prototype is not None:
        # exact duplicate row
        return dict(duplicate_prototype)

    if anomaly == "null_row":
        return {k: None for k in
                ["emp_id","name","email","salary","hire_date","title","is_active","age","department_code","assignments"]}

    email_map = {
        "bad_email":  "MISSING_AT_SIGN_AND_DOMAIN",
        None: rand_email(emp_id.replace("-","_").lower())
    }
    salary_map = {
        "negative_salary":  -random.randint(1000, 9999),
        "outlier_salary":    9_999_999,
        None:                random.randint(35_000, 200_000)
    }
    hire_map = {
        "future_hire": "2029-07-01",
        None: rand_date(2008, 2023)
    }
    is_active_val = random.choice([True, False, "yes", 1]) if anomaly == "mixed_is_active" else random.choice([True, False])

    return {
        "emp_id":         emp_id,
        "name":           f"Employee {emp_id}",
        "email":          email_map.get(anomaly, email_map[None]),
        "salary":         salary_map.get(anomaly, salary_map[None]),
        "hire_date":      hire_map.get(anomaly, hire_map[None]),
        "title":          random.choice(TITLES),
        "is_active":      is_active_val,
        "age":            random.randint(22, 65),
        "department_code": f"D{random.randint(100,999)}",
        "skills":         random.sample(SKILLS, k=random.randint(1, 5)),
        "middle_name":    None if random.random() < 0.4 else f"M{random.randint(1,99)}",  # HIGH_NULL_RATE
        "assignments": [
            make_assignment(f"{emp_id}-ASN{i}", inject_anomaly=(anomaly == "null_row" or (i == 1 and random.random() < 0.1)))
            for i in range(1, 3)
        ]
    }


def make_employees_for_team(team_id, team_idx):
    emps = []
    anom_pick = list(_EMP_ANOMALIES)
    random.shuffle(anom_pick)
    first_proto = None
    for e in range(1, 5):
        eid = f"{team_id}-EMP{e}"
        anom = anom_pick[e - 1]
        emp = make_employee(eid, anomaly=anom)
        if e == 1:
            first_proto = emp
        emps.append(emp)
    # one exact duplicate of first employee
    emps.append(make_employee(None, duplicate_prototype=first_proto))
    return emps


# ── level 8: teams ───────────────────────────────────────────────────────────

def make_team(team_id, team_idx):
    return {
        "team_id":    team_id,
        "name":       f"Team {team_id}",
        "focus":      random.choice(FOCUSES),
        "kpi_target": round(random.uniform(0.6, 1.0), 2),
        "formed_date": rand_date(2015, 2023),
        "budget_allocated": -random.randint(10000, 50000) if random.random() < 0.08 else random.randint(50000, 500000),
        "employees": make_employees_for_team(team_id, team_idx)
    }


# ── level 7: departments ─────────────────────────────────────────────────────

DEPT_NAMES = ["Engineering","Sales","Finance","HR","Product","Legal","Marketing","Operations","R&D","Compliance"]

def make_department(dept_id, dept_idx):
    teams = [make_team(f"{dept_id}-TM{i}", dept_idx * 10 + i) for i in range(1, 3)]
    return {
        "dept_id":       dept_id,
        "name":          random.choice(DEPT_NAMES),
        "headcount":     random.randint(5, 300),
        "annual_budget": -random.randint(100_000, 500_000) if random.random() < 0.08 else random.randint(500_000, 15_000_000),
        "manager_email": None if random.random() < 0.15 else rand_email(f"mgr_{dept_id.lower()}"),
        "cost_center":   f"CC-{random.randint(1000,9999)}",
        "teams":         teams
    }


# ── level 6: offices ─────────────────────────────────────────────────────────

def make_office(off_id, off_idx):
    depts = [make_department(f"{off_id}-DP{i}", off_idx * 10 + i) for i in range(1, 3)]
    return {
        "office_id":   off_id,
        "address":     f"{random.randint(1,9999)} Business Ave",
        "phone":       None if random.random() < 0.2 else f"+1-555-{random.randint(1000,9999)}",
        "capacity":    random.randint(50, 1000),
        "is_hq":       random.random() < 0.05,
        "opened_date": rand_date(1990, 2022),
        "lease_cost_usd_monthly": -random.randint(5000, 20000) if random.random() < 0.05 else random.randint(5000, 80000),
        "departments": depts
    }


# ── level 5: cities ──────────────────────────────────────────────────────────

def make_city(city_id, city_idx):
    offices = [make_office(f"{city_id}-OF{i}", city_idx * 10 + i) for i in range(1, 3)]
    return {
        "city_id":    city_id,
        "name":       f"City-{city_id}",
        "population": random.randint(100_000, 10_000_000),
        "timezone":   random.choice(TIMEZONES),
        "latitude":   round(random.uniform(-90, 90), 4),
        "longitude":  round(random.uniform(-180, 180), 4),
        "offices":    offices
    }


# ── level 4: regions ─────────────────────────────────────────────────────────

def make_region(region_id, region_idx):
    cities = [make_city(f"{region_id}-CT{i}", region_idx * 10 + i) for i in range(1, 3)]
    return {
        "region_id":      region_id,
        "name":           f"Region-{region_id}",
        "area_km2":       random.randint(50_000, 1_000_000),
        "gdp_usd_bil":    round(random.uniform(20, 3000), 1),
        "cities":         cities
    }


# ── level 3: countries ───────────────────────────────────────────────────────

def make_country(country_id, country_idx):
    regions = [make_region(f"{country_id}-RG{i}", country_idx * 10 + i) for i in range(1, 3)]
    return {
        "country_id":          country_id,
        "name":                f"Country-{country_id}",
        "iso_code":            country_id[:2].upper(),
        "population_millions": round(random.uniform(0.5, 1400), 2),
        "currency":            random.choice(CURRENCIES),
        "gdp_trillion_usd":    round(random.uniform(0.01, 25), 3),
        "tax_rate_pct":        round(random.uniform(5, 55), 1),
        "regions":             regions
    }


# ── level 2: continents ──────────────────────────────────────────────────────

CONT_NAMES = ["Americas","Europe","Asia-Pacific","Middle East & Africa","Oceania"]

def make_continent(cont_id, cont_idx):
    countries = [make_country(f"{cont_id}-CO{i}", cont_idx * 10 + i) for i in range(1, 3)]
    return {
        "continent_id": cont_id,
        "name":         CONT_NAMES[cont_idx % len(CONT_NAMES)],
        "area_km2":     random.randint(5_000_000, 44_000_000),
        "countries":    countries
    }


# ── level 1: enterprise ──────────────────────────────────────────────────────

continents = [make_continent(f"CN{i}", i) for i in range(1, 4)]  # 3 continents

enterprise = {
    "enterprise_id":        "ENT-001",
    "name":                 "MegaGlobal Corp",
    "founded_date":         "1970-01-15",
    "hq_city":              "New York",
    "ceo_name":             "John Doe",
    "ceo_email":            "ceo@megaglobal.com",
    "annual_revenue_usd":   50_000_000_000,
    "employee_count":       95000,
    "stock_ticker":         "MGC",
    "is_public":            True,
    "fiscal_year_end_month": 12,
    "credit_rating":        "AAA",
    "nps_score":            72,
    "last_audit_date":      "2099-01-01",   # FUTURE_DATE
    "net_margin_pct":       -2.5,            # NEGATIVE
    "continents":           continents
}

# Inject a top-level enterprise duplicate as well
enterprise_dup = dict(enterprise)
enterprise_dup["continents"] = []  # lighter duplicate

output = [enterprise, enterprise_dup]

# ── write file ───────────────────────────────────────────────────────────────

out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "input", "stress_test_data.json")

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, default=str)

# ── count records ────────────────────────────────────────────────────────────

def count_levels(obj, level=0, counts=None):
    if counts is None:
        counts = {}
    if isinstance(obj, list):
        for item in obj:
            count_levels(item, level, counts)
    elif isinstance(obj, dict):
        counts[level] = counts.get(level, 0) + 1
        for v in obj.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                count_levels(v, level + 1, counts)
    return counts

c = count_levels(output)
print("Records per nesting level:")
level_names = {0:"enterprise",1:"continents",2:"countries",3:"regions",4:"cities",
               5:"offices",6:"departments",7:"teams",8:"employees",
               9:"assignments",10:"tasks",11:"subtasks",12:"activities",13:"activity_logs"}
total = 0
for lv in sorted(c):
    n = c[lv]
    total += n
    print(f"  Level {lv:02d} ({level_names.get(lv,'unknown'):20s}): {n:5d} records")
print(f"\nTotal records : {total}")
file_size_kb = os.path.getsize(out_path) / 1024
print(f"File size     : {file_size_kb:.1f} KB  ({out_path})")
