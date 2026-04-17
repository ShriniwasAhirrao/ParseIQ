# ParseIQ — System Architecture & Workflow Diagrams

> Author: Shriniwas Ahirrao
> Project: ParseIQ — AI-Powered Data Quality Agent
> Date: April 2026

---

## 1. High-Level System Architecture

```mermaid
graph TB
    subgraph "Input Layer"
        A[JSON File] --> FL
        B[CSV File] --> FL
        C[XML File] --> FL
        D[Excel File] --> FL
        E[URL / S3 / DB] --> CON[Connectors]
        CON --> FL
    end

    FL[FileLoader<br/>Format Detection<br/>Encoding Detection<br/>Delimiter Sniffing] --> NORM[Normalised Data<br/>Dict of str to List of Dict]

    subgraph "Step 1: Metadata Extraction"
        NORM --> FLAT[Nested JSON Flattener<br/>Recursive + FK Injection]
        FLAT --> TABLES[Table Registry<br/>Multiple Tables]
        TABLES --> PROF[Per-Attribute Profiler<br/>30+ Statistics]
        PROF --> ANOM[Anomaly Detectors<br/>11 Types]
        ANOM --> SCORE[Quality Scorer<br/>Rate-Based Algorithm]
        SCORE --> CROSS[Cross-Table Analysis<br/>Range Violations<br/>Schema Polymorphism]
        CROSS --> RULES[Rules Engine<br/>YAML/JSON Sidecar]
        RULES --> META[Raw Metadata<br/>Complete Profile]
    end

    subgraph "Step 2: LLM Enrichment (Optional)"
        META --> COMP[Metadata Compressor<br/>Summary + Samples]
        COMP --> ROUTE[Provider Router<br/>7+ Providers]
        ROUTE --> LLM[LLM API Call<br/>BYOK Architecture]
        LLM --> MERGE[Merge Enrichment<br/>Grade + Recommendations]
        LLM -.->|Failure| FALL[Fallback<br/>Local Enrichment]
        FALL --> MERGE
    end

    subgraph "Step 3: Output Generation"
        META --> OUT
        MERGE --> OUT[Output Builder]
        OUT --> XLS[Excel Workbook<br/>Grouped Sheets]
        OUT --> CSV_OUT[CSV Summaries<br/>2 Files]
        OUT --> JSON_OUT[JSON Metadata<br/>3 Files]
    end

    style FL fill:#1a1a2e,stroke:#e94560,color:#fff
    style FLAT fill:#1a1a2e,stroke:#e94560,color:#fff
    style ANOM fill:#1a1a2e,stroke:#0f3460,color:#fff
    style LLM fill:#1a1a2e,stroke:#16213e,color:#fff
    style OUT fill:#1a1a2e,stroke:#533483,color:#fff
```

---

## 2. Nested JSON Flattening Pipeline

```mermaid
graph TD
    ROOT["Root JSON Object"] --> CHECK{"Value Type?"}
    
    CHECK -->|"List of Dicts"| CHILD["Extract as Child Table<br/>Inject _ref_{parent}_id"]
    CHECK -->|"Dict"| RECURSE["Recurse into Dict"]
    CHECK -->|"Primitive"| LEAF["Add as Column"]
    CHECK -->|"Primitive List"| JOIN["Join as comma-separated string"]
    
    RECURSE --> DEEP["_deep_flatten_scalars()<br/>Walk dict tree"]
    DEEP --> SCALAR["Scalar Leaves<br/>prefix__key1__key2 = value"]
    SCALAR --> PARENT["Add to Parent Record"]
    
    RECURSE --> SUBCHECK{"Nested Arrays?"}
    SUBCHECK -->|Yes| CHILD
    
    CHILD --> FKGEN["Generate FK Column<br/>_ref_{parent}_id = index"]
    FKGEN --> CHILDTABLE["New Table Entry"]
    CHILDTABLE --> RECURSE2["Recurse into<br/>Child Records"]
    
    LEAF --> PARENT
    JOIN --> PARENT
```

### Flattening Example

```
Input:
{
  "departments": [
    {
      "name": "Engineering",
      "head": {
        "name": "Alice",
        "performance": {
          "fy2025": { "rating": 4.5, "bonus": 15000 }
        }
      },
      "team": [
        { "name": "Bob", "role": "SDE" },
        { "name": "Carol", "role": "PM" }
      ]
    }
  ]
}

Output Tables:
  departments:
    | name        | head__name | head__performance__fy2025__rating | head__performance__fy2025__bonus |
    |-------------|------------|-----------------------------------|----------------------------------|
    | Engineering | Alice      | 4.5                               | 15000                            |

  team:
    | _ref_departments_id | name  | role |
    |---------------------|-------|------|
    | 0                   | Bob   | SDE  |
    | 0                   | Carol | PM   |
```

---

## 3. Anomaly Detection Pipeline

```mermaid
graph LR
    subgraph "Per-Attribute Detectors"
        A1[NULL Rate Check<br/>threshold: 30%]
        A2[Uniqueness Check<br/>threshold: 10%<br/>boolean exempt]
        A3[Mixed Types Check<br/>int+float grouped]
        A4[Future Date Check<br/>vs datetime.today]
        A5[Outlier Detection<br/>Z-score + IQR dual]
        A6[Negative Values<br/>domain-aware suppression]
        A7[Pattern Check<br/>email/phone/UUID]
        A8[Duplicate Rows<br/>full-row hash]
    end

    subgraph "Cross-Table Detectors"
        B1[Range Violation<br/>parent *_range* columns<br/>name-match + FK fallback]
        B2[Schema Polymorphism<br/>discriminator detection<br/>type-conditional reclassify]
    end

    subgraph "Rule-Based Detectors"
        C1[Scale Violation<br/>max_value / min_value]
        C2[Constraint Violation<br/>cross_table_compare<br/>FK join + comparison]
    end

    A1 --> FLAGS[Anomaly Flags]
    A2 --> FLAGS
    A3 --> FLAGS
    A4 --> FLAGS
    A5 --> FLAGS
    A6 --> FLAGS
    A7 --> FLAGS
    A8 --> FLAGS
    B1 --> FLAGS
    B2 --> FLAGS
    C1 --> FLAGS
    C2 --> FLAGS

    FLAGS --> SCORING[Quality Scoring<br/>Per-attribute + Per-table]
```

---

## 4. Quality Scoring Algorithm

```mermaid
graph TD
    subgraph "Per-Attribute Score"
        BASE1["Base = 100"] --> MP["Missing Penalty<br/>scaled by severity"]
        MP --> AP["Anomaly Penalty<br/>-15pt per flag"]
        AP --> OP["Outlier Penalty<br/>if applicable"]
        OP --> CLAMP1["max(0, score)"]
    end

    subgraph "Per-Table Score"
        CLAMP1 --> AVG["Average of<br/>attribute scores"]
        AVG --> RATE["Rate Penalty<br/>min(anomaly_rate * 20, 20)"]
        RATE --> DUP["Duplicate Penalty<br/>min(dup_rate * 20, 20)"]
        DUP --> CLAMP2["max(0, min(100, score))"]
    end

    subgraph "Overall Score"
        CLAMP2 --> WAVG["Weighted Average<br/>by record count"]
        WAVG --> FINAL["Overall Dataset Score<br/>0-100"]
    end
```

### Why Rate-Based Penalty?

**Before (v0.0.2)**: `table_penalty = total_anomalies * 3`
- Problem: Wide table with 62 columns and 40 anomalies -> penalty = 120 -> score = 0
- Root cause: Raw count grows linearly with table width

**After (v0.0.3)**: `table_penalty = min(anomaly_rate * 20, 20)`
- Fix: Penalty is proportional to fraction of affected columns, capped at 20
- Result: Same table -> penalty = min(40/62 * 20, 20) = 12.9 -> meaningful non-zero score

---

## 5. LLM Provider Routing

```mermaid
graph TD
    REQ["LLM Request"] --> DETECT{"Provider<br/>Detection"}
    
    DETECT -->|"model: claude-*"| ANTHROPIC["Anthropic SDK<br/>anthropic.Anthropic()"]
    DETECT -->|"model: gemini-*"| GEMINI["Gemini SDK<br/>genai.GenerativeModel()"]
    DETECT -->|"provider: openrouter"| OPENAI_COMPAT["OpenAI-Compatible REST<br/>POST /chat/completions"]
    DETECT -->|"provider: openai"| OPENAI_COMPAT
    DETECT -->|"provider: perplexity"| OPENAI_COMPAT
    DETECT -->|"provider: azure"| OPENAI_COMPAT
    DETECT -->|"provider: ollama"| OPENAI_COMPAT
    
    ANTHROPIC --> RESPONSE["LLM Response"]
    GEMINI --> RESPONSE
    OPENAI_COMPAT --> RESPONSE
    
    RESPONSE -->|Success| PARSE["Parse JSON Response"]
    RESPONSE -->|"402 Error"| CREDIT["Credit Exhaustion<br/>Show free alternatives"]
    RESPONSE -->|"Other Error"| FALLBACK["Local Fallback<br/>No crash, full report"]
    
    PARSE --> ENRICH["Enriched Metadata"]
    CREDIT --> FALLBACK
    FALLBACK --> LOCAL["Local Enrichment<br/>Quality grade from scores<br/>Auto-generated recommendations"]
```

---

## 6. Web UI Architecture

```mermaid
graph TB
    subgraph "Browser (React SPA)"
        UP[Upload Page<br/>FileDropzone + Config] -->|POST /api/upload| API
        UP -->|"file + settings"| API
        
        PROC[Processing Page<br/>Real-time Event Feed] -->|"GET /api/job/{id}<br/>polling"| API
        
        DASH[Dashboard Page<br/>Score Gauges + Tables] -->|"GET /api/results/{id}"| API
        
        DETAIL[Table Detail Page<br/>Columns + Data + Nested] -->|"GET /api/results/{id}/table/{name}"| API
        
        SETTINGS[Settings Page<br/>API Key + Model + Provider] -->|"POST /api/settings"| API
    end

    subgraph "FastAPI Backend"
        API[API Router] --> UPLOAD[Upload Route<br/>Chunked 1MB streaming<br/>File validation]
        API --> JOB[Job Status Route<br/>Events since timestamp]
        API --> RESULTS[Results Route<br/>Table summaries + detail]
        API --> DOWNLOAD[Download Route<br/>Path traversal protection]
        
        UPLOAD --> RUNNER[Pipeline Runner<br/>Thread-safe Job Store]
        RUNNER --> POOL["Thread Pool<br/>_job_semaphore(4)"]
    end

    subgraph "Pipeline Worker Thread"
        POOL --> WORKER["Worker Thread"]
        WORKER --> STDOUT["Capture stdout<br/>_stdout_lock"]
        WORKER --> PIPELINE["ParseIQ Pipeline<br/>Step 1 + Step 2 + Step 3"]
        STDOUT --> EVENTS["Job Events<br/>Timestamped + Redacted"]
        PIPELINE --> OUTPUT["Output Files<br/>Excel + CSV + JSON"]
    end

    style UP fill:#0d1117,stroke:#58a6ff,color:#fff
    style PROC fill:#0d1117,stroke:#58a6ff,color:#fff
    style DASH fill:#0d1117,stroke:#58a6ff,color:#fff
    style API fill:#161b22,stroke:#8b949e,color:#fff
    style RUNNER fill:#161b22,stroke:#8b949e,color:#fff
    style PIPELINE fill:#21262d,stroke:#f0883e,color:#fff
```

### Thread Safety Model

```
Main Thread (FastAPI/Uvicorn)
    |
    +-- Request Handler
    |       |
    |       +-- _jobs_lock.acquire()
    |       +-- Read/write _jobs dict
    |       +-- _jobs_lock.release()
    |
    +-- Background Workers (up to 4)
            |
            +-- _job_semaphore.acquire()  # blocks if 4 running
            +-- _stdout_lock.acquire()
            +-- sys.stdout = StringIO()   # capture pipeline output
            +-- _stdout_lock.release()
            +-- Run ParseIQ pipeline
            +-- _stdout_lock.acquire()
            +-- sys.stdout = original     # restore
            +-- _stdout_lock.release()
            +-- _job_semaphore.release()
```

---

## 7. Data Flow — End to End

```mermaid
sequenceDiagram
    participant U as User
    participant FE as React Frontend
    participant API as FastAPI Backend
    participant W as Worker Thread
    participant P as ParseIQ Pipeline

    U->>FE: Drop file + configure settings
    FE->>API: POST /api/upload (multipart, 1MB chunks)
    API->>API: Validate file type + size
    API->>W: Create job + spawn worker thread
    API-->>FE: { job_id }
    FE->>FE: Navigate to /processing/{job_id}

    loop Every 1s
        FE->>API: GET /api/job/{id}?since={ts}
        API-->>FE: { status, events[], progress }
        FE->>FE: Render event feed + progress bar
    end

    W->>P: Pipeline.run(file, llm=settings.useLlm)
    P->>P: Step 1: Extract metadata
    P->>P: Step 2: LLM enrichment (optional)
    P->>P: Step 3: Generate outputs
    P-->>W: PipelineResult

    W->>API: Update job status = "completed"
    FE->>API: GET /api/job/{id} -> status: completed
    FE->>FE: Navigate to /results/{job_id}

    FE->>API: GET /api/results/{id}
    API-->>FE: { tables, scores, anomalies }
    FE->>FE: Render dashboard

    U->>FE: Click table card
    FE->>API: GET /api/results/{id}/table/{name}
    API-->>FE: { columns, data_preview, nested_tables }
    FE->>FE: Render table detail
```

---

## 8. Excel Output Structure

```
complete_data_analysis.xlsx
|
+-- 00_Summary
|   [Table_Name | Records | Quality_Score | Anomalies | Top_Issues]
|
+-- 01_LLM_Assessment  (if LLM used)
|   [Quality_Grade | Overall_Score | Production_Readiness |
|    Key_Strengths | Primary_Concerns | Model_Used]
|
+-- 02_LLM_Recommendations  (if LLM used)
|   [Priority | Category | Recommendation | Expected_Impact | Effort]
|
+-- Data_employees        <- Raw data rows
+-- Meta_employees        <- 30-column attribute profile
+-- Quality_employees     <- Long-format quality metrics
|
+-- Data_departments
+-- Meta_departments
+-- Quality_departments
|
+-- ... (3 sheets per table, grouped)
|
+-- 99_Issues_Recommendations
    [Priority | Table | Column | Issue_Type | Description |
     Business_Impact | Recommended_Fix | Effort]
    (Sorted: CRITICAL -> HIGH -> MEDIUM -> LOW)
```

---

## 9. Incremental Processing Flow

```mermaid
graph TD
    START["Pipeline.run()"] --> CHECK{"Cache file<br/>exists?"}
    
    CHECK -->|No| FULL["Full Analysis<br/>All Tables"]
    CHECK -->|Yes| LOAD["Load Cache<br/>.parseiq_cache.json"]
    
    LOAD --> HASH["Compute SHA-256<br/>per table"]
    HASH --> COMPARE{"Hash<br/>matches?"}
    
    COMPARE -->|Same| SKIP["Skip Table<br/>Reuse Previous Results"]
    COMPARE -->|Different| ANALYSE["Analyse Table<br/>Full Profiling"]
    
    FULL --> SAVE["Save Cache<br/>Update Hashes"]
    ANALYSE --> SAVE
    SKIP --> MERGE["Merge Results"]
    ANALYSE --> MERGE
    MERGE --> OUTPUT["Generate Output"]
    SAVE --> OUTPUT
    
    FORCE["--force Flag"] -->|Override| FULL
```

---

## 10. Configuration Priority Chain

```
Highest Priority
    |
    v
[1] Pipeline.run() Parameters
    llm_api_key="sk-...", llm_model="gpt-4o"
    |
    v
[2] CLI Flags
    --llm-api-key, --llm-model, --llm-provider
    |
    v
[3] Environment Variables
    OPENROUTER_API_KEY, OPENAI_API_KEY, etc.
    |
    v
[4] .env File (auto-loaded if python-dotenv installed)
    OPENROUTER_API_KEY=sk-or-v1-...
    |
    v
[5] Built-in Defaults
    provider=openrouter, model=nvidia/nemotron-3-super-120b-a12b:free

Lowest Priority
```

---

## 11. Deployment Topology

```mermaid
graph TB
    subgraph "Development"
        DEV_FE["Vite Dev Server<br/>:5173<br/>Hot Module Reload"]
        DEV_BE["Uvicorn<br/>:8000<br/>Auto-reload"]
        DEV_FE <-->|"proxy /api/*"| DEV_BE
    end

    subgraph "Production"
        STATIC["FastAPI Static Mount<br/>frontend/dist/"]
        PROD_BE["Uvicorn<br/>:8000<br/>host=127.0.0.1"]
        STATIC --> PROD_BE
    end

    subgraph "CLI Mode"
        CLI["parseiq analyze data.json"]
        CLI --> PIP["ParseIQ Pipeline<br/>Direct Execution"]
    end

    subgraph "CI/CD"
        CI["parseiq analyze data.json<br/>--no-llm --fail-under 80"]
        CI --> EXIT{"Exit Code"}
        EXIT -->|"0"| PASS["Quality >= 80<br/>Pipeline Continues"]
        EXIT -->|"1"| FAIL["Quality < 80<br/>Pipeline Fails"]
    end
```

---

## 12. Component Interaction Map

```
parseiq/
  __init__.py          <-- Public API surface
       |
  pipeline.py          <-- Orchestrator (Step 1 + 2 + 3)
       |
       +-- file_loader/loader.py
       |       |
       |       +-- _flatten_nested_json()
       |       +-- _deep_flatten_scalars()
       |       +-- _load_json/csv/xml/excel()
       |
       +-- step1_metadata_extractor/extractor.py
       |       |
       |       +-- _analyze_table_detailed()
       |       +-- _detect_anomalies()
       |       +-- _detect_schema_groups()
       |       +-- _calculate_quality_score()
       |       +-- _detect_cross_level_range_violations()
       |       +-- utils.py (zscore, IQR)
       |
       +-- step2_llm_enricher/llm_agent.py
       |       |
       |       +-- _detect_provider()
       |       +-- _make_api_request_anthropic()
       |       +-- _make_api_request_gemini()
       |       +-- _make_api_request_openai_compatible()
       |       +-- _create_fallback_enrichment()
       |       +-- _calculate_corrected_quality_score()
       |
       +-- _generate_outputs()
       |       +-- Excel: 00_Summary, 01/02_LLM, Data/Meta/Quality per table, 99_Issues
       |       +-- CSV: summary + issues
       |       +-- JSON: raw + enriched + pipeline_info
       |
       +-- _find_rules_file() / _load_rules() / _apply_rules()
       +-- alerts.py (post-analysis rule evaluation)
       +-- result.py (PipelineResult dataclass)

  connectors/
       +-- file.py    <-- Local files
       +-- url.py     <-- HTTP/S download
       +-- s3.py      <-- Amazon S3
       +-- postgres.py <-- PostgreSQL
       +-- mongodb.py  <-- MongoDB

  _cli.py              <-- CLI entry point (parseiq command)
       +-- analyze, validate, init, models, config, version

web/
  run.py               <-- Dev/prod launcher
  api/
       +-- main.py     <-- FastAPI app, CORS, exception handlers
       +-- routes/upload.py     <-- Chunked multipart upload
       +-- routes/results.py    <-- Table summaries + detail + download
       +-- routes/settings.py   <-- Configuration endpoints
       +-- services/pipeline_runner.py  <-- Thread-safe job management
  frontend/
       +-- src/App.tsx          <-- Router + ErrorBoundary
       +-- src/pages/           <-- 5 page components
       +-- src/components/      <-- 8+ reusable components
       +-- src/hooks/           <-- useJobPoller
       +-- src/lib/             <-- API client + types
```
