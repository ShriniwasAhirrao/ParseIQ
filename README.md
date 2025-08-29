# ParseIQ: AI-Powered Data Quality and Metadata Enrichment Agent

ParseIQ is a sophisticated, AI-driven agent designed to perform deep analysis of your data. It automates the process of metadata extraction, data profiling, anomaly detection, and intelligent enrichment for various file formats, including complex nested JSON, CSV, and XML files. By leveraging the power of large language models (LLMs), ParseIQ provides actionable insights and a comprehensive data quality assessment, helping you understand and improve your data assets.

## Key Features

- **Multi-Format Support:** Ingests and processes a variety of file formats, including JSON, CSV, XML, and Excel.
- **Recursive JSON Parsing:** Intelligently discovers and analyzes tables within deeply nested JSON structures, providing a complete picture of your hierarchical data.
- **Comprehensive Data Profiling:** Automatically extracts detailed technical metadata, including data types, statistical distributions, uniqueness, and anomaly detection.
- **AI-Powered Enrichment:** Uses a large language model to provide context-aware insights, business-level analysis, and actionable recommendations for data quality improvement.
- **User-Friendly Reports:** Generates detailed and easy-to-understand reports in CSV and Excel formats, including a comprehensive data quality scorecard.
- **Robust and Resilient:** Includes error handling and fallback mechanisms to ensure the pipeline runs smoothly even when faced with unexpected data or LLM issues.

## How It Works

ParseIQ operates on a simple yet powerful 3-step pipeline:

1.  **Extract:** The agent loads the input file and uses the `MetadataExtractor` to perform a deep technical analysis, generating a rich set of statistical and structural metadata.
2.  **Enrich:** The extracted metadata is then sent to a large language model, which, guided by a sophisticated prompt, provides a qualitative analysis, identifies business-level issues, and generates strategic recommendations.
3.  **Report:** The agent combines the technical metadata and the LLM's insights into a set of user-friendly reports, including individual CSV files for each discovered table and a consolidated Excel workbook with a comprehensive data quality assessment.
![Solution](<./ParseIQ Diagram.jpg>)



## Getting Started

### Prerequisites

- Python 3.8 or higher
- An [OpenRouter API key](https://openrouter.ai/)

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/ParseIQ.git
    cd ParseIQ
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure your API key:**

    Create a `.env` file in the project root and add your OpenRouter API key:

    ```
    OPENROUTER_API_KEY="your-api-key-here"
    ```

## Usage

1.  **Place your input file** in the `input/` directory. The agent is pre-configured to look for a file named `input_data.json`, but you can easily modify this in `main.py`.

2.  **Run the agent:**

    ```bash
    python main.py
    ```

3.  **Check the `output/` directory** for the analysis reports.

## Output

The agent generates a variety of output files in the `output/` directory, including:

-   **`raw_metadata.json`:** The complete technical metadata extracted from the input file.
-   **`enriched_metadata.json`:** The final output, combining the raw metadata with the LLM's insights.
-   **`llm_insights.json`:** The raw response from the language model.
-   **Individual CSV files:** For each table discovered in the input data, the agent generates:
    -   `*_data.csv`: The raw data for the table.
    -   `*_metadata.csv`: A detailed breakdown of the metadata for each attribute.
    -   `*_quality_report.csv`: A data quality assessment for the table.
-   **`complete_data_analysis.xlsx`:** A comprehensive Excel workbook containing all the generated reports in separate sheets.

## Configuration

The agent's behavior can be customized through the `config.py` file. Here, you can adjust settings such as:

-   **LLM model:** Change the language model used for enrichment.
-   **Data quality thresholds:** Modify the thresholds for anomaly detection.
-   **File processing settings:** Adjust the maximum file size and supported formats.

## Dependencies

This project relies on the following key libraries:

-   `requests`: For making API calls to the language model.
-   `pandas`: For data manipulation and analysis.
-   `numpy` and `scipy`: For statistical calculations.
-   `openpyxl`: For creating Excel reports.
-   `xmltodict`: For parsing XML files.
-   `python-dotenv`: For managing environment variables.

For a complete list of dependencies, please see the `requirements.txt` file.
