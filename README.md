# AI Data Quality Analyst

An intelligent **data quality assessment application** built with **Streamlit** that allows users to upload datasets, define their intended use, and evaluate whether the data is ready for that purpose.

The application performs structural data analysis, calculates a readiness score, generates actionable recommendations, and produces a **professional PDF report** summarizing the results.

---

# Overview

Before building reports, performing analysis, or training machine learning models, it is critical to understand whether the dataset is **actually suitable for the intended task**.

Many datasets appear usable but contain hidden problems such as:

- Missing values
- Duplicate rows
- Inconsistent data types
- Extreme outliers
- Poor column structure

This application helps analysts and teams answer the key question:

> **Is this dataset ready for my goal?**

The system evaluates the dataset based on the selected goal and provides:

- A **Readiness Score (0–100)**
- A **Goal-based suitability status**
- **Automated dataset diagnostics**
- **AI-powered analysis**
- **Interactive dashboards**
- **A downloadable PDF report**

---

# Key Features

## 1. Dataset Upload

The application supports common data formats:

- CSV files
- Excel (.xlsx)
- Excel (.xls)

Users can upload datasets directly through the Streamlit interface.

---

## 2. Goal-Based Data Evaluation

The dataset is evaluated based on the user's intended use.

Available goals:

- **Build Report**
- **Data Analysis**
- **Build Model**

Each goal uses a slightly different scoring strategy because:

- Reporting tolerates some issues
- Analysis requires cleaner data
- Machine learning requires stricter quality

---

## 3. Automated Data Quality Checks

The system automatically analyzes:

### Dataset Structure
- Number of rows
- Number of columns
- Numeric vs text columns

### Missing Values
- Total missing cells
- Missing values per column

### Duplicate Rows
- Duplicate detection
- Duplicate ratio relative to dataset size

### Outlier Detection
- Numeric outliers detected using the **IQR method**

---

## 4. AI-Powered Dataset Assessment

The application integrates with **OpenAI** to generate a professional analysis of the dataset.

Users can ask questions such as:

- Is this dataset suitable for machine learning?
- What issues should be fixed before analysis?
- Is the dataset ready for building a report?

The AI produces a structured response including:

- Executive Summary
- Key Findings
- Readiness Score
- Suitability Assessment
- Recommended Fixes

---

## 5. Interactive Dashboard

After analysis, the application displays a visual dashboard including:

### Readiness Score
A progress indicator showing the dataset readiness level.

### Dataset Metrics
- Rows
- Columns
- Missing cells
- Duplicate rows

### Visual Data Quality Insights

Charts include:

- Missing values by column
- Data type distribution
- Outliers by column

These visualizations help quickly identify problematic areas in the dataset.

---

## 6. Professional PDF Report

The system can generate a **downloadable PDF report** containing:

- Dataset metadata
- Selected goal
- Readiness score
- Dataset overview
- AI-generated assessment
- Recommended actions
- Data quality charts

This report can be shared with teams or stakeholders.

---

# Project Structure
├── app.py
├── requirements.txt
├── .gitignore
└── src
├── analyzer.py
├── agent.py
└── io_utils.py

---

# File Descriptions

## app.py

Main Streamlit application responsible for:

- File upload
- Goal selection
- Dataset evaluation
- OpenAI integration
- Dashboard rendering
- PDF generation

---

## src/io_utils.py

Handles dataset loading and basic overview calculations.

Functions include:

- CSV / Excel loading
- Dataset summary generation

---

## src/analyzer.py

Responsible for performing **data quality analysis**, including:

- Quality score calculation
- Missing value analysis
- Duplicate row detection
- Outlier detection
- Data type validation
- Column consistency checks

---

## src/agent.py

Provides logic for:

- Understanding the user's question
- Determining analysis focus
- Generating structured reports
- Producing executive summaries and recommendations

---

# How It Works

The system follows this workflow:

1. User uploads a dataset
2. User selects a goal
3. The dataset structure is analyzed
4. Data quality metrics are calculated
5. AI generates a contextual evaluation
6. Results are visualized
7. A PDF report can be generated

---

# Technologies Used

The project uses the following technologies:

- Python
- Streamlit
- Pandas
- Matplotlib
- OpenAI API
- ReportLab
- OpenPyXL

---

# Installation

Make sure you have **Python 3.10 or higher** installed.

---

## Clone the Repository

```bash
git clone https://github.com/your-username/ai-data-quality-analyst.git
```bash
cd ai-data-quality-analyst

## Create Virtual Environment

### macOS / Linux
```bash
python -m venv venv
source venv/bin/activate
```

### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

## Install Dependencies
```bash
pip install -r requirements.txt
```

## OpenAI API Setup

The application requires an OpenAI API key.

Set it as an environment variable.

### Linux / macOS
```bash
export OPENAI_API_KEY="your_api_key_here"
```

### Windows (PowerShell)
```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

## Run the Application

Start the Streamlit server:

```bash
streamlit run app.py
```

The application will open automatically in your browser.

## Example User Questions

Users can ask questions such as:

- Is this dataset suitable for machine learning?
- What issues exist in this dataset?
- Does the dataset need cleaning before analysis?
- Are there risks in using this dataset for reporting?
- What improvements should be made before model training?

## Data Quality Scoring Method

The readiness score is calculated using penalties based on:

- Missing value ratio  
- Duplicate row ratio  
- Outlier ratio  

Additional goal-based adjustments are applied depending on:

- Reporting  
- Analysis  
- Machine learning  

The score ranges from **0 to 100**.

## Typical Use Cases

This tool is useful for:

- Data analysts  
- Business intelligence teams  
- Machine learning engineers  
- Data governance teams  
- Organizations receiving external datasets  

It helps ensure datasets are evaluated before they are used in critical workflows.

## Future Improvements

Potential enhancements include:

- Automatic data cleaning suggestions  
- Feature importance analysis  
- Column relationship detection  
- Large dataset optimization  
- Database connections  
- Multi-file dataset comparison  
- Advanced anomaly detection  
- Data schema validation  

## .gitignore

The project ignores the following files:

```
__pycache__/
.env
*.pyc
.DS_Store
```

## License

You can choose any license depending on your needs.

Example:

MIT License

```
This project is licensed under the MIT License.
```

## Contributing

Contributions are welcome.

Possible contribution areas:

- Improving data quality scoring  
- Enhancing UI/UX  
- Adding new analysis modules  
- Improving report generation  
- Performance optimizations  

## Author

AI Data Quality Analyst Project

You may add your name here if publishing the project:

```
Developed by Your Name
```

## Summary

AI Data Quality Analyst is designed to help teams quickly answer an important question:

**Is this dataset ready for the task we want to perform?**

By combining automated analysis, visual insights, and AI-powered evaluation, the tool helps reduce risks and improve decision-making before working with data.
