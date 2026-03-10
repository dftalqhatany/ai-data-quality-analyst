
import pandas as pd

def analyze_dataset(df, analysis_mode="Full Quality Audit"):

    rows, cols = df.shape
    total_cells = max(rows*cols,1)

    missing_counts = df.isna().sum()
    duplicate_rows = int(df.duplicated().sum())

    missing_by_column=[
        {"column":c,"missing":int(v)}
        for c,v in missing_counts.items() if v>0
    ]

    issue_summary=[
        {"issue_type":"Missing Values","count":len(missing_by_column)},
        {"issue_type":"Duplicate Rows","count":duplicate_rows}
    ]

    missing_penalty=min(40,(missing_counts.sum()/total_cells)*100)
    duplicate_penalty=min(20,(duplicate_rows/rows)*100 if rows else 0)

    quality_score=int(max(0,100-missing_penalty-duplicate_penalty))

    if quality_score>85:
        readiness="Ready for AI"
    elif quality_score>65:
        readiness="Needs Cleaning"
    else:
        readiness="High Risk Dataset"

    column_risks=[]
    for col in df.columns:
        miss=int(missing_counts[col])
        if miss>0:
            column_risks.append({
                "column":col,
                "risk":"missing values",
                "count":miss
            })

    return {
        "quality_score":quality_score,
        "readiness":readiness,
        "issue_summary":issue_summary,
        "column_risks":column_risks
    }
