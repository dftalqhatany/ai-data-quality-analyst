import pandas as pd

def analyze_dataset(df, analysis_mode="Full Quality Audit"):
    rows, cols = df.shape
    total_cells = max(rows * cols, 1)

    missing_counts = df.isna().sum()
    duplicate_rows = int(df.duplicated().sum())

    numeric_df = df.select_dtypes(include=["number"])

    result = {
        "analysis_mode": analysis_mode,
        "rows": rows,
        "columns": cols,
    }

    if analysis_mode == "Missing Values":
        missing_by_column = [
            {"column": c, "missing": int(v)}
            for c, v in missing_counts.items() if v > 0
        ]

        result.update({
            "issue_summary": [
                {"issue_type": "Missing Values", "count": int(missing_counts.sum())}
            ],
            "missing_by_column": missing_by_column
        })
        return result

    elif analysis_mode == "Duplicate Rows":
        result.update({
            "issue_summary": [
                {"issue_type": "Duplicate Rows", "count": duplicate_rows}
            ],
            "duplicate_rows": duplicate_rows
        })
        return result

    elif analysis_mode == "Data Types Check":
        dtype_summary = {
            col: str(dtype) for col, dtype in df.dtypes.items()
        }

        mixed_type_columns = []
        for col in df.columns:
            non_null_types = df[col].dropna().map(type).astype(str).unique().tolist()
            if len(non_null_types) > 1:
                mixed_type_columns.append({
                    "column": col,
                    "types": non_null_types
                })

        result.update({
            "issue_summary": [
                {"issue_type": "Mixed Data Types", "count": len(mixed_type_columns)}
            ],
            "dtypes": dtype_summary,
            "mixed_type_columns": mixed_type_columns
        })
        return result

    elif analysis_mode == "Outlier Detection":
        outlier_summary = []

        for col in numeric_df.columns:
            q1 = numeric_df[col].quantile(0.25)
            q3 = numeric_df[col].quantile(0.75)
            iqr = q3 - q1

            if iqr == 0:
                outlier_count = 0
            else:
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outlier_count = int(((numeric_df[col] < lower) | (numeric_df[col] > upper)).sum())

            outlier_summary.append({
                "column": col,
                "outliers": outlier_count
            })

        result.update({
            "issue_summary": [
                {"issue_type": "Outliers", "count": sum(x["outliers"] for x in outlier_summary)}
            ],
            "outlier_summary": outlier_summary
        })
        return result

    elif analysis_mode == "Column Consistency":
        consistency_issues = []

        for col in df.select_dtypes(include=["object"]).columns:
            stripped = df[col].dropna().astype(str)

            leading_trailing_spaces = int((stripped != stripped.str.strip()).sum())
            case_variants = int(stripped.str.lower().nunique() != stripped.nunique())

            if leading_trailing_spaces > 0 or case_variants > 0:
                consistency_issues.append({
                    "column": col,
                    "leading_trailing_spaces": leading_trailing_spaces,
                    "case_variants_detected": case_variants
                })

        result.update({
            "issue_summary": [
                {"issue_type": "Column Consistency Issues", "count": len(consistency_issues)}
            ],
            "consistency_issues": consistency_issues
        })
        return result

    # Full Quality Audit
    missing_by_column = [
        {"column": c, "missing": int(v)}
        for c, v in missing_counts.items() if v > 0
    ]

    column_risks = []
    for col in df.columns:
        miss = int(missing_counts[col])
        if miss > 0:
            column_risks.append({
                "column": col,
                "risk": "missing values",
                "count": miss
            })

    missing_penalty = min(40, (missing_counts.sum() / total_cells) * 100)
    duplicate_penalty = min(20, (duplicate_rows / rows) * 100 if rows else 0)

    quality_score = int(max(0, 100 - missing_penalty - duplicate_penalty))

    if quality_score > 85:
        readiness = "Ready for AI"
    elif quality_score > 65:
        readiness = "Needs Cleaning"
    else:
        readiness = "High Risk Dataset"

    result.update({
        "quality_score": quality_score,
        "readiness": readiness,
        "issue_summary": [
            {"issue_type": "Missing Values", "count": int(missing_counts.sum())},
            {"issue_type": "Duplicate Rows", "count": duplicate_rows}
        ],
        "missing_by_column": missing_by_column,
        "column_risks": column_risks
    })

    return result
