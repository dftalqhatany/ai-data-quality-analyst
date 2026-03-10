def build_analysis_context(user_goal, filename, analysis_mode, analysis):
    return {
        "user_goal": user_goal,
        "filename": filename,
        "analysis_mode": analysis_mode,
        "analysis": analysis
    }

def generate_final_report(context):
    analysis = context["analysis"]
    mode = context["analysis_mode"]

    if mode == "Missing Values":
        total_missing = 0
        for item in analysis.get("missing_by_column", []):
            total_missing += item["missing"]

        return {
            "executive_summary": f"Missing values analysis completed. Total missing values: {total_missing}.",
            "recommendations": ["Handle missing values using mean, median, mode, or row removal depending on business context."],
            "final_decision": "Review Missing Data"
        }

    elif mode == "Duplicate Rows":
        duplicates = analysis.get("duplicate_rows", 0)
        return {
            "executive_summary": f"Duplicate row analysis completed. Total duplicate rows: {duplicates}.",
            "recommendations": ["Remove duplicate rows before training or reporting."] if duplicates > 0 else ["No duplicate row issue detected."],
            "final_decision": "Review Duplicates"
        }

    elif mode == "Data Types Check":
        mixed = analysis.get("mixed_type_columns", [])
        return {
            "executive_summary": f"Data types analysis completed. Mixed-type columns found: {len(mixed)}.",
            "recommendations": ["Standardize column types before downstream use."] if mixed else ["Column data types look consistent."],
            "final_decision": "Review Data Types"
        }

    elif mode == "Outlier Detection":
        outliers = analysis.get("outlier_summary", [])
        total_outliers = sum(item["outliers"] for item in outliers)
        return {
            "executive_summary": f"Outlier detection completed. Total detected outliers: {total_outliers}.",
            "recommendations": ["Investigate extreme values and decide whether to cap, transform, or remove them."] if total_outliers > 0 else ["No significant outlier issue detected."],
            "final_decision": "Review Outliers"
        }

    elif mode == "Column Consistency":
        issues = analysis.get("consistency_issues", [])
        return {
            "executive_summary": f"Column consistency check completed. Columns with consistency issues: {len(issues)}.",
            "recommendations": ["Normalize spacing and text casing in categorical columns."] if issues else ["No major consistency issues detected."],
            "final_decision": "Review Consistency"
        }

    # Full Quality Audit
    score = analysis.get("quality_score", 0)
    readiness = analysis.get("readiness", "Unknown")

    summary = f"Dataset quality score is {score}/100. Status: {readiness}."

    recs = []

    if any(i["issue_type"] == "Missing Values" and i["count"] > 0 for i in analysis.get("issue_summary", [])):
        recs.append("Handle missing values using median or mean.")

    if any(i["issue_type"] == "Duplicate Rows" and i["count"] > 0 for i in analysis.get("issue_summary", [])):
        recs.append("Remove duplicate rows.")

    if not recs:
        recs.append("Dataset looks clean for modeling.")

    return {
        "executive_summary": summary,
        "recommendations": recs,
        "final_decision": readiness
    }
