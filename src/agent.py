def infer_task_focus(user_goal: str) -> str:
    text = (user_goal or "").strip().lower()

    if not text:
        return "general"

    if any(word in text for word in ["null", "missing", "na", "nan", "empty"]):
        return "missing_values"

    if any(word in text for word in ["duplicate", "duplicated", "repeat", "repeated"]):
        return "duplicate_rows"

    if any(word in text for word in ["outlier", "extreme", "anomaly"]):
        return "outlier_detection"

    if any(word in text for word in ["type", "dtype", "format", "schema"]):
        return "data_types"

    if any(word in text for word in ["consistency", "case", "spaces", "text values"]):
        return "column_consistency"

    return "general"


def build_analysis_context(user_goal, filename, analysis_mode, analysis, task_focus="general"):
    return {
        "user_goal": user_goal,
        "filename": filename,
        "analysis_mode": analysis_mode,
        "analysis": analysis,
        "task_focus": task_focus,
    }


def generate_final_report(context):
    analysis = context["analysis"]
    mode = context["analysis_mode"]
    task_focus = context.get("task_focus", "general")

    if mode == "Missing Values":
        total_missing = analysis.get("total_missing_values", 0)
        return {
            "executive_summary": f"Missing values analysis completed. Total missing values: {total_missing}. Data quality score: {analysis.get('quality_score', 0)}/100.",
            "recommendations": [
                "Handle missing values using mean, median, mode, or row removal depending on business context."
            ],
            "final_decision": "Review Missing Data",
            "task_focus": task_focus
        }

    elif mode == "Duplicate Rows":
        duplicates = analysis.get("duplicate_rows", 0)
        return {
            "executive_summary": f"Duplicate row analysis completed. Total duplicate rows: {duplicates}. Data quality score: {analysis.get('quality_score', 0)}/100.",
            "recommendations": ["Remove duplicate rows before training or reporting."] if duplicates > 0 else ["No duplicate row issue detected."],
            "final_decision": "Review Duplicates",
            "task_focus": task_focus
        }

    elif mode == "Data Types Check":
        mixed = analysis.get("mixed_type_columns", [])
        return {
            "executive_summary": f"Data types analysis completed. Mixed-type columns found: {len(mixed)}. Data quality score: {analysis.get('quality_score', 0)}/100.",
            "recommendations": ["Standardize column types before downstream use."] if mixed else ["Column data types look consistent."],
            "final_decision": "Review Data Types",
            "task_focus": task_focus
        }

    elif mode == "Outlier Detection":
        outliers = analysis.get("outlier_summary", [])
        total_outliers = sum(item["outliers"] for item in outliers)
        return {
            "executive_summary": f"Outlier detection completed. Total detected outliers: {total_outliers}. Data quality score: {analysis.get('quality_score', 0)}/100.",
            "recommendations": ["Investigate extreme values and decide whether to cap, transform, or remove them."] if total_outliers > 0 else ["No significant outlier issue detected."],
            "final_decision": "Review Outliers",
            "task_focus": task_focus
        }

    elif mode == "Column Consistency":
        issues = analysis.get("consistency_issues", [])
        return {
            "executive_summary": f"Column consistency check completed. Columns with consistency issues: {len(issues)}. Data quality score: {analysis.get('quality_score', 0)}/100.",
            "recommendations": ["Normalize spacing and text casing in categorical columns."] if issues else ["No major consistency issues detected."],
            "final_decision": "Review Consistency",
            "task_focus": task_focus
        }

    score = analysis.get("quality_score", 0)
    readiness = analysis.get("readiness", "Unknown")

    recs = []

    total_missing = analysis.get("total_missing_values", 0)
    duplicate_rows = 0
    for item in analysis.get("issue_summary", []):
        if item["issue_type"] == "Duplicate Rows":
            duplicate_rows = item["count"]

    if total_missing > 0:
        recs.append("Handle missing values using median, mean, mode, or deletion based on business context.")

    if duplicate_rows > 0:
        recs.append("Remove duplicate rows before downstream analytics or model training.")

    if not recs:
        recs.append("Dataset looks reasonably clean for modeling.")

    task_note = "General quality audit performed."
    if task_focus == "missing_values":
        task_note = "User task indicates a focus on missing/null values."
    elif task_focus == "duplicate_rows":
        task_note = "User task indicates a focus on duplicate rows."
    elif task_focus == "outlier_detection":
        task_note = "User task indicates a focus on outlier detection."
    elif task_focus == "data_types":
        task_note = "User task indicates a focus on data type validation."
    elif task_focus == "column_consistency":
        task_note = "User task indicates a focus on categorical/text consistency."

    return {
        "executive_summary": f"Dataset quality score is {score}/100. Status: {readiness}.",
        "recommendations": recs,
        "final_decision": readiness,
        "task_focus": task_focus,
        "task_note": task_note
    }
