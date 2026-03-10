
def build_analysis_context(user_goal,filename,analysis_mode,analysis):
    return {
        "user_goal":user_goal,
        "filename":filename,
        "analysis_mode":analysis_mode,
        "analysis":analysis
    }

def generate_final_report(context):

    analysis=context["analysis"]
    score=analysis["quality_score"]
    readiness=analysis["readiness"]

    summary=f"Dataset quality score is {score}/100. Status: {readiness}."

    recs=[]

    if any(i["issue_type"]=="Missing Values" and i["count"]>0 for i in analysis["issue_summary"]):
        recs.append("Handle missing values using median or mean.")
    if any(i["issue_type"]=="Duplicate Rows" and i["count"]>0 for i in analysis["issue_summary"]):
        recs.append("Remove duplicate rows.")

    if not recs:
        recs.append("Dataset looks clean for modeling.")

    return {
        "executive_summary":summary,
        "recommendations":recs,
        "final_decision":readiness
    }
