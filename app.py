if submitted:
    with st.spinner("Analyzing dataset..."):
        try:
            final_question = question or "Assess the dataset for the selected goal."

            analysis_text = ask_gpt(
                question=final_question,
                structured=structured,
                goal_label=goal_label,
                recommendations=recommendations,
            )

            st.session_state.analysis_text = analysis_text
            st.session_state.question_submitted = True

            st.session_state.pdf_bytes = build_pdf_report(
                filename=uploaded_file.name,
                question=final_question,
                structured=structured,
                analysis_text=analysis_text,
                df=df,
                recommendations=recommendations,
                goal_label=goal_label,
            )

            st.success("Request sent successfully ✅")

        except Exception as exc:
            st.error(f"OpenAI request failed: {exc}")
