PROMPTS = {
    "sql_generator": "Convert natural language query into SQL for credit risk dataset.",
    "risk_explainer": "Explain why a customer is high or low risk.",
    "summary": "Summarize credit risk analysis results."
}


if __name__ == "__main__":
    print(PROMPTS["sql_generator"])