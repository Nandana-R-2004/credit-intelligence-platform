def apply_rules(probability):

    if probability >= 0.85:
        decision = "REJECT"
        reason = "Very high default risk"

    elif probability >= 0.65:
        decision = "MANUAL REVIEW"
        reason = "Moderate risk"

    elif probability >= 0.40:
        decision = "APPROVE WITH CONDITIONS"
        reason = "Medium risk"

    else:
        decision = "APPROVE"
        reason = "Low risk"

    return {
        "decision": decision,
        "reason": reason,
        "risk_score": probability
    }


# TEST
if __name__ == "__main__":
    print(apply_rules(0.72))