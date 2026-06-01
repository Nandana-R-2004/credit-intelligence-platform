def risk_label(prob):

    if prob >= 0.8:
        return "HIGH RISK"
    elif prob >= 0.5:
        return "MEDIUM RISK"
    else:
        return "LOW RISK"