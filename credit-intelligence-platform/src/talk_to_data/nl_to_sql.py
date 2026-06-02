def convert_to_sql(query):

    query = query.lower()

    if "default" in query:
        return "SELECT * FROM loans WHERE TARGET = 1"

    elif "safe" in query:
        return "SELECT * FROM loans WHERE TARGET = 0"

    elif "high income" in query:
        return "SELECT * FROM loans WHERE AMT_INCOME_TOTAL > 200000"

    else:
        return "SELECT * FROM loans LIMIT 10"


if __name__ == "__main__":
    print(convert_to_sql("show default customers"))