def select_features(df):

    columns_to_drop = [
        "Patient_ID",
        "Contact_Number"
    ]

    #columns_to_drop = [
    #    i for i in columns_to_drop
    #    if i in df.columns
    #]

    X = df.drop(columns_to_drop, axis=1)

    if "Cancer_Detected" in df.columns:

        y = df["Cancer_Detected"].map({
            "No": 0,
            "Yes": 1
        })

        return X, y

    return X