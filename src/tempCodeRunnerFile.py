    val = pd.read_csv("data/strict_time_balanced/val.csv")
    test = pd.read_csv("data/strict_time_balanced/test.csv")

    print(train["ts"].min(), train["ts"].max())
    print(val["ts"].min(), val["ts"].max())
    print(test["ts"].min(), test["ts"].max())