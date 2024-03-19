import pandas as pd
def load_dataset(datasetPath):
    usecols = ["class", "message", "c1", "c2", "c3"]

    # Load dataset
    data = pd.read_csv(datasetPath, encoding="utf-8", encoding_errors="replace", names=usecols)
    data = data.iloc[1:] # get rid of first row

    # TODO: Rewrite this, this is ugly
    # join messages, as they are split in excel
    messages = data[data.columns[1:]]
    tmp = messages["message"].astype(str) + messages['c1'].astype(str) + messages['c2'].astype(str) + messages['c3'].astype(str)

    # Create main dataset
    dataset = pd.DataFrame()
    dataset["messages"] = tmp
    dataset["classes"] = pd.factorize(data["class"])[0]


    return dataset